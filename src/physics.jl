# =============================================================================
#  physics.jl — PMS5003 deterministic transfer function
#
#  All defended scalar assumptions live in PhysicsParams.  Sensitivity scripts
#  swap one field at a time; the ODE RHS is built via make_dudt() so it closes
#  over the chosen params with GF_dry pre-computed (not re-evaluated on every
#  ODE call under ForwardDiff).
#
#  No heavy dependencies here — this file can be included by lightweight
#  sensitivity scripts without pulling in UniversalDiffEq / Lux.
# =============================================================================
using Printf

# -----------------------------------------------------------------------------
# DEFENDED ASSUMPTIONS  (see src/sensitivity/ for individual challenges)
# -----------------------------------------------------------------------------
"""
Every scalar assumption that can be individually challenged.
Defaults = validated baseline from physics_check.py (Mie + kappa-Koehler).

Fields
------
p_scat   exponent in f(RH) = (GF(RH)/GF_dry)^p_scat.  Fixed at 1.25 — the
         Mie+κ-Köhler–validated value (R²=0.996, Dg=0.15 µm, σg=1.60).  NOT
         fitted: κ_eff and p_scat trade off along a degenerate ridge; fixing
         p_scat makes kappa_eff the single identifiable RH-response knob.
rh_dry   reference RH for the dry-state calibration anchor.  f(rh_dry) ≡ 1 by
         construction for any kappa_eff (same κ in numerator and denominator
         of the GF ratio), so C0↔level separation is untouched.
cal0015  2022 empirical b_sp1/CH1 factor [Mm⁻¹/count].  Sanity check only.

NOTE: kappa is no longer a PhysicsParams field.  It is the fitted scalar
kappa_eff in the UDE parameter set p.  kappa_eff is a lumped RH-response
parameter — it absorbs aerosol growth, sensor optics, refractive-index change
on humidification, and the fixed-p_scat approximation.  It is NOT aerosol
hygroscopicity.  Always log and report it as kappa_eff.
"""
Base.@kwdef struct PhysicsParams
    p_scat::Float64  = 1.25
    rh_dry::Float64  = 0.20
    cal0015::Float64 = 0.015
end

# -----------------------------------------------------------------------------
# PHYSICS FUNCTIONS
# -----------------------------------------------------------------------------

"""
kappa-Koehler diameter growth factor GF = Dp_wet/Dp_dry.

Takes kappa_eff::Real so it accepts both Float64 and AD dual numbers.
Kelvin (surface-tension) term neglected (<1% for the scattering-dominant
size range 0.2–1 µm; see src/sensitivity/kelvin.jl).
"""
@inline function growth_factor(rh::Float64, kappa_eff::Real)
    aw = clamp(rh, 0.0, 0.985)
    return (1.0 + kappa_eff * aw / (1.0 - aw))^(1.0 / 3.0)
end

"""
Humidification factor f(RH) = b_obs(RH) / b_obs(rh_dry).

    f(RH) = (GF(RH, kappa_eff) / GF(rh_dry, kappa_eff))^p_scat

kappa_eff is the lumped fitted RH-response scalar (not raw aerosol κ).
f(rh_dry) ≡ 1 for any kappa_eff by construction.
"""
@inline function f_RH(rh::Float64, kappa_eff::Real, pp::PhysicsParams)
    gf_d = growth_factor(pp.rh_dry, kappa_eff)
    return (growth_factor(rh, kappa_eff) / gf_d)^pp.p_scat
end

"Physics-predicted CH1 at given RH, C0, and lumped kappa_eff."
@inline CH1_phys(rh::Float64, C0::Float64, kappa_eff::Real, pp::PhysicsParams) =
    C0 * f_RH(rh, kappa_eff, pp)

"""
Build the UDE right-hand-side  dudt(u, x, p, t)  for a given PhysicsParams.

The NN enters as a dry-anchored multiplicative correction g(x, p):
    du/dt = k · (C0 · f(RH) · (1 + g(x, p)) − u)
where g is supplied by the caller and must equal zero at the dry calibration
state so that C0 carries the level and g carries only fractional departures.
The fixed point is unique and analytic:
    u*(x) = C0 · f(RH) · (1 + g(x))  — linear in u, no bisection needed.

GF_dry is pre-computed once so it is NOT re-evaluated inside the ODE hot path
under ForwardDiff.

Covariate layout in x (set by the column order of the X DataFrame):
    x[1] = RH          (fraction)
    x[2] = T           (°C)
    x[3] = wind        (m/s)
    x[4] = t_deploy    (years since deployment)
    x[5] = dRHdt       (RH/day)
"""
function make_dudt(pp::PhysicsParams, g_fn)
    # gf_dry cannot be pre-computed: kappa_eff is a fitted parameter, so AD
    # (ForwardDiff / Zygote) must see both GF calls inside the hot path.
    function dudt(u, x, p, t)
        k       = max(abs(p.log_k), 1.0) + 1e-3
        C0      = abs(p.C0)
        kappa_e = abs(p.kappa_eff) + 1e-3
        gf      = growth_factor(Float64(x[1]), kappa_e)
        gf_dry  = growth_factor(pp.rh_dry, kappa_e)
        f_rh    = (gf / gf_dry)^pp.p_scat
        return [k * (C0 * f_rh * (1.0 + g_fn(x, p)) - u[1])]
    end
    return dudt
end

"""
Stage-1 ODE RHS with kappa_eff locked to a constant.

kappa_eff is still present in p (so the parameter vector has the right shape),
but the ODE uses kappa_lock directly — gradient w.r.t. p.kappa_eff is zero and
ADAM leaves it unchanged.  After stage-1 convergence, rebuild with make_dudt to
release kappa_eff for stage-2 fine-tuning.
"""
function make_dudt_locked(pp::PhysicsParams, g_fn, kappa_lock::Float64)
    gf_dry_locked = growth_factor(pp.rh_dry, kappa_lock)   # constant, safe to pre-compute
    function dudt(u, x, p, t)
        k    = max(abs(p.log_k), 1.0) + 1e-3
        C0   = abs(p.C0)
        gf   = growth_factor(Float64(x[1]), kappa_lock)
        f_rh = (gf / gf_dry_locked)^pp.p_scat
        return [k * (C0 * f_rh * (1.0 + g_fn(x, p)) - u[1])]
    end
    return dudt
end

# -----------------------------------------------------------------------------
# COMPARISON UTILITY  (used by sensitivity scripts)
# -----------------------------------------------------------------------------

"""
Print an f(RH) comparison table across a list of (pp, kappa_eff) scenarios.
All columns shown as % deviation from scenarios[baseline_idx].

scenarios  — vector of (PhysicsParams, kappa_eff::Float64) tuples
labels     — display name for each scenario
"""
function print_f_rh_table(scenarios, labels;
                           rh_pts       = (0.40, 0.60, 0.80, 0.90),
                           baseline_idx = 1)
    n = length(scenarios)
    @printf "%-5s" "RH"
    for l in labels; @printf "  %-22s" l; end
    println()
    println("-"^(5 + 24 * n))
    for rh in rh_pts
        vals = [f_RH(rh, ke, pp) for (pp, ke) in scenarios]
        ref  = vals[baseline_idx]
        @printf "%3.0f%%" rh * 100
        for (i, v) in enumerate(vals)
            if i == baseline_idx
                @printf "  %6.3f (baseline)      " v
            else
                @printf "  %6.3f (%+6.2f%%)       " v (v / ref - 1.0) * 100.0
            end
        end
        println()
    end
end
