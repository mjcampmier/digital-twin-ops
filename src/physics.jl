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
kappa    hygroscopicity parameter; literature range 0.01 (soot/dust) – 0.9 (sea salt);
         0.20 is a mid-range value for mixed urban accumulation mode.
p_scat   exponent in f(RH) = (GF(RH)/GF_dry)^p_scat; fit by least squares to the
         full Mie+kappa-Koehler curve (R²=0.996 at kappa=0.20, Dg=0.15 µm, σg=1.60).
         Sensitive to assumed size distribution and refractive index (see p_scat.jl).
rh_dry   reference RH for the dry-state calibration anchor.  f(rh_dry) ≡ 1 by
         construction; shifts what the fitted C0 actually represents.
cal0015  2022 empirical b_sp1/CH1 factor [Mm⁻¹/count].  Used only as an outside
         sanity check, not in the ODE.
"""
Base.@kwdef struct PhysicsParams
    kappa::Float64   = 0.20
    p_scat::Float64  = 1.2466
    rh_dry::Float64  = 0.20
    cal0015::Float64 = 0.015
end

# -----------------------------------------------------------------------------
# PHYSICS FUNCTIONS
# -----------------------------------------------------------------------------

"""
kappa-Koehler diameter growth factor GF = Dp_wet/Dp_dry.

Kelvin (surface-tension) term is neglected.  For Dp > 0.1 µm the Kelvin
correction to water activity is <2%; for the scattering-dominant range
(0.2–1 µm) it is <1%.  See src/sensitivity/kelvin.jl for the quantitative
perturbative argument.
"""
@inline function growth_factor(rh::Float64, pp::PhysicsParams)
    aw = clamp(rh, 0.0, 0.985)
    return (1.0 + pp.kappa * aw / (1.0 - aw))^(1.0 / 3.0)
end

"""
Humidification factor f(RH) = b_obs(RH) / b_obs(rh_dry).

Cheap surrogate for the full Mie+kappa-Koehler integral:
    f(RH) = (GF(RH) / GF(rh_dry))^p_scat

Recomputes GF_dry on every call; use make_dudt() to get an ODE RHS
with GF_dry pre-computed (avoids redundant evaluation in the hot path).
"""
@inline function f_RH(rh::Float64, pp::PhysicsParams)
    gf_d = growth_factor(pp.rh_dry, pp)
    return (growth_factor(rh, pp) / gf_d)^pp.p_scat
end

"Physics-predicted CH1 at given RH and dry-state calibration constant C0."
@inline CH1_phys(rh::Float64, C0::Float64, pp::PhysicsParams) = C0 * f_RH(rh, pp)

"""
Build the UDE right-hand-side  dudt(u, x, p, t)  for a given PhysicsParams.

GF_dry is pre-computed once so it is NOT re-evaluated inside the ODE hot path
under ForwardDiff.  `residual_fn(u, x, p)` closes over the NN and must be
defined before calling make_dudt.

Covariate layout in x (set by the column order of the X DataFrame):
    x[1] = RH          (fraction)
    x[2] = T           (°C)
    x[3] = wind        (m/s)
    x[4] = t_deploy    (years since deployment)
    x[5] = dRHdt       (RH/day)
"""
function make_dudt(pp::PhysicsParams, residual_fn)
    gf_dry_val = growth_factor(pp.rh_dry, pp)
    function dudt(u, x, p, t)
        k  = abs(p.log_k) + 1e-3
        C0 = abs(p.C0)
        gf = growth_factor(Float64(x[1]), pp)
        known = k * (C0 * (gf / gf_dry_val)^pp.p_scat - u[1])
        return [known + residual_fn(u, x, p)]
    end
    return dudt
end

# -----------------------------------------------------------------------------
# COMPARISON UTILITY  (used by sensitivity scripts)
# -----------------------------------------------------------------------------

"""
Print an f(RH) comparison table for a list of PhysicsParams scenarios.
All columns are shown as % deviation from pp_list[baseline_idx].
"""
function print_f_rh_table(pp_list, labels;
                           rh_pts      = (0.40, 0.60, 0.80, 0.90),
                           baseline_idx = 1)
    n = length(pp_list)
    # header
    @printf "%-5s" "RH"
    for l in labels; @printf "  %-22s" l; end
    println()
    println("-"^(5 + 24 * n))
    for rh in rh_pts
        vals = [f_RH(rh, pp) for pp in pp_list]
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
