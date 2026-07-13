# =============================================================================
#  rhs.jl — §3: Re-architected UDE right-hand side
#
#  du/dt = k · (G · S_phys(RH, κ_eff) · regime · (1 + g_L3) − u)
#
#  where g_L3 = g_hyst(dRH/dt) + g_flow(wind)
#
#  Key design constraints:
#    k-floor:   k = max(|log_k|, 1) + 1e-3  — prevents ODE stiffness collapseunder
#               near-zero time constants; log_k is the trainable parameter
#    g_hyst:    odd function of dRH/dt, tanh-shaped, g_hyst(0) = 0
#               (multiplicative anchor — no offset at steady state)
#    g_flow:    binary gate at wind > W_THRESH (CONST — from teardown data,
#               not fitted, un-fittable by type exclusion)
#    S_phys:    bilinear table lookup; Zygote-differentiable w.r.t. κ_eff
#    regime:    per-site multiplicative factor exp(log_regime); default 1.0
#    u:         internal particle count state [dimensionless ch1 units]
#
#  Requires: s_phys.jl (interp_s_phys, SPHYSTable, PMS5003Design)
# =============================================================================


# ---------------------------------------------------------------------------
# §3-A  Correction terms (g_L3 components)
# ---------------------------------------------------------------------------

"""
    g_hyst(drhdt, a_hyst, τ_hyst) → correction ∈ (-1, 1)

RH-hysteresis correction term.  Odd function of dRH/dt — positive for rising
RH, negative for falling.  Multiplicatively anchored: g_hyst(0) = 0 so the
steady-state equilibrium point is unchanged.

drhdt  : dRH/dt [fraction/hour]  (negative → drying)
a_hyst : amplitude  (typical range ±0.10)
τ_hyst : time scale [fraction/hour]; clamped to ≥ 1e-3 to avoid ÷0
"""
@inline function g_hyst(drhdt::Real, a_hyst::Real, τ_hyst::Real)
    τ_safe = max(abs(τ_hyst), 1e-3)
    return a_hyst * tanh(drhdt / τ_safe)
end


"""
    g_flow(wind, a_flow, design) → correction

Binary flow-gate correction.  When wind speed exceeds the design threshold
W_THRESH, a multiplicative bias `a_flow` is applied (expected negative —
high wind reduces particle residence time in the sensing chamber).

`design.w_thresh` is a CONST from teardown measurements; it is excluded from
the optimisation parameter vector.
"""
@inline function g_flow(wind::Real, a_flow::Real, design::PMS5003Design)
    return a_flow * (wind > Float64(design.w_thresh) ? 1.0 : 0.0)
end


"""
    g_L3(drhdt, wind, a_hyst, τ_hyst, a_flow, design) → total correction

Combined Layer-3 correction.  Additive: g_hyst + g_flow.
"""
@inline function g_L3(drhdt::Real, wind::Real,
                       a_hyst::Real, τ_hyst::Real,
                       a_flow::Real, design::PMS5003Design)
    return g_hyst(drhdt, a_hyst, τ_hyst) + g_flow(wind, a_flow, design)
end


# ---------------------------------------------------------------------------
# §3-B  k-floor transform
# ---------------------------------------------------------------------------

"""
    k_from_logk(log_k) → k ≥ 1.001

Maps the trainable parameter log_k to the physical time constant k.
Floor ensures the ODE remains non-stiff regardless of log_k value.
Matches the convention in mie_physics.jl `make_dudt_mie`.
"""
@inline k_from_logk(log_k::Real) = max(abs(log_k), 1.0) + 1e-3


# ---------------------------------------------------------------------------
# §3-C  Per-unit RHS closure
# ---------------------------------------------------------------------------

"""
    CalibRHSParams

Flat parameter struct for a single unit's ODE right-hand side.
Distinct from the hierarchical CalibParams (§4) — this is what the ODE closure sees.
"""
Base.@kwdef struct CalibRHSParams
    κ_eff      :: Float64 = 0.20    # shared across units
    log_k      :: Float64 = 1.0     # time constant (shared or per-unit)
    log_G      :: Float64 = 7.0     # log gain: G = exp(log_G)  [ch1 counts at ref]
    log_regime :: Float64 = 0.0     # per-site regime factor: regime = exp(log_regime)
    a_hyst     :: Float64 = 0.0
    τ_hyst     :: Float64 = 0.05
    a_flow     :: Float64 = 0.0
end


"""
    make_calib_rhs(sphys_table, design, covariate_fn) → (u, p, t) → du

Build a UDE right-hand side closure for a single unit.

Arguments
---------
sphys_table   : SPHYSTable (precomputed)
design        : PMS5003Design (CONST)
covariate_fn  : t → (rh, wind, drhdt) — linear interpolant over the unit's data

Returns a function `rhs!(du, u, p, t)` where p::CalibRHSParams.
Suitable for DifferentialEquations.jl `ODEProblem(rhs!, u0, tspan, p)`.

Note: this is the *in-place* form for OrdinaryDiffEq; use `rhs(u, p, t)` form
(out-of-place) for Zygote differentation through the ODE solve.
"""
function make_calib_rhs(sphys_table::SPHYSTable,
                         design::PMS5003Design,
                         covariate_fn)
    function rhs!(du, u, p, t)
        rh, wind, drhdt = covariate_fn(t)
        k       = k_from_logk(p.log_k)
        G       = exp(p.log_G)
        regime  = exp(p.log_regime)
        S       = interp_s_phys(sphys_table, rh, clamp(p.κ_eff, 0.01, 0.60))
        g       = g_L3(drhdt, wind, p.a_hyst, p.τ_hyst, p.a_flow, design)
        du[1]   = k * (G * S * regime * (1.0 + g) - u[1])
    end
    return rhs!
end


"""
    make_calib_rhs_oop(sphys_table, design, covariate_fn) → (u, p, t) → SVector

Out-of-place variant for use with StaticArrays / Zygote.
Returns du as a 1-element SVector.
"""
function make_calib_rhs_oop(sphys_table::SPHYSTable,
                              design::PMS5003Design,
                              covariate_fn)
    function rhs(u, p, t)
        rh, wind, drhdt = covariate_fn(t)
        k      = k_from_logk(p.log_k)
        G      = exp(p.log_G)
        regime = exp(p.log_regime)
        S      = interp_s_phys(sphys_table, rh, clamp(p.κ_eff, 0.01, 0.60))
        g      = g_L3(drhdt, wind, p.a_hyst, p.τ_hyst, p.a_flow, design)
        return [k * (G * S * regime * (1.0 + g) - u[1])]
    end
    return rhs
end


# ---------------------------------------------------------------------------
# §3-D  Covariate interpolant builder
# ---------------------------------------------------------------------------

"""
    build_covariate_fn(t_days, rh_vec, wind_vec, drhdt_vec) → t → (rh, wind, drhdt)

Build a piecewise-linear covariate function from unit data.
Returns values at the nearest available time on extrapolation (clamp).
"""
function build_covariate_fn(t_days::AbstractVector,
                              rh_vec::AbstractVector,
                              wind_vec::AbstractVector,
                              drhdt_vec::AbstractVector)
    t_d = Float64.(t_days)
    rh  = Float64.(rh_vec)
    wi  = Float64.(wind_vec)
    dr  = Float64.(drhdt_vec)
    n   = length(t_d)

    function lerp1(xs, ys, xq)
        idx_f = searchsortedlast(xs, xq)
        idx   = clamp(idx_f, 1, n - 1)
        α     = (xq - xs[idx]) / (xs[idx+1] - xs[idx])
        α     = clamp(α, 0.0, 1.0)
        return ys[idx] * (1 - α) + ys[idx+1] * α
    end

    return t -> (lerp1(t_d, rh, t), lerp1(t_d, wi, t), lerp1(t_d, dr, t))
end
