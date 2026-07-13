# =============================================================================
#  fit.jl — §5: Two-stage calibration fit
#
#  Optimises CalibParams against multi-unit panel data by minimising the
#  conditional negative log-likelihood (NLL):
#
#    NLL(θ) = Σ_u Σ_t  0.5·(y_ut − û_ut(θ))²/σ_u²  +  log(σ_u)
#
#  where û_ut is the ODE forward solution and σ_u = exp(log_σ_u) is estimated
#  jointly (not fixed at 0.05 — it is a per-unit parameter in UnitParams).
#
#  Optimisation engine:
#    Production plan: ADAM via Optimisers.jl with Zygote adjoint through the
#                     ODE solve (DifferentialEquations sensitivity interface).
#    Scaffold impl:   ADAM with finite-difference gradients (35 × ODE calls per
#                     step) — demonstrates wiring without the AD overhead.
#    Switch:          Replace `_gradient_fd` with Zygote.gradient when ready.
#
#  Two-stage schedule:
#    Stage 1: κ_eff locked at init value; fit G, σ, k, g_L3 per-unit gains
#    Stage 2: release κ_eff; fine-tune all params jointly
#
#  Requires: parameters.jl, rhs.jl, s_phys.jl
#            OrdinaryDiffEq.jl
# =============================================================================

using Dates, Printf, Statistics
# OrdinaryDiffEq is a transitive dependency of UniversalDiffEq; import what we need:
using OrdinaryDiffEq: ODEProblem, solve, Tsit5, ReturnCode


# ---------------------------------------------------------------------------
# §5-A  Panel → per-unit data bundles
# ---------------------------------------------------------------------------

"""
    UnitData

Data for a single unit extracted from the panel DataFrame.
"""
struct UnitData
    unit_id   :: String
    site_id   :: String
    site_idx  :: Int
    t_days    :: Vector{Float64}   # time axis [days from panel epoch]
    ch1       :: Vector{Float64}   # observed raw counts
    rh        :: Vector{Float64}   # RH [fraction]
    wind      :: Vector{Float64}   # wind speed [m/s]
    drhdt     :: Vector{Float64}   # dRH/dt [fraction/hour]
    batch_idx :: Vector{Int}       # which batch (phase cohort) each row belongs to
end


"""
    extract_unit_data(df, site_order, batch_order) → Vector{UnitData}

Split a panel DataFrame into per-unit bundles.  Rows are sorted by timestamp
within each unit; time is converted to Float64 days from the earliest timestamp.
"""
function extract_unit_data(df, site_order::Vector{String},
                            batch_order::Vector{String})
    t_epoch  = minimum(df.timestamp)
    units_df = groupby(df, :unit_id)
    result   = UnitData[]

    for gdf in units_df
        rows   = sort(gdf, :timestamp)
        uid    = rows.unit_id[1]
        sid    = rows.site_id[1]
        s_idx  = findfirst(==(sid), site_order)
        isnothing(s_idx) && error("site $sid not in site_order")

        t_days = [Float64(Dates.value(r.timestamp - t_epoch) / 86400000) for r in eachrow(rows)]
        ch1    = Float64.(rows.ch1)
        rh     = Float64.(rows.rh)
        wind   = Float64.(rows.wind)
        drhdt  = Float64.(rows.drhdt)
        bidx   = [findfirst(==(r.phase), batch_order) for r in eachrow(rows)]
        bidx   = [isnothing(b) ? 1 : b for b in bidx]

        push!(result, UnitData(uid, sid, s_idx, t_days, ch1, rh, wind, drhdt, bidx))
    end
    return result
end


# ---------------------------------------------------------------------------
# §5-B  Forward ODE solve for one unit
# ---------------------------------------------------------------------------

"""
    solve_unit(ud, rhs_p, sphys_table, design) → ŷ::Vector{Float64}

Solve the calibration ODE for a single unit and return predicted ch1 at the
observed time points.

Uses Tsit5 (explicit Runge-Kutta 4/5) — appropriate for the moderately stiff
single-state ODE.  Absolute tolerance 1.0 (ch1 units), relative 1e-4.
"""
function solve_unit(ud::UnitData,
                    rhs_p::CalibRHSParams,
                    sphys_table::SPHYSTable,
                    design::PMS5003Design)
    cov_fn = build_covariate_fn(ud.t_days, ud.rh, ud.wind, ud.drhdt)
    rhs!   = make_calib_rhs(sphys_table, design, cov_fn)

    t0     = ud.t_days[1]
    tf     = ud.t_days[end]
    u0     = [ud.ch1[1]]

    prob   = ODEProblem(rhs!, u0, (t0, tf), rhs_p)
    sol    = solve(prob, Tsit5(),
                   saveat   = ud.t_days,
                   abstol   = 1.0,
                   reltol   = 1e-4,
                   maxiters = 10_000)

    if sol.retcode !== ReturnCode.Success
        return fill(NaN, length(ud.t_days))
    end
    return [u[1] for u in sol.u]
end


# ---------------------------------------------------------------------------
# §5-C  Conditional NLL
# ---------------------------------------------------------------------------

"""
    unit_nll(ud, rhs_p, batch_offsets, sphys_table, design) → NLL::Float64

Conditional negative log-likelihood for one unit.
σ_obs = exp(rhs_p.log_G is not σ; σ is passed separately)
"""
function unit_nll(ud::UnitData,
                  rhs_p::CalibRHSParams,
                  σ_obs::Float64,
                  batch_offsets::Vector{Float64},
                  sphys_table::SPHYSTable,
                  design::PMS5003Design)
    ŷ = solve_unit(ud, rhs_p, sphys_table, design)
    any(isnan, ŷ) && return 1e9

    n   = length(ud.ch1)
    nll = 0.0
    for i in 1:n
        b_off = batch_offsets[ud.batch_idx[i]]
        r     = (ud.ch1[i] - ŷ[i] - b_off) / σ_obs
        nll  += 0.5 * r^2
    end
    nll += n * log(σ_obs)
    return nll
end


"""
    total_nll(θ_vec, template, unit_data, sphys_table, design; lock_kappa) → NLL

Total NLL across all units.  `θ_vec` is the flat parameter vector produced by
`flatten(CalibParams; lock_kappa=...)`.
"""
function total_nll(θ_vec::AbstractVector,
                   template::CalibParams,
                   unit_data::Vector{UnitData},
                   sphys_table::SPHYSTable,
                   design::PMS5003Design;
                   lock_kappa::Bool = false)
    p = unflatten(θ_vec, template; lock_kappa = lock_kappa)

    nll_total = 0.0
    for (iu, ud) in enumerate(unit_data)
        rhs_p     = rhs_params_for_unit(p, iu, ud.site_idx)
        σ_obs     = exp(p.units[iu].log_σ)
        b_offsets = [b.offset for b in p.batches]
        nll_total += unit_nll(ud, rhs_p, σ_obs, b_offsets, sphys_table, design)
    end
    return nll_total
end


# ---------------------------------------------------------------------------
# §5-D  Finite-difference gradient (scaffold; swap for Zygote in production)
# ---------------------------------------------------------------------------

"""
    _gradient_fd(f, θ; ε) → ∇f::Vector{Float64}

Centred finite-difference gradient of scalar function f at θ.
ε = 1e-5 is appropriate for Float64 functions with ΔNll ≈ O(1–100).
Production path: replace this with Zygote.gradient(f, θ)[1].
"""
function _gradient_fd(f, θ::AbstractVector; ε::Float64 = 1e-5)
    n  = length(θ)
    g  = zeros(Float64, n)
    θ_ = copy(θ)
    for i in 1:n
        θ_[i]  = θ[i] + ε
        fp     = f(θ_)
        θ_[i]  = θ[i] - ε
        fm     = f(θ_)
        g[i]   = (fp - fm) / (2ε)
        θ_[i]  = θ[i]
    end
    return g
end


# ---------------------------------------------------------------------------
# §5-E  ADAM optimiser (manual — no Optimisers.jl dependency)
# ---------------------------------------------------------------------------

"""
    adam_step!(θ, g, m, v, t; lr, β1, β2, ε_adam) → nothing

In-place ADAM update.  Mirrors Optimisers.Adam exactly.
Production: replace with Optimisers.Adam rule applied to a ComponentArray.
"""
function adam_step!(θ::Vector{Float64}, g::Vector{Float64},
                    m::Vector{Float64}, v::Vector{Float64}, t::Int;
                    lr    = 1e-3,
                    β1    = 0.9,
                    β2    = 0.999,
                    ε_adam= 1e-8)
    for i in eachindex(θ)
        m[i] = β1 * m[i] + (1 - β1) * g[i]
        v[i] = β2 * v[i] + (1 - β2) * g[i]^2
        m̂    = m[i] / (1 - β1^t)
        v̂    = v[i] / (1 - β2^t)
        θ[i] -= lr * m̂ / (sqrt(v̂) + ε_adam)
    end
end


# ---------------------------------------------------------------------------
# §5-F  Two-stage fit
# ---------------------------------------------------------------------------

"""
    fit_stage1(init_params, unit_data, sphys_table, design;
               n_epochs, lr, verbose) → CalibParams

Stage 1: κ_eff is LOCKED; fit all other parameters.
Objective: get gain, noise, and g_L3 corrections right before releasing the
physics parameter κ_eff.
"""
function fit_stage1(init_params::CalibParams,
                    unit_data::Vector{UnitData},
                    sphys_table::SPHYSTable,
                    design::PMS5003Design;
                    n_epochs :: Int = 100,
                    lr       :: Float64 = 5e-3,
                    verbose  :: Bool = true)
    θ, _ = flatten(init_params; lock_kappa = true)
    m    = zeros(length(θ)); v = zeros(length(θ))

    loss_fn = θv -> total_nll(θv, init_params, unit_data, sphys_table, design;
                              lock_kappa = true)

    verbose && println("── Stage 1 (κ_eff locked = $(init_params.shared.κ_eff)) ──")
    for ep in 1:n_epochs
        g   = _gradient_fd(loss_fn, θ)
        adam_step!(θ, g, m, v, ep; lr = lr)
        if verbose && mod(ep, max(1, n_epochs ÷ 5)) == 0
            nll = loss_fn(θ)
            @printf "  Stage1 epoch %3d/%d  NLL = %.3e\n" ep n_epochs nll
        end
    end

    return unflatten(θ, init_params; lock_kappa = true)
end


"""
    fit_stage2(stage1_params, unit_data, sphys_table, design;
               n_epochs, lr, verbose) → CalibParams

Stage 2: release κ_eff and fine-tune all parameters jointly.
Initialised from Stage-1 solution.
"""
function fit_stage2(stage1_params::CalibParams,
                    unit_data::Vector{UnitData},
                    sphys_table::SPHYSTable,
                    design::PMS5003Design;
                    n_epochs :: Int = 150,
                    lr       :: Float64 = 2e-3,
                    verbose  :: Bool = true)
    θ, _ = flatten(stage1_params; lock_kappa = false)
    m    = zeros(length(θ)); v = zeros(length(θ))

    loss_fn = θv -> total_nll(θv, stage1_params, unit_data, sphys_table, design;
                              lock_kappa = false)

    verbose && println("── Stage 2 (κ_eff active) ──")
    for ep in 1:n_epochs
        g   = _gradient_fd(loss_fn, θ)
        adam_step!(θ, g, m, v, ep; lr = lr)
        if verbose && mod(ep, max(1, n_epochs ÷ 5)) == 0
            nll = loss_fn(θ)
            @printf("  Stage2 epoch %3d/%d  NLL = %.3e  κ_eff = %.4f\n",
                ep, n_epochs, nll, θ[1])   # θ[1] = κ_eff in Stage-2 layout
        end
    end

    return unflatten(θ, stage1_params; lock_kappa = false)
end


"""
    two_stage_fit(init_params, unit_data, sphys_table, design; kwargs...) → CalibParams

Run the full two-stage calibration.  Returns the final fitted CalibParams.
"""
function two_stage_fit(init_params::CalibParams,
                        unit_data::Vector{UnitData},
                        sphys_table::SPHYSTable,
                        design::PMS5003Design;
                        s1_epochs:: Int = 100,
                        s2_epochs:: Int = 150,
                        s1_lr    :: Float64 = 5e-3,
                        s2_lr    :: Float64 = 2e-3,
                        verbose  :: Bool = true)
    p1 = fit_stage1(init_params, unit_data, sphys_table, design;
                    n_epochs = s1_epochs, lr = s1_lr, verbose = verbose)
    p2 = fit_stage2(p1, unit_data, sphys_table, design;
                    n_epochs = s2_epochs, lr = s2_lr, verbose = verbose)
    return p2
end
