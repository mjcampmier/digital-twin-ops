# =============================================================================
#  parameters.jl — §4: Hierarchical calibration parameter structs
#
#  Hierarchy:
#    SharedParams  — one value across all units (κ_eff, log_k, g_L3 shape)
#    SiteParams    — per-site aerosol regime (multiplicative factor on S_phys)
#    UnitParams    — per-unit gain G and observation noise σ_obs
#    BatchParams   — per-batch (phase/cohort) additive offset for defect units
#
#  CalibParams aggregates all four levels.
#
#  Flat-vector encoding:
#    `flatten(p::CalibParams)` → Vector{Float64} for optimisation
#    `unflatten(θ, template)` → CalibParams
#
#  Two-stage schedule masks:
#    Stage 1: κ_eff locked  → shared[κ_eff] excluded from active set
#    Stage 2: all active
#
#  Requires: nothing (pure parameter algebra; no emulator dependencies)
# =============================================================================


# ---------------------------------------------------------------------------
# §4-A  Typed parameter structs
# ---------------------------------------------------------------------------

"""
    SharedParams

Parameters shared across all units and sites.

κ_eff   : lumped RH-response (NOT raw aerosol hygroscopicity; see physics.jl)
log_k   : time constant transform; k = max(|log_k|, 1) + 1e-3
a_hyst  : RH-hysteresis amplitude (signed; 0 = no hysteresis)
τ_hyst  : hysteresis time scale [fraction/hour]; optimised as log(τ_hyst)
"""
Base.@kwdef struct SharedParams
    κ_eff  :: Float64 = 0.20
    log_k  :: Float64 = 1.0
    a_hyst :: Float64 = 0.00
    τ_hyst :: Float64 = 0.05   # stored as raw value; log(τ_hyst) in flat vec
end

const N_SHARED = 4


"""
    SiteParams

Per-site aerosol regime factor.
log_regime = 0 → neutral (S_phys used directly)
log_regime > 0 → site scatters more than baseline PSD
"""
Base.@kwdef struct SiteParams
    log_regime :: Float64 = 0.0
end

const N_SITE = 1


"""
    UnitParams

Per-unit sensor parameters.
log_G   : log of gain (ch1 signal at reference conditions); G = exp(log_G)
log_σ   : log of observation noise std; σ_obs = exp(log_σ)
a_flow  : flow-gate amplitude (expected negative; wind-induced dilution)
"""
Base.@kwdef struct UnitParams
    log_G  :: Float64 = 7.5    # exp(7.5) ≈ 1800, near C0_mean
    log_σ  :: Float64 = 4.5    # exp(4.5) ≈ 90 counts, ~3% of C0
    a_flow :: Float64 = 0.0
end

const N_UNIT = 3


"""
    BatchParams

Per-batch (deployment phase / defect cohort) additive offset.
Captures systematic level shifts in defect cohorts C5-C7.
offset = 0 for healthy cohorts.
"""
Base.@kwdef struct BatchParams
    offset :: Float64 = 0.0
end

const N_BATCH = 1


"""
    CalibParams

Full hierarchical calibration parameter set.

Fields
------
shared  : SharedParams
sites   : Vector{SiteParams}, length n_sites
units   : Vector{UnitParams}, length n_units
batches : Vector{BatchParams}, length n_batches
"""
struct CalibParams
    shared  :: SharedParams
    sites   :: Vector{SiteParams}
    units   :: Vector{UnitParams}
    batches :: Vector{BatchParams}
end

function CalibParams(; n_sites::Int, n_units::Int, n_batches::Int = 1,
                       shared  = SharedParams(),
                       C0_mean = 2600.0)
    sites   = [SiteParams()   for _ in 1:n_sites]
    units   = [UnitParams(log_G = log(C0_mean)) for _ in 1:n_units]
    batches = [BatchParams()  for _ in 1:n_batches]
    return CalibParams(shared, sites, units, batches)
end


# ---------------------------------------------------------------------------
# §4-B  Flat-vector encoding/decoding
# ---------------------------------------------------------------------------

"""
    flatten(p::CalibParams; lock_kappa=false) → (θ::Vector{Float64}, mask)

Serialise CalibParams to a flat optimisation vector θ.
If `lock_kappa=true` (Stage-1 schedule), κ_eff is EXCLUDED from θ.

Returns (θ, n_shared_active) where n_shared_active is 3 (Stage-1) or 4 (Stage-2).
"""
function flatten(p::CalibParams; lock_kappa::Bool = false)
    shared_vals = if lock_kappa
        [p.shared.log_k, p.shared.a_hyst, log(max(p.shared.τ_hyst, 1e-4))]
    else
        [p.shared.κ_eff, p.shared.log_k, p.shared.a_hyst, log(max(p.shared.τ_hyst, 1e-4))]
    end

    site_vals  = [s.log_regime for s in p.sites]
    unit_vals  = vcat([[u.log_G, u.log_σ, u.a_flow] for u in p.units]...)
    batch_vals = [b.offset for b in p.batches]

    θ = vcat(shared_vals, site_vals, unit_vals, batch_vals)
    n_shared_active = length(shared_vals)
    return θ, n_shared_active
end


"""
    unflatten(θ, template; lock_kappa=false) → CalibParams

Reconstruct a CalibParams from a flat vector θ.
If `lock_kappa=true`, κ_eff is taken from `template.shared.κ_eff` (unchanged).
"""
function unflatten(θ::AbstractVector, template::CalibParams;
                   lock_kappa::Bool = false)
    n_sites   = length(template.sites)
    n_units   = length(template.units)
    n_batches = length(template.batches)
    n_shared  = lock_kappa ? 3 : 4

    idx = 1
    if lock_kappa
        log_k   = θ[idx];   idx += 1
        a_hyst  = θ[idx];   idx += 1
        τ_hyst  = exp(θ[idx]); idx += 1
        κ_eff   = template.shared.κ_eff   # locked
    else
        κ_eff   = θ[idx];   idx += 1
        log_k   = θ[idx];   idx += 1
        a_hyst  = θ[idx];   idx += 1
        τ_hyst  = exp(θ[idx]); idx += 1
    end
    shared = SharedParams(κ_eff = κ_eff, log_k = log_k,
                          a_hyst = a_hyst, τ_hyst = τ_hyst)

    sites = SiteParams[]
    for _ in 1:n_sites
        push!(sites, SiteParams(log_regime = θ[idx])); idx += 1
    end

    units = UnitParams[]
    for _ in 1:n_units
        log_G  = θ[idx]; idx += 1
        log_σ  = θ[idx]; idx += 1
        a_flow = θ[idx]; idx += 1
        push!(units, UnitParams(log_G = log_G, log_σ = log_σ, a_flow = a_flow))
    end

    batches = BatchParams[]
    for _ in 1:n_batches
        push!(batches, BatchParams(offset = θ[idx])); idx += 1
    end

    return CalibParams(shared, sites, units, batches)
end


# ---------------------------------------------------------------------------
# §4-C  Construct per-unit CalibRHSParams from CalibParams
# ---------------------------------------------------------------------------

"""
    rhs_params_for_unit(p, unit_idx, site_idx) → CalibRHSParams

Extract a flat CalibRHSParams for a single unit from the hierarchical CalibParams.
Used to construct per-unit RHS closures.
"""
function rhs_params_for_unit(p::CalibParams, unit_idx::Int, site_idx::Int)
    u = p.units[unit_idx]
    s = p.sites[site_idx]
    return CalibRHSParams(
        κ_eff      = p.shared.κ_eff,
        log_k      = p.shared.log_k,
        log_G      = u.log_G,
        log_regime = s.log_regime,
        a_hyst     = p.shared.a_hyst,
        τ_hyst     = p.shared.τ_hyst,
        a_flow     = u.a_flow,
    )
end


# ---------------------------------------------------------------------------
# §4-D  Summary print
# ---------------------------------------------------------------------------

function Base.show(io::IO, p::CalibParams)
    n_u = length(p.units); n_s = length(p.sites); n_b = length(p.batches)
    println(io, "CalibParams ($n_s sites, $n_u units, $n_b batches)")
    k_val = max(abs(p.shared.log_k), 1.0) + 1e-3
    @printf(io, "  shared: κ_eff=%.4f  k=%.3f  a_hyst=%.4f  τ_hyst=%.4f\n",
        p.shared.κ_eff, k_val, p.shared.a_hyst, p.shared.τ_hyst)
    for (i, s) in enumerate(p.sites)
        @printf io "  site[%d]: regime=%.4f\n" i exp(s.log_regime)
    end
    for (i, u) in enumerate(p.units)
        @printf(io, "  unit[%d]: G=%.1f  σ_obs=%.1f  a_flow=%.4f\n",
            i, exp(u.log_G), exp(u.log_σ), u.a_flow)
    end
end
