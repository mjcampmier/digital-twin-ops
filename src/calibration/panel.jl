# =============================================================================
#  panel.jl — §0: Multi-unit panel schema, validation, and synthetic generator
#
#  The data contract (frozen; do not alter without coordinating with Mark's
#  Python loader):
#
#    unit_id    ::String          unique sensor identifier
#    site_id    ::String          deployment site
#    timestamp  ::DateTime (UTC)  hourly/daily observation time
#    phase      ::String          deployment phase label (C1…C9)
#    ch1        ::Float64         raw particle counts
#    deploy_age ::Float64         days since unit was deployed
#    bam_pm     ::Union{Float64,Missing}  µg/m³ from BAM at reference sites only
#    rh         ::Float64         relative humidity as FRACTION 0–1 (NOT percent)
#    temp       ::Float64         air temperature [°C]
#    wind       ::Float64         wind speed [m/s], ERA5-imputed
#    drhdt      ::Float64         dRH/dt [fraction/hour]
#
#  RH IS A FRACTION.  Timestamps ARE UTC.  These two bite silently.
# =============================================================================

using DataFrames, Dates, Statistics, Printf, Random

# ---------------------------------------------------------------------------
# §0-A  Schema constants
# ---------------------------------------------------------------------------

const PANEL_REQUIRED_COLS = [
    :unit_id, :site_id, :timestamp, :phase,
    :ch1, :deploy_age, :rh, :temp, :wind, :drhdt,
]
const PANEL_OPTIONAL_COLS = [:bam_pm]   # Missing allowed at non-reference sites


# ---------------------------------------------------------------------------
# §0-B  validate_panel — call at every load; hard error on violation
# ---------------------------------------------------------------------------

"""
    validate_panel(df::DataFrame) → nothing  (throws on violation)

Checks:
  1. All required columns present with correct eltype
  2. RH ∈ [0, 1]  (fraction, not %)
  3. No NaN in required numeric columns
  4. bam_pm is Union{Float64,Missing} if present
"""
function validate_panel(df::DataFrame)
    # 1. Required columns exist
    for col in PANEL_REQUIRED_COLS
        col ∈ propertynames(df) || error("validate_panel: missing column $col")
    end

    # 2. String columns
    for col in (:unit_id, :site_id, :phase)
        eltype(df[!, col]) <: AbstractString ||
            error("validate_panel: $col must be String, got $(eltype(df[!, col]))")
    end

    # 3. DateTime column
    eltype(df.timestamp) <: DateTime ||
        error("validate_panel: timestamp must be DateTime (UTC implied), got $(eltype(df.timestamp))")

    # 4. Float64 numeric columns
    for col in (:ch1, :deploy_age, :rh, :temp, :wind, :drhdt)
        eltype(df[!, col]) <: Real ||
            error("validate_panel: $col must be numeric, got $(eltype(df[!, col]))")
    end

    # 5. RH is fraction 0–1
    rh_min, rh_max = extrema(df.rh)
    rh_max <= 1.005 || error("validate_panel: rh max = $rh_max > 1 — RH must be FRACTION (0–1), not percent")
    rh_min >= -0.005 || error("validate_panel: rh min = $rh_min < 0 — impossible RH")

    # 6. No NaN in required numeric columns
    for col in (:ch1, :deploy_age, :rh, :temp, :wind, :drhdt)
        any(isnan, df[!, col]) && error("validate_panel: NaN found in $col")
    end

    # 7. bam_pm type check (if present)
    if :bam_pm ∈ propertynames(df)
        T = eltype(df.bam_pm)
        (T <: Union{Float64, Missing} || T <: Float64 || T <: Missing) ||
            error("validate_panel: bam_pm must be Union{Float64,Missing}, got $T")
    end

    nothing
end


# ---------------------------------------------------------------------------
# §0-C  synthetic_panel — exact schema, multi-unit/multi-site
# ---------------------------------------------------------------------------

"""
    synthetic_panel(; n_units, n_sites, n_phases, n_days, seed) → DataFrame

Generate a synthetic multi-unit panel in the exact §0 schema.  One BAM unit
per site is included (bam_pm non-missing).

Parameters
----------
n_units   : total sensor units (distributed across sites)
n_sites   : number of deployment sites
n_phases  : deployment phases (C1…CK)
n_days    : days per unit (before random dropout / stagger)
kappa_true: true lumped κ_eff used in the synthetic RH signal
C0_unit_sd: unit-to-unit C0 spread (fraction of mean C0)
seed      : random seed

Returns
-------
DataFrame with columns matching the §0 schema verbatim.
"""
function synthetic_panel(;
    n_units       = 8,
    n_sites       = 3,
    n_phases      = 4,
    n_days        = 120,
    kappa_true    = 0.25,
    a_flow_true   = -0.12,
    w_thresh      = 5.0,
    C0_mean       = 2600.0,
    C0_unit_sd    = 0.10,    # fraction
    bam_bias      = 1.02,    # BAM slightly biased high (realistic)
    seed          = 42,
)
    rng      = Random.MersenneTwister(seed)
    site_ids = ["SITE_" * lpad(s, 2, '0') for s in 1:n_sites]
    unit_ids = ["UNIT_" * lpad(u, 3, '0') for u in 1:n_units]
    phases   = ["C$p" for p in 1:n_phases]

    # Assign units to sites (round-robin)
    unit_site = [site_ids[mod1(u, n_sites)] for u in 1:n_units]

    # Per-unit C0
    C0_units = C0_mean .* (1.0 .+ C0_unit_sd .* randn(rng, n_units))

    rows = []

    for (iu, uid) in enumerate(unit_ids)
        sid       = unit_site[iu]
        C0_u      = C0_units[iu]
        # Stagger start date per unit
        start_day = round(Int, 5 * randn(rng))
        n_actual  = n_days + round(Int, 10 * randn(rng))
        n_actual  = max(30, n_actual)

        for d in 1:n_actual
            t_abs   = d + start_day   # day index (arbitrary origin)
            age     = Float64(d - 1)  # deploy_age in days
            phase   = phases[mod1(div(d - 1, div(n_actual, n_phases)) + 1, n_phases)]

            # Environmental drivers
            rh      = clamp(0.55 + 0.20 * sin(2π * t_abs / 365 + 1.0)
                           + 0.08 * randn(rng), 0.15, 0.95)
            temp    = 20.0 + 8.0 * sin(2π * t_abs / 365) + 2.0 * randn(rng)
            wind    = clamp(2.5 - 1.5 * log(-log(rand(rng))), 0.0, 20.0)
            drhdt   = (d > 1) ? (rh - clamp(0.55 + 0.20 * sin(2π * (t_abs-1) / 365 + 1.0), 0.15, 0.95)) : 0.0

            # Physics signal (power-law f(RH) placeholder — S_phys replaces this)
            rh_dry  = 0.20
            aw      = clamp(rh, 0.0, 0.985)
            aw_dry  = clamp(rh_dry, 0.0, 0.985)
            gf_rh   = (1.0 + kappa_true * aw / (1.0 - aw))^(1.0/3.0)
            gf_dry  = (1.0 + kappa_true * aw_dry / (1.0 - aw_dry))^(1.0/3.0)
            f_rh    = (gf_rh / gf_dry)^1.25

            # Injected effects
            flow_fac = 1.0 + a_flow_true * (wind > w_thresh ? 1.0 : 0.0)
            hyst_fac = 1.0 - 0.04 * tanh(drhdt / 0.05)
            noise    = 1.0 + 0.05 * randn(rng)

            ch1 = C0_u * f_rh * flow_fac * hyst_fac * noise

            # BAM reference: first unit at each site has collocated BAM
            bam_pm_val = if uid == first(filter(u -> unit_site[findfirst(==(u), unit_ids)] == sid, unit_ids))
                # BAM converts scattering to PM2.5 with a bias
                (C0_u * 0.015 * f_rh * bam_bias) * (1.0 + 0.02 * randn(rng))
            else
                missing
            end

            # Use a fixed epoch (2024-01-01) as the synthetic time origin
            ts = DateTime(2024, 1, 1) + Day(t_abs - 1)

            push!(rows, (
                unit_id    = uid,
                site_id    = sid,
                timestamp  = ts,
                phase      = phase,
                ch1        = ch1,
                deploy_age = age,
                bam_pm     = bam_pm_val,
                rh         = rh,          # FRACTION, not %
                temp       = temp,
                wind       = wind,
                drhdt      = drhdt,
            ))
        end
    end

    df = DataFrame(rows)
    validate_panel(df)
    return df
end


# ---------------------------------------------------------------------------
# §0-D  Quick sanity print
# ---------------------------------------------------------------------------

function describe_panel(df::DataFrame)
    @printf "Panel: %d rows  %d units  %d sites  %d phases\n" nrow(df) length(unique(df.unit_id)) length(unique(df.site_id)) length(unique(df.phase))
    @printf "  timestamp range: %s → %s\n" string(minimum(df.timestamp)) string(maximum(df.timestamp))
    @printf "  RH range: [%.3f, %.3f]  (fraction; if >1 → bug)\n" minimum(df.rh) maximum(df.rh)
    @printf "  BAM present: %d rows (%.1f%%)\n" count(!ismissing, df.bam_pm) (100*count(!ismissing, df.bam_pm)/nrow(df))
    @printf "  CH1 range: [%.1f, %.1f]\n" minimum(df.ch1) maximum(df.ch1)
end
