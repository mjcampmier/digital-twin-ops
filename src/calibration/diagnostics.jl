# =============================================================================
#  diagnostics.jl — §6: Post-fit residual diagnostics
#
#  Four diagnostic tools applied to residuals r_t = y_t − ŷ_t after fit:
#
#    1. Allan deviation  σ_A(τ) — reveals dominant noise structure vs time scale
#       τ.  White noise: σ_A ~ τ^{-½}.  Drift: σ_A ~ τ^{+½}.  If σ_A is flat,
#       there is a flicker (1/f) noise component — usually an instrument defect.
#
#    2. Power spectral density on residuals (periodogram via FFT).
#       Unstructured residuals look like a flat spectrum.  Peaks at diurnal or
#       annual frequency indicate missing environmental covariate structure.
#
#    3. frac_above(r, σ_obs, threshold=2.0)  →  fraction with |r| > thresh·σ.
#       Should be ≈ 4.6% for Gaussian.  If >> 5%, the model is under-dispersed
#       or the noise distribution is heavy-tailed.
#
#    4. Per-site RH coverage flag.  If the RH range at a site is < Δrh_min
#       (default 0.30 fraction), κ_eff has insufficient leverage and the fitted
#       value is unreliable for that site.
#
#  Requires: fit.jl (UnitData, solve_unit), parameters.jl
# =============================================================================

using Statistics, Printf


# ---------------------------------------------------------------------------
# §6-A  Allan deviation
# ---------------------------------------------------------------------------

"""
    allan_deviation(r, t_days; max_lags) → (τ_days, σ_A)

Allan deviation of a residual time series `r` sampled at times `t_days`.
Assumes approximately uniform spacing; uses the median spacing as Δt.

τ_days : lag values [days] (powers of 2 × Δt)
σ_A    : Allan deviation at each lag
"""
function allan_deviation(r::AbstractVector, t_days::AbstractVector;
                          max_lags::Int = 8)
    Δt = median(diff(t_days))
    n  = length(r)
    τs = Float64[]
    σs = Float64[]

    for k in 0:max_lags
        stride = 2^k
        stride >= n ÷ 2 && break
        # Allan variance at lag stride: mean((r[i+stride]-r[i])^2 / 2)
        sq = 0.0; cnt = 0
        for i in 1:(n - stride)
            sq  += (r[i + stride] - r[i])^2
            cnt += 1
        end
        push!(τs, stride * Δt)
        push!(σs, sqrt(sq / cnt / 2))
    end
    return τs, σs
end


"""
    print_allan(τs, σs)

Pretty-print Allan deviation table to stdout.
"""
function print_allan(τs::AbstractVector, σs::AbstractVector)
    println("Allan deviation:")
    @printf "  %-12s  %-12s\n" "τ [days]" "σ_A [counts]"
    println("  " * "-"^26)
    for (τ, σ) in zip(τs, σs)
        @printf "  %-12.3f  %-12.2f\n" τ σ
    end
end


# ---------------------------------------------------------------------------
# §6-B  Power spectral density (periodogram)
# ---------------------------------------------------------------------------

"""
    residual_psd(r, t_days) → (freqs, power)

One-sided Lomb-Scargle-style periodogram (DFT magnitude) using a Hanning window.
Uses a direct O(N²) DFT — no FFTW dependency; fast enough for N ≤ 500.
freqs  : [cycles/day]
power  : spectral power [counts² / (cycles/day)]

Note: if residuals are long (N > 1000), consider adding FFTW.jl and replacing
the inner loop with `fft(rw)` for O(N log N) performance.
"""
function residual_psd(r::AbstractVector, t_days::AbstractVector)
    n   = length(r)
    Δt  = median(diff(t_days))
    win = 0.5 .* (1.0 .- cos.(2π .* (0:n-1) ./ (n - 1)))   # Hanning window
    rw  = ComplexF64.((r .- mean(r)) .* win)
    n2  = div(n, 2) + 1
    freqs = collect(0:n2-1) ./ (n * Δt)
    pw    = Vector{Float64}(undef, n2)

    # Direct DFT
    for k in 0:n2-1
        phase = ComplexF64(0, -2π * k / n)
        s     = sum(rw[j+1] * exp(phase * j) for j in 0:n-1)
        pw[k+1] = abs2(s) * Δt / sum(abs2, win)
    end
    pw[2:end-1] .*= 2.0   # one-sided doubling
    return freqs, pw
end


"""
    print_psd_peaks(freqs, power; n_peaks, diurnal_atol)

Report the top-n spectral peaks and flag diurnal (1 cyc/day) content.
"""
function print_psd_peaks(freqs::AbstractVector, power::AbstractVector;
                          n_peaks::Int = 5, diurnal_atol::Float64 = 0.1)
    idx_sorted = sortperm(power, rev = true)
    println("PSD top peaks:")
    for i in 1:min(n_peaks, length(freqs))
        f   = freqs[idx_sorted[i]]
        pw  = power[idx_sorted[i]]
        tag = abs(f - 1.0) < diurnal_atol ? " ← diurnal" : ""
        @printf "  f=%.3f cyc/day  power=%.2e%s\n" f pw tag
    end
end


# ---------------------------------------------------------------------------
# §6-C  frac_above
# ---------------------------------------------------------------------------

"""
    frac_above(r, σ_obs; threshold=2.0) → Float64

Fraction of residuals with absolute value > threshold × σ_obs.
Gaussian expectation: 4.55% at threshold=2.  >> 5% → model under-dispersed.
"""
function frac_above(r::AbstractVector, σ_obs::Real; threshold::Float64 = 2.0)
    return mean(abs.(r) .> threshold * σ_obs)
end


# ---------------------------------------------------------------------------
# §6-D  Per-site RH coverage flag
# ---------------------------------------------------------------------------

"""
    rh_coverage_flag(df; rh_min_range=0.30) → Dict{String, Bool}

Returns a Dict mapping site_id → true (adequate coverage) / false (too narrow).
Sites with RH range < rh_min_range have insufficient leverage for κ_eff fitting.
"""
function rh_coverage_flag(df; rh_min_range::Float64 = 0.30)
    result = Dict{String, Bool}()
    for gdf in groupby(df, :site_id)
        sid   = gdf.site_id[1]
        rh_range = maximum(gdf.rh) - minimum(gdf.rh)
        result[sid] = rh_range >= rh_min_range
    end
    return result
end


# ---------------------------------------------------------------------------
# §6-E  Full diagnostics report for all units
# ---------------------------------------------------------------------------

"""
    run_diagnostics(fitted_params, unit_data, sphys_table, design, panel_df;
                    verbose)

Run all four diagnostics for every unit and print a summary report.
Returns a named tuple with structured results for programmatic inspection.
"""
function run_diagnostics(fitted_params::CalibParams,
                          unit_data::Vector{UnitData},
                          sphys_table::SPHYSTable,
                          design::PMS5003Design,
                          panel_df;
                          verbose::Bool = true)
    unit_results = []

    for (iu, ud) in enumerate(unit_data)
        rhs_p  = rhs_params_for_unit(fitted_params, iu, ud.site_idx)
        σ_obs  = exp(fitted_params.units[iu].log_σ)
        ŷ      = solve_unit(ud, rhs_p, sphys_table, design)
        r      = ud.ch1 .- ŷ

        τs, σA = allan_deviation(r, ud.t_days)
        fa     = frac_above(r, σ_obs)
        nll_u  = sum(0.5 .* (r ./ σ_obs).^2) + length(r) * log(σ_obs)

        if verbose
            println("\n── Unit $(ud.unit_id)  (site=$(ud.site_id)) ──")
            @printf "  σ_obs=%.1f  frac_above_2σ=%.1f%%  NLL=%.2e\n" σ_obs (100*fa) nll_u
            print_allan(τs, σA)
            freqs, pw = residual_psd(r, ud.t_days)
            print_psd_peaks(freqs, pw)
        end

        push!(unit_results, (
            unit_id   = ud.unit_id,
            residuals = r,
            ŷ         = ŷ,
            σ_obs     = σ_obs,
            frac_above= fa,
            allan_τ   = τs,
            allan_σA  = σA,
            nll       = nll_u,
        ))
    end

    rh_flags = rh_coverage_flag(panel_df)
    if verbose
        println("\n── Per-site RH coverage ──")
        for (sid, ok) in sort(collect(rh_flags))
            @printf "  %-12s  %s\n" sid ok ? "OK (≥0.30)" : "WARN: narrow RH range"
        end
    end

    return (units = unit_results, rh_flags = rh_flags)
end
