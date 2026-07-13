# =============================================================================
#  synthetic.jl — self-contained synthetic ground truth for PMS5003 UDE
#
#  Requires physics.jl to be included first (uses f_RH and PhysicsParams).
#  The injected effects (aging, hysteresis, flow) are the unmodeled signals
#  the NN must recover; their amplitudes are defended below.
# =============================================================================
using DataFrames, Random, Statistics

"""
Generate a synthetic 400-day PMS5003 deployment.

Defended assumptions in the injected effects:
  aging  exp(−0.105·t_yr)  ≈ 10%/yr sensitivity loss.  Source: Fig S15 (2022 paper).
  hyst   1 − 0.04·tanh(dRHdt/0.05)  rising-RH suppression.  Amplitude 4% and
         timescale 0.05 RH/day are deliberately mild so the NN can recover them
         without being told the functional form.
  flow   binary gust gate: g_flow_true = a_flow_true · (wind > w_thresh).
         Aspiration loss above threshold lowers signal; a_flow_true ∈ [−0.10,−0.15].
         Binary matches the real covariate's information content (ERA5 wind at
         ~25 km grid has no sub-hourly turbulence skill; enclosure distortion
         further degrades continuous shape).  Zero below threshold by construction.
  noise  5% multiplicative log-normal.  Order-of-magnitude estimate; real sensor
         noise is partly systematic (quantisation in 0.1 µg/m³ bins).

kappa_true is the injected lumped RH-response parameter.  It is distinct from
the UDE init value (≈0.2) so that kappa_eff recovery is a real identifiability
test, not a trivial zero-distance check.  p_scat is taken from pp (fixed 1.25).
"""
function synthetic_deployment(pp::PhysicsParams;
                               ndays        = 400,
                               C0_true      = 2600.0,
                               kappa_true   = 0.25,
                               a_flow_true  = -0.12,
                               w_thresh     = 5.0,
                               seed         = 1,
                               f_rh_fn      = nothing)
    rng  = Random.MersenneTwister(seed)
    td   = collect(0.0:1.0:(ndays - 1))
    tyr  = td ./ 365.0

    RH    = clamp.(0.55 .+ 0.20 .* sin.(2π .* td ./ 365 .+ 1.0)
                       .+ 0.08 .* randn(rng, ndays), 0.15, 0.95)
    Tair  = 20.0 .+ 8.0 .* sin.(2π .* td ./ 365) .+ 2.0 .* randn(rng, ndays)
    # Gumbel(μ=2.5, β=1.5): mean≈3.4 m/s, P(wind>5)≈17% — realistic ERA5 daily wind
    wind  = clamp.(2.5 .- 1.5 .* log.(-log.(rand(rng, ndays))), 0.0, 20.0)
    dRHdt = vcat(0.0, diff(RH))

    aging = ones(ndays)
    hyst  = 1.0 .- 0.04 .* tanh.(dRHdt ./ 0.05)
    flow  = 1.0 .+ a_flow_true .* (wind .> w_thresh)

    # f_rh_fn = nothing → power-law (default); otherwise e.g. (rh,k)->interp_frh(tbl,rh,k)
    f_rh_vals = isnothing(f_rh_fn) ?
        f_RH.(RH, kappa_true, Ref(pp)) :
        [f_rh_fn(rh, kappa_true) for rh in RH]

    CH1_true = C0_true .* f_rh_vals .* aging .* hyst .* flow
    CH1_obs  = CH1_true .* (1.0 .+ 0.05 .* randn(rng, ndays))

    data  = DataFrame(time = td, CH1 = CH1_obs)
    X     = DataFrame(time = td, RH = RH, T = Tair,
                      wind = wind, t_deploy = tyr, dRHdt = dRHdt)
    frac_above = mean(wind .> w_thresh)
    truth = (; td, tyr, RH, wind, dRHdt, aging, hyst, flow, CH1_true,
               C0_true, kappa_true, a_flow_true, w_thresh, frac_above,
               physics = isnothing(f_rh_fn) ? :power_law : :mie)
    return data, X, truth
end
