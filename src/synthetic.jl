# =============================================================================
#  synthetic.jl — self-contained synthetic ground truth for PMS5003 UDE
#
#  Requires physics.jl to be included first (uses f_RH and PhysicsParams).
#  The injected effects (aging, hysteresis, flow) are the unmodeled signals
#  the NN must recover; their amplitudes are defended below.
# =============================================================================
using DataFrames, Random

"""
Generate a synthetic 400-day PMS5003 deployment.

Defended assumptions in the injected effects:
  aging  exp(−0.105·t_yr)  ≈ 10%/yr sensitivity loss.  Source: Fig S15 (2022 paper).
  hyst   1 − 0.04·tanh(dRHdt/0.05)  rising-RH suppression.  Amplitude 4% and
         timescale 0.05 RH/day are deliberately mild so the NN can recover them
         without being told the functional form.
  flow   1 − 0.03·(wind/5)  inlet wind loading.  Source: Ouimette 2024 ~30% CH1
         reduction at 1 Pa impedance; 3% linear at reference wind is conservative.
  noise  5% multiplicative log-normal.  Order-of-magnitude estimate; real sensor
         noise is partly systematic (quantisation in 0.1 µg/m³ bins).

The physics argument (f_RH) is parameterised by `pp` so kappa/p_scat sensitivity
analyses can regenerate synthetic data under a different assumed physics.
"""
function synthetic_deployment(pp::PhysicsParams;
                               ndays   = 400,
                               C0_true = 2600.0,
                               seed    = 1)
    rng  = Random.MersenneTwister(seed)
    td   = collect(0.0:1.0:(ndays - 1))
    tyr  = td ./ 365.0

    RH    = clamp.(0.55 .+ 0.20 .* sin.(2π .* td ./ 365 .+ 1.0)
                       .+ 0.08 .* randn(rng, ndays), 0.15, 0.95)
    Tair  = 20.0 .+ 8.0 .* sin.(2π .* td ./ 365) .+ 2.0 .* randn(rng, ndays)
    wind  = clamp.(2.0 .+ 1.0 .* randn(rng, ndays), 0.0, 8.0)
    dRHdt = vcat(0.0, diff(RH))

    aging = exp.(-0.105 .* tyr)
    hyst  = 1.0 .- 0.04 .* tanh.(dRHdt ./ 0.05)
    flow  = 1.0 .- 0.03 .* (wind ./ 5.0)

    CH1_true = C0_true .* f_RH.(RH, Ref(pp)) .* aging .* hyst .* flow
    CH1_obs  = CH1_true .* (1.0 .+ 0.05 .* randn(rng, ndays))

    data  = DataFrame(time = td, CH1 = CH1_obs)
    X     = DataFrame(time = td, RH = RH, T = Tair,
                      wind = wind, t_deploy = tyr, dRHdt = dRHdt)
    truth = (; td, tyr, RH, wind, dRHdt, aging, hyst, flow, CH1_true, C0_true)
    return data, X, truth
end
