# =============================================================================
#  PMS5003 digital twin as a state-space Universal Differential Equation
#  UniversalDiffEq.jl  (Jack-H-Buckner/UniversalDiffEq.jl)
#
#  known physics (deterministic) + neural residual (aging / composition /
#  RH hysteresis / flow), fit to a CH1 time series with covariates.
#
#  Everything mechanistic here was validated numerically in physics_check.py
#  (full Mie via miepython + kappa-Koehler + PSE, lambda = 657 nm) and reduced
#  to cheap, autodiff-friendly closed forms so the "known physics" term is
#  a couple of lines inside the ODE RHS rather than a Mie integral solved
#  thousands of times under ForwardDiff.
#
#  ---------------------------------------------------------------------------
#  API notes (checked against the package source, not memory):
#   * With covariates, the RHS is called OUT-OF-PLACE as  dudt(u, x, p, t)
#     returning a vector  (check_arguments_X: nargs==5 branch). The concept
#     note's `dudt(u,p,t,X)` is wrong on ordering -- the covariate vector `x`
#     is argument #2, between the state and the parameters. (In-place
#     `dudt!(du,u,x,p,t)` also works; out-of-place matches the LotkaVolterra
#     idiom and composes cleanly with the NN.)
#   * `x` holds the covariates at time t, linearly interpolated, in the COLUMN
#     ORDER of the wide X DataFrame minus the time column (process_data).
#     Here that is:  x = [RH, T, wind, t_deploy, dRHdt].
#   * get_right_hand_side(model) returns (u,x,t) -> du when covariates are
#     present -- the hook we use below to read out the learned residual.
#
#  Install (Julia >= 1.9):
#     import Pkg
#     Pkg.add(["UniversalDiffEq","DataFrames","Lux","ComponentArrays",
#              "Random","Statistics","Plots"])
# =============================================================================

using UniversalDiffEq, DataFrames, Lux, ComponentArrays, Random, Statistics, Printf
# using Plots   # uncomment for the diagnostic figures at the bottom

Random.seed!(20260707)

# -----------------------------------------------------------------------------
# 1.  KNOWN PHYSICS  (validated constants from physics_check.py)
# -----------------------------------------------------------------------------
const RH_DRY  = 0.20        # reference "dry" RH the calibration is anchored at
const KAPPA   = 0.20        # kappa-Koehler hygroscopicity (ambient mixed aerosol)
const P_SCAT  = 1.2466      # scattering-vs-growth exponent   (fit R^2 = 0.996)
const CAL0015 = 0.015       # 2022 paper: b_sp1 [Mm^-1] = 0.015 * CH1  (RH<40%)

"kappa-Koehler diameter growth factor GF = Dp_wet/Dp_dry (Kelvin neglected)."
@inline function growth_factor(rh)
    aw = clamp(rh, 0.0, 0.985)                 # water activity ~ RH; clip blowup
    return (1.0 + KAPPA * aw / (1.0 - aw))^(1.0/3.0)
end
const GF_DRY = growth_factor(RH_DRY)

"""
Humidification factor f(RH) = b_obs(RH)/b_obs(dry), i.e. the RH scaling of the
sensor-observed scattering. Surrogate for the full Mie+kappa-Koehler curve:
    f(RH) = ( GF(RH) / GF(dry) )^P_SCAT
Reproduces the first-principles curve to R^2 = 0.996 (physics_check.py).
"""
@inline f_RH(rh) = (growth_factor(rh) / GF_DRY)^P_SCAT

"Instantaneous physics-predicted CH1 given RH and a dry-state calibration C0."
@inline CH1_phys(rh, C0) = C0 * f_RH(rh)

# NB: the OPC undersizing (PSE, Ouimette 2024) enters the ABSOLUTE dry level
# (observed/true scattering ~0.50 with natural-log PSE) and is therefore folded
# into the single per-sensor calibration constant C0 -- it does not change the
# *shape* f(RH). If you later feed a measured size distribution as a covariate,
# promote the PSE-weighted Mie integral back into CH1_phys and let C0 float.

# -----------------------------------------------------------------------------
# 2.  NEURAL RESIDUAL  (the actual unknowns)
# -----------------------------------------------------------------------------
# fixed input-normalization scales (keep NN inputs O(1); independent of fitted C0)
const CH1_SCALE  = 2600.0
const T_SCALE    = 30.0
const WIND_SCALE = 5.0
const DRH_SCALE  = 0.10

# 6 inputs -> 1 output:  [CH1_norm, RH, T_norm, wind_norm, t_deploy_yr, dRHdt_norm]
NN, NN_p0 = SimpleNeuralNetwork(6, 1; hidden = 12)

@inline function residual(u, x, p)
    inp = [ u[1] / CH1_SCALE,      # current reading (normalized)
            x[1],                  # RH               (fraction, already ~O(1))
            x[2] / T_SCALE,        # air temperature
            x[3] / WIND_SCALE,     # wind speed  -> inlet flow loading
            x[4],                  # time since deployment [yr] -> aging proxy
            x[5] / DRH_SCALE ]     # dRH/dt -> gives the NN the info to represent
                                   #          deliquescence/efflorescence hysteresis
                                   #          WITHOUT needing a hidden memory state
    return NN(inp, p.NN)[1] * CH1_SCALE      # residual tendency [counts / day]
end

# -----------------------------------------------------------------------------
# 3.  THE UDE RHS   dudt(u, x, p, t)   (state u = [CH1], fully observed)
# -----------------------------------------------------------------------------
# Relaxation form, matching the concept note's  `known = mie(...) - u[1]`:
# CH1 relaxes toward the humidity-adjusted physics target on timescale 1/k, and
# the NN supplies the slow multiplicative drift (aging), path-dependence
# (hysteresis via dRHdt) and flow bias (wind). At steady state
#     u* = CH1_phys(RH) + residual/k ,
# so the residual encodes the fractional departure from clean physics.
#
# (Alternative tendency form, no restoring constant:
#     du = C0 * f'(RH) * dRHdt  +  residual
#  -- purer, but relies on the state-space filter to re-anchor the level. Kept
#  the relaxation form for robustness during the inner ODE solves.)
function dudt(u, x, p, t)
    k  = abs(p.log_k) + 1e-3            # positive relaxation rate [1/day]
    C0 = abs(p.C0)                      # per-sensor dry calibration [counts]
    known = k * (CH1_phys(x[1], C0) - u[1])
    resid = residual(u, x, p)
    return [known + resid]
end

# initial parameters: NN weights + the two physical scalars
init_parameters = (NN = NN_p0, log_k = 1.0, C0 = 2600.0)

# -----------------------------------------------------------------------------
# 4.  SELF-CONTAINED SYNTHETIC GROUND TRUTH
#     (so the script runs with no external data; the injected effects are what
#      the validation in §6 tries to recover from the fitted NN)
# -----------------------------------------------------------------------------
function synthetic_deployment(; ndays = 400, C0_true = 2600.0, seed = 1)
    rng   = Random.MersenneTwister(seed)
    td    = collect(0.0:1.0:(ndays-1))          # time [days]
    tyr   = td ./ 365.0
    # covariates
    RH    = clamp.(0.55 .+ 0.20 .* sin.(2π .* td ./ 365 .+ 1.0) .+ 0.08 .* randn(rng, ndays), 0.15, 0.95)
    Tair  = 20.0 .+ 8.0 .* sin.(2π .* td ./ 365) .+ 2.0 .* randn(rng, ndays)     # degC
    wind  = clamp.(2.0 .+ 1.0 .* randn(rng, ndays), 0.0, 8.0)                    # m/s
    dRHdt = vcat(0.0, diff(RH))                                                  # /day
    # ground-truth UNMODELED multiplicative effects (what the NN must learn):
    aging = exp.(-0.105 .* tyr)                    # ~10%/yr sensitivity loss (Fig S15, 2022)
    hyst  = 1.0 .- 0.04 .* tanh.(dRHdt ./ 0.05)    # rising RH slightly suppresses (path dep.)
    flow  = 1.0 .- 0.03 .* (wind ./ WIND_SCALE)    # inlet wind loading
    CH1_true = C0_true .* f_RH.(RH) .* aging .* hyst .* flow
    CH1_obs  = CH1_true .* (1.0 .+ 0.05 .* randn(rng, ndays))   # 5% observation error
    data = DataFrame(time = td, CH1 = CH1_obs)
    # covariate DataFrame -- COLUMN ORDER defines x = [RH,T,wind,t_deploy,dRHdt]
    X = DataFrame(time = td, RH = RH, T = Tair, wind = wind, t_deploy = tyr, dRHdt = dRHdt)
    truth = (; td, tyr, RH, wind, dRHdt, aging, hyst, flow, CH1_true, C0_true)
    return data, X, truth
end

data, X, truth = synthetic_deployment()

# -----------------------------------------------------------------------------
# 5.  BUILD + TRAIN
# -----------------------------------------------------------------------------
model = CustomDerivatives(data, X, dudt, init_parameters;
                          time_column_name = "time",
                          proc_weight = 2.0,     # trust the process model a bit more
                          obs_weight  = 1.0,
                          reg_weight  = 1e-4,    # L2 on the NN keeps the residual small
                          reg_type    = "L2")

# Fast first fit: "derivative matching" (spline the data, match du/dt).
# For the proper noisy state-space treatment switch to the UKF losses:
#   loss_function = "conditional likelihood"   (or "marginal likelihood"),
#   loss_options  = (observation_error = 0.05, process_error = 0.02)
train!(model;
       loss_function = "derivative matching",
       optimizer     = "ADAM",
       verbose       = true,
       optim_options = (maxiter = 2500,))

# refine with BFGS (optional but usually worth it)
train!(model;
       loss_function = "derivative matching",
       optimizer     = "BFGS",
       verbose       = true,
       optim_options = (maxiter = 400,))

# -----------------------------------------------------------------------------
# 6.  VALIDATION  --  does the fitted residual reproduce the injected effects
#     WITHOUT having been told about them?  (the real test of the decomposition)
# -----------------------------------------------------------------------------
RHS   = get_right_hand_side(model)          # (u, x, t) -> du   (covariate model)
pars  = get_parameters(model)
k_hat = abs(pars.log_k) + 1e-3
C0hat = abs(pars.C0)
println("\nfitted:  k = $(round(k_hat,digits=3)) /day   C0 = $(round(C0hat,digits=1)) counts   (true C0 = $(truth.C0_true))")

# tiny bisection for the RHS fixed point in u at a held covariate vector
function fixed_point_u(x; lo = 0.3*C0hat, hi = 2.0*C0hat)
    g(u1) = RHS([u1], x, 0.0)[1]
    glo, ghi = g(lo), g(hi)
    for _ in 1:60
        mid = 0.5*(lo+hi); gm = g(mid)
        (sign(gm) == sign(glo)) ? (lo, glo = mid, gm) : (hi = mid)
    end
    return 0.5*(lo+hi)
end

# (a) recovered AGING curve: hold RH=dry (f_RH=1), wind=0, dRHdt=0, sweep t_deploy
println("\n-- recovered aging vs injected exp(-0.105*t_yr) --")
println("  t[yr]   recovered   injected")
for tyr in (0.0, 0.25, 0.5, 0.75, 1.0)
    xa = [RH_DRY, 20.0, 0.0, tyr, 0.0]          # [RH,T,wind,t_deploy,dRHdt]
    corr = fixed_point_u(xa) / C0hat            # u*/C0 = f_RH(dry)*aging*... = aging here
    @printf("  %4.2f     %6.3f      %6.3f\n", tyr, corr, exp(-0.105*tyr))
end

# (b) recovered HUMIDIFICATION vs the known physics f(RH) (should track closely,
#     since f(RH) is in the deterministic term; deviation = what NN adds on RH)
println("\n-- recovered CH1(RH)/CH1(dry) at deployment vs physics f(RH) --")
println("  RH     recovered   f_RH(physics)")
base = fixed_point_u([RH_DRY, 20.0, 0.0, 0.0, 0.0])
for rh in (0.40, 0.60, 0.80, 0.90)
    xr = [rh, 20.0, 0.0, 0.0, 0.0]
    @printf("  %4.2f    %6.3f      %6.3f\n", rh, fixed_point_u(xr)/base, f_RH(rh))
end

# (c) outside sanity check: 2022 empirical b_sp1 = 0.015*CH1 at low RH
b_sp1_pred = CAL0015 * C0hat
@printf("\n-- 2022 baseline: b_sp1 = 0.015*CH1 => %.2f Mm^-1 at dry-state C0 --\n", b_sp1_pred)

# ---- optional figures ----
# using Plots
# plt = plot_predictions(model);          savefig(plt, "fit.png")
# plt2 = plot_state_estimates(model);     savefig(plt2, "state.png")

# =============================================================================
# 7.  MULTI-UNIT EXTENSION  (collocated PurpleAir units, shared nonlinearity +
#     per-sensor offset/scale) -- Table S1 (2022) shows huge unit-to-unit spread,
#     so per-sensor scale/offset alongside the shared correction is essential.
#
#     Multi + covariates RHS signature (check_arguments_multi_X, nargs==6 branch):
#         dudt_multi(u, i, x, p, t)      # i = series index (Int-valued)
#     data / X need a `series` column; parameters may carry per-series vectors.
# =============================================================================
#
# NSERIES = 4
# NN_m, NNm_p0 = SimpleNeuralNetwork(6, 1; hidden = 12)   # shared residual
#
# function dudt_multi(u, i, x, p, t)
#     s   = round(Int, i)
#     k   = abs(p.log_k) + 1e-3
#     C0  = abs(p.C0[s])                    # per-sensor scale
#     off = p.offset[s]                     # per-sensor additive offset (counts)
#     inp = [u[1]/CH1_SCALE, x[1], x[2]/T_SCALE, x[3]/WIND_SCALE, x[4], x[5]/DRH_SCALE]
#     known = k * (CH1_phys(x[1], C0) + off - u[1])
#     resid = NN_m(inp, p.NN)[1] * CH1_SCALE
#     return [known + resid]
# end
#
# init_parameters_multi = (NN = NNm_p0,
#                          log_k  = 1.0,
#                          C0     = fill(2600.0, NSERIES),
#                          offset = zeros(NSERIES))
#
# # data_multi, X_multi must include a `series` column (1..NSERIES)
# model_multi = MultiCustomDerivatives(data_multi, X_multi, dudt_multi,
#                                      init_parameters_multi;
#                                      time_column_name   = "time",
#                                      series_column_name = "series")
# train!(model_multi; loss_function = "derivative matching",
#        optimizer = "ADAM", optim_options = (maxiter = 3000,))
