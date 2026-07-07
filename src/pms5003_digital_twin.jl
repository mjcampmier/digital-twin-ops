# =============================================================================
#  pms5003_digital_twin.jl — PMS5003 UDE digital twin (entry point)
#
#  Known physics (deterministic) + neural residual (aging / composition /
#  RH hysteresis / flow), fit to a CH1 time series with covariates.
#  Built on UniversalDiffEq.jl (Jack-H-Buckner/UniversalDiffEq.jl).
#
#  Run:   julia --project . src/pms5003_digital_twin.jl
#
#  ---------------------------------------------------------------------------
#  API notes (checked against the package source):
#   * With covariates, RHS is called OUT-OF-PLACE as  dudt(u, x, p, t).
#     x holds the covariates at time t, linearly interpolated, in column order
#     of the X DataFrame minus the time column:  x = [RH, T, wind, t_deploy, dRHdt].
#   * get_right_hand_side(model) returns (u, x, t) -> du when covariates present.
#
#  All scalar physics assumptions live in PhysicsParams (src/physics.jl).
#  Sensitivity analyses are in src/sensitivity/*.jl.
# =============================================================================

include(joinpath(@__DIR__, "physics.jl"))
include(joinpath(@__DIR__, "synthetic.jl"))

using UniversalDiffEq, DataFrames, Lux, ComponentArrays, Random, Statistics, Printf
# using Plots   # uncomment for diagnostic figures at the bottom

Random.seed!(20260707)

# Baseline physics — all assumptions at their defended default values.
const PP = PhysicsParams()

# -----------------------------------------------------------------------------
# 1.  NEURAL RESIDUAL
# -----------------------------------------------------------------------------
# Fixed input-normalization scales keep NN inputs O(1) and independent of C0.
const CH1_SCALE  = 2600.0
const T_SCALE    = 30.0
const WIND_SCALE = 5.0
const DRH_SCALE  = 0.10

# 6 inputs -> 1 output:  [CH1_norm, RH, T_norm, wind_norm, t_deploy_yr, dRHdt_norm]
NN, NN_p0 = SimpleNeuralNetwork(6, 1; hidden = 12)

@inline function residual(u, x, p)
    inp = [u[1] / CH1_SCALE,
           x[1],
           x[2] / T_SCALE,
           x[3] / WIND_SCALE,
           x[4],
           x[5] / DRH_SCALE]
    return NN(inp, p.NN)[1] * CH1_SCALE
end

# -----------------------------------------------------------------------------
# 2.  UDE RHS  — relaxation toward physics target + NN residual
# -----------------------------------------------------------------------------
# GF_dry is pre-computed once inside make_dudt; not re-evaluated under ForwardDiff.
# At steady state:  u* = CH1_phys(RH) + residual/k
# Relaxation form is more robust during inner ODE solves than the pure tendency form.
const dudt = make_dudt(PP, residual)

init_parameters = (NN = NN_p0, log_k = 1.0, C0 = 2600.0)

# -----------------------------------------------------------------------------
# 3.  SYNTHETIC GROUND TRUTH
# -----------------------------------------------------------------------------
data, X, truth = synthetic_deployment(PP)

# -----------------------------------------------------------------------------
# 4.  BUILD + TRAIN
# -----------------------------------------------------------------------------
model = CustomDerivatives(data, X, dudt, init_parameters;
                          time_column_name = "time",
                          proc_weight      = 2.0,
                          obs_weight       = 1.0,
                          reg_weight       = 1e-4,
                          reg_type         = "L2")

# Stage 1: derivative matching (fast; splines the data and matches du/dt).
# For proper UKF state-space treatment use:
#   loss_function = "conditional likelihood"
#   loss_options  = (observation_error = 0.05, process_error = 0.02)
train!(model;
       loss_function = "derivative matching",
       optimizer     = "ADAM",
       verbose       = true,
       optim_options = (maxiter = 2500,))

# Stage 2: BFGS refinement (optional but consistently improves final loss).
train!(model;
       loss_function = "derivative matching",
       optimizer     = "BFGS",
       verbose       = true,
       optim_options = (maxiter = 400,))

# -----------------------------------------------------------------------------
# 5.  VALIDATION — does the NN recover the injected effects without being told?
# -----------------------------------------------------------------------------
RHS  = get_right_hand_side(model)
pars = get_parameters(model)
k_hat  = abs(pars.log_k) + 1e-3
C0_hat = abs(pars.C0)
println("\nfitted:  k = $(round(k_hat, digits=3)) /day   C0 = $(round(C0_hat, digits=1)) counts   (true C0 = $(truth.C0_true))")

function fixed_point_u(x; lo = 0.3 * C0_hat, hi = 2.0 * C0_hat)
    g(u1) = RHS([u1], x, 0.0)[1]
    glo = g(lo)
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        gm  = g(mid)
        (sign(gm) == sign(glo)) ? (lo = mid; glo = gm) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

# (a) Aging recovery: hold RH=dry, wind=0, dRHdt=0, sweep deployment time.
println("\n-- recovered aging vs injected exp(−0.105·t_yr) --")
println("  t[yr]   recovered   injected")
for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    xa   = [PP.rh_dry, 20.0, 0.0, t, 0.0]
    corr = fixed_point_u(xa) / C0_hat
    @printf "  %4.2f     %6.3f      %6.3f\n" t corr exp(-0.105 * t)
end

# (b) Humidification recovery: compare fitted u*(RH)/u*(dry) against physics f(RH).
println("\n-- recovered CH1(RH)/CH1(dry) vs physics f(RH) --")
println("  RH     recovered   f_RH(physics)")
base = fixed_point_u([PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
for rh in (0.40, 0.60, 0.80, 0.90)
    xr = [rh, 20.0, 0.0, 0.0, 0.0]
    @printf "  %4.2f    %6.3f      %6.3f\n" rh fixed_point_u(xr) / base f_RH(rh, PP)
end

# (c) 2022 empirical sanity check: b_sp1 = 0.015 × CH1 at RH < 40%.
@printf "\n-- 2022 baseline: b_sp1 = 0.015×C0 => %.2f Mm⁻¹ at dry-state C0 --\n" PP.cal0015 * C0_hat

# ---- optional figures ----
# using Plots
# plt = plot_predictions(model);       savefig(plt, "fit.png")
# plt2 = plot_state_estimates(model);  savefig(plt2, "state.png")

# =============================================================================
# 6.  MULTI-UNIT EXTENSION  (collocated PurpleAir units)
#
#  Per Table S1 (2022), sensor-to-sensor CH1 spread in filtered air spans
#  0.10–377 across 42 units — per-sensor C0 + offset are essential alongside
#  the shared nonlinear correction.  Signature for MultiCustomDerivatives:
#      dudt_multi(u, i, x, p, t)     (i = series index, Int-valued)
#  data/X need a `series` column; parameters may carry per-series vectors.
# =============================================================================
#
# NSERIES = 4
# NN_m, NNm_p0 = SimpleNeuralNetwork(6, 1; hidden = 12)
#
# function residual_m(u, x, p)
#     inp = [u[1]/CH1_SCALE, x[1], x[2]/T_SCALE, x[3]/WIND_SCALE, x[4], x[5]/DRH_SCALE]
#     return NN_m(inp, p.NN)[1] * CH1_SCALE
# end
#
# function dudt_multi(u, i, x, p, t)
#     s   = round(Int, i)
#     k   = abs(p.log_k) + 1e-3
#     C0  = abs(p.C0[s])
#     off = p.offset[s]
#     gf  = growth_factor(Float64(x[1]), PP)
#     gf_d = growth_factor(PP.rh_dry, PP)
#     known = k * (( C0 * (gf / gf_d)^PP.p_scat + off ) - u[1])
#     return [known + residual_m(u, x, p)]
# end
#
# init_parameters_multi = (NN     = NNm_p0,
#                          log_k  = 1.0,
#                          C0     = fill(2600.0, NSERIES),
#                          offset = zeros(NSERIES))
#
# model_multi = MultiCustomDerivatives(data_multi, X_multi, dudt_multi,
#                                      init_parameters_multi;
#                                      time_column_name   = "time",
#                                      series_column_name = "series")
# train!(model_multi; loss_function = "derivative matching",
#        optimizer = "ADAM", optim_options = (maxiter = 3000,))
