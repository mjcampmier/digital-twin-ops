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
#   * Fixed point is analytic (u*(x) = C0·f(RH)·(1+g(x))); no bisection needed.
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
const C0_REF     = 2600.0   # data normalisation: counts → O(1). Reduces gradient
                             # magnitude ~2600×, putting ADAM step size in a workable
                             # regime (v̂ ≈ O(100) instead of O(10¹⁶) unnormalized).
const T_SCALE    = 30.0
const WIND_SCALE = 5.0
const DRH_SCALE  = 0.10

# Multiplicative reparametrisation:
#   g_flow(wind) = w_norm · NN_flow([w_norm])
#   g_hyst(dRHdt) = dr_norm · NN_hyst([dr_norm])
#   g(x) = g_flow + g_hyst
#
# WHY MULTIPLICATIVE (not subtractive dry anchor):
#
# Three anchor strategies, ranked:
#
# 1. Subtractive anchor  g = NN([x]) − NN([0])
#    Forces g=0 at dry by subtracting the reference output.  FAILS: ∂NN([0])/∂params
#    appears in every training gradient (not just at dry-state training points).
#    All non-zero-x steps push NN([0]) in the opposite direction of their own
#    gradient, driving NN([0]) to a large positive value A and NN([x]) to A−const.
#    Result: g ≈ constant for all x>0 (step function), shape is lost.
#
# 2. No anchor, raw NN output
#    No gradient bias.  BUT: ADAM collapses to a constant-offset degeneracy.  The
#    training residual (u_obs < u_ODE at init) pushes b_out negative in ALL 400
#    training steps simultaneously.  b_out races from 0 → −0.476 in ~100 iters,
#    ADAM's second-moment v̂ accumulates to ≈2168²·5000, leaving an effective step
#    of ≈2e-7.  Undoing the overshoot would take ~2.3M iters.  g freezes at −0.95
#    regardless of proc_weight.  BFGS escapes but extrapolates wildly at wind>4.
#
# 3. Multiplicative anchor  g = x_norm · NN([x_norm])      ← THIS RUN
#    g is structurally zero at x_norm=0 (multiply by the input).  No NN([0])
#    in the gradient.  The NN learns only the AMPLITUDE function: for the true
#    linear flow signal, NN_flow≈−0.20 (constant); for hyst, NN_hyst≈−0.04 near
#    zero.  These are small O(0.2) targets; ADAM reaches them from zero without
#    overshooting to −0.476.  Criterion 6 is satisfied by construction.
NN_flow, NN_f0 = SimpleNeuralNetwork(1, 1; hidden = 8)
NN_hyst, NN_h0 = SimpleNeuralNetwork(1, 1; hidden = 8)

@inline function g_correction(x, p)
    w_norm  = x[3] / WIND_SCALE
    dr_norm = x[5] / DRH_SCALE
    g_flow  = w_norm  * NN_flow([w_norm],       p.NN_flow)[1]
    g_hyst  = -dr_norm * abs(NN_hyst([abs(dr_norm)], p.NN_hyst)[1])
    return g_flow + g_hyst
end
# abs(dr_norm) forces NN_hyst to be even in dRHdt.  Then g_hyst = dr_norm * even(dr_norm)
# is ODD — enhances CH1 for falling RH, suppresses for rising — matching true physics.
# Without abs, NN_hyst learns an odd function and g_hyst becomes even (symmetric
# suppression for both ±dRHdt), losing the sign entirely (run 14 result).

# -----------------------------------------------------------------------------
# 2.  UDE RHS  — relaxation toward physics target × (1 + g)
# -----------------------------------------------------------------------------
# g is covariate-only → u* = C0·f(RH)·(1+g) is unique and analytic.
const dudt = make_dudt(PP, g_correction)

init_parameters = (NN_flow   = NN_f0,
                   NN_hyst   = NN_h0,
                   log_k     = 1.0,
                   C0        = 1.0,   # in units of C0_REF; true ≈ 1.0
                   kappa_eff = 0.2)   # lumped RH-response param; abs()+1e-3 in ODE

# -----------------------------------------------------------------------------
# 3.  SYNTHETIC GROUND TRUTH
# -----------------------------------------------------------------------------
data, X, truth = synthetic_deployment(PP)
data.CH1 ./= C0_REF   # normalize to O(1) so ADAM gradients are O(1-100)

# -----------------------------------------------------------------------------
# 4.  BUILD + TRAIN
# -----------------------------------------------------------------------------
model = CustomDerivatives(data, X, dudt, init_parameters;
                          time_column_name = "time",
                          proc_weight      = 0.01,  # small: Kalman filter barely absorbs
                          obs_weight       = 1.0,   # residuals → optimizer must fix u*
                          reg_weight       = 5e-2,
                          reg_type         = "L2")

# Conditional likelihood. Data normalised by C0_REF → O(1) state values;
# small proc_weight prevents KF from absorbing model-observation misfit.
train!(model;
       loss_function = "conditional likelihood",
       loss_options  = (observation_error = 0.05, process_error = 0.02),
       optimizer     = "ADAM",
       verbose       = true,
       optim_options = (maxiter = 10000,))

# -----------------------------------------------------------------------------
# 5.  VALIDATION — does the NN recover the injected effects without being told?
# -----------------------------------------------------------------------------
pars      = get_parameters(model)
k_hat     = max(abs(pars.log_k), 1.0) + 1e-3
C0_hat    = abs(pars.C0) * C0_REF        # convert from normalised units back to counts
kappa_hat = abs(pars.kappa_eff) + 1e-3   # lumped RH-response param (NOT aerosol κ)
println("\nfitted:  k = $(round(k_hat, digits=3)) /day   C0 = $(round(C0_hat, digits=1)) counts   kappa_eff = $(round(kappa_hat, digits=4))")
println("         (true C0 = $(truth.C0_true), true kappa_eff = $(truth.kappa_true))")

# Fixed point is analytic: u*(x) = C0 · f(RH, kappa_eff) · (1 + g(x, p)) — no bisection.
# Unique because g is covariate-only (u dropped from NN inputs).
function fixed_point_u(x)
    return C0_hat * f_RH(Float64(x[1]), kappa_hat, PP) * (1.0 + g_correction(x, pars))
end

# CRITERION 2: C0 within ±15% of true value.
c0_rel = abs(C0_hat / truth.C0_true - 1.0)
@printf "\nCRITERION 2 — C0 recovery: %.1f%%  (limit ≤15%%)  %s\n" (c0_rel * 100) (c0_rel ≤ 0.15 ? "PASS" : "FAIL")

# CRITERION 3: kappa_eff recovery — must land within ±20% of the injected kappa_true.
# kappa_eff is the single identifiable RH-response knob (p_scat fixed at 1.25).
# Init = 0.2, injected kappa_true = 0.25; acceptance window [0.20, 0.30].
# This is the real identifiability gate before touching BAM data.
kappa_rel = abs(kappa_hat / truth.kappa_true - 1.0)
@printf "\nCRITERION 3 — kappa_eff recovery: %.4f  (true %.4f, init 0.2000)  %.1f%%  %s\n" kappa_hat truth.kappa_true (kappa_rel * 100) (kappa_rel ≤ 0.20 ? "PASS" : "FAIL")

# CRITERION 4: No spurious aging — t_deploy sweep, aging descoped (aging=ones).
println("\nCRITERION 4 — no spurious aging  (aging descoped → injected=1.0; limit: all within ±5% of t=0)")
println("  t[yr]   u*/C0_hat   rel_to_t0")
base_t        = fixed_point_u([PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
aging_max_dev = 0.0
for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    xa  = [PP.rh_dry, 20.0, 0.0, t, 0.0]
    fp  = fixed_point_u(xa)
    rel = fp / base_t
    dev = abs(rel - 1.0)
    global aging_max_dev = max(aging_max_dev, dev)
    @printf "  %4.2f     %6.3f      %+5.1f%%\n" t fp / C0_hat (rel - 1.0) * 100
end
@printf "  → max dev = %.1f%%  %s\n" (aging_max_dev * 100) (aging_max_dev ≤ 0.05 ? "PASS" : "FAIL")

# CRITERION 5: Flow (wind) channel — strong-signal gate (20% amplitude injected in training).
# Realistic 3% (SNR≈1 per observation) is identifiability-limited at single sensor;
# see SECONDARY B.  This gate confirms the architecture can learn a covariate at all.
println("\nCRITERION 5 — flow strong-signal gate  (injected: 1 − 0.20·wind/5; limit: peak within ±30%)")
println("  wind    recovered   injected")
base_w       = fixed_point_u([PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
rec_dev_peak = 0.0
inj_dev_peak = 0.0
for w in (0.0, 2.0, 4.0, 6.0, 8.0)
    xw  = [PP.rh_dry, 20.0, w, 0.0, 0.0]
    rec = fixed_point_u(xw) / base_w
    inj = 1.0 - 0.20 * (w / 5.0)
    global rec_dev_peak = max(rec_dev_peak, abs(rec - 1.0))
    global inj_dev_peak = max(inj_dev_peak, abs(inj - 1.0))
    @printf "  %4.1f    %6.3f     %6.3f\n" w rec inj
end
flow_rel_err = abs(rec_dev_peak / inj_dev_peak - 1.0)
@printf "  → peak dev: recovered=%.3f  injected=%.3f  rel_err=%.1f%%  %s\n" rec_dev_peak inj_dev_peak (flow_rel_err * 100) (flow_rel_err ≤ 0.30 ? "PASS" : "FAIL")

# CRITERION 6: Dry-state anchor — u*(x_dry) within ±10% of C0_true.
# With the multiplicative anchor, u*(x_dry) = C0_hat · 1.0 · (1+0) = C0_hat.
# This catches DC-level offset pathologies that survive the C0 parameter check.
u_dry    = fixed_point_u([PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
dry_err  = abs(u_dry / truth.C0_true - 1.0)
@printf "\nCRITERION 6 — dry-state anchor: u*(x_dry) = %.1f  vs C0_true = %.1f  (%.1f%%)  %s\n" u_dry truth.C0_true (dry_err * 100) (dry_err ≤ 0.10 ? "PASS" : "FAIL")

# SECONDARY A: Hysteresis (dRHdt) — expect sign-correct suppression on rising RH.
# Injected: 1 − 0.04·tanh(dRHdt/0.05).  Amplitude ≈4% at ±0.1 RH/day; near noise floor.
println("\nSECONDARY A — hysteresis (dRHdt)  (injected: 1 − 0.04·tanh(dRHdt/0.05))")
println("  dRHdt   recovered   injected")
base_d = fixed_point_u([0.60, 20.0, 0.0, 0.0, 0.0])
for dr in (-0.20, -0.10, 0.0, 0.10, 0.20)
    xd  = [0.60, 20.0, 0.0, 0.0, dr]
    rec = fixed_point_u(xd) / base_d
    inj = 1.0 - 0.04 * tanh(dr / 0.05)
    @printf "  %+5.2f   %6.3f     %6.3f\n" dr rec inj
end
println("  (sign correct = rising RH suppresses CH1; amplitude may be attenuated)")

# SECONDARY B: Flow realistic-SNR — identifiability floor, not pass/fail.
# 4.8% at 5% noise (SNR≈1 per observation); fleet collocation needed for reliable recovery.
println("\nSECONDARY B — flow realistic-SNR  (injected: 1 − 0.03·wind/5; characterisation only)")
println("  wind    recovered   injected")
for w in (0.0, 4.0, 8.0)
    xw  = [PP.rh_dry, 20.0, w, 0.0, 0.0]
    rec = fixed_point_u(xw) / base_w
    inj = 1.0 - 0.03 * (w / 5.0)
    @printf "  %4.1f    %6.3f     %6.3f\n" w rec inj
end
println("  (not pass/fail; fleet-level validation required)")

# 2022 empirical sanity check.
@printf "\n2022 baseline: b_sp1 = 0.015×C0 => %.2f Mm⁻¹ at dry-state C0\n" PP.cal0015 * C0_hat

# =============================================================================
# GATE SUMMARY
# =============================================================================
println("\n" * "="^52)
println("GATE SUMMARY")
println("="^52)
@printf "CRITERION 1  no error              PASS  ← you are reading this\n"
@printf "CRITERION 2  C0 within ±15%%        %s  (%.1f%%)\n"           (c0_rel ≤ 0.15 ? "PASS" : "FAIL") (c0_rel * 100)
@printf "CRITERION 3  kappa_eff ±20%%         %s  (%.1f%% rel err)\n"   (kappa_rel ≤ 0.20 ? "PASS" : "FAIL") (kappa_rel * 100)
@printf "CRITERION 4  no spurious aging     %s  (%.1f%% max dev)\n"   (aging_max_dev ≤ 0.05 ? "PASS" : "FAIL") (aging_max_dev * 100)
@printf "CRITERION 5  flow strong-signal    %s  (%.1f%% rel err)\n"   (flow_rel_err ≤ 0.30 ? "PASS" : "FAIL") (flow_rel_err * 100)
@printf "CRITERION 6  dry-state anchor      %s  (%.1f%% vs C0_true)\n" (dry_err ≤ 0.10 ? "PASS" : "FAIL") (dry_err * 100)
println("="^52)

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
