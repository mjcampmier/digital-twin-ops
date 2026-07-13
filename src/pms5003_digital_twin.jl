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
include(joinpath(@__DIR__, "mie_emulator.jl"))
include(joinpath(@__DIR__, "mie_physics.jl"))
include(joinpath(@__DIR__, "synthetic.jl"))

using UniversalDiffEq, DataFrames, Lux, ComponentArrays, Random, Statistics, Printf
# using Plots   # uncomment for diagnostic figures at the bottom

Random.seed!(20260707)

# Baseline physics — all assumptions at their defended default values.
const PP = PhysicsParams()

# -----------------------------------------------------------------------------
# 1.  NEURAL RESIDUAL
# -----------------------------------------------------------------------------
const C0_REF    = 2600.0   # data normalisation: counts → O(1)
const DRH_SCALE = 0.10
const W_THRESH  = 5.0     # binary gust gate threshold [m/s]; fixed physical constant, NOT fitted

# -----------------------------------------------------------------------------
# 1.  NEURAL RESIDUAL
# -----------------------------------------------------------------------------
# g = g_flow + g_hyst
#
# g_flow  — binary gust gate: a_flow · 𝟙(wind > W_THRESH)
#   a_flow is a single signed scalar (init 0).  No NN, no continuous shape.
#   Rationale: ERA5 wind (~25 km grid, no sub-hourly turbulence) and enclosure
#   distortion mean a continuous NN_flow was fitting shape the covariate cannot
#   resolve.  Binary matches the real information content and eliminates the
#   κ_eff ↔ flow degeneracy that caused C5=140% in run 20.
#
# g_hyst  — multiplicative NN anchor: −dr_norm · |NN_hyst([|dr_norm|])|
#   abs on input  → NN_hyst is even in dr_norm → g_hyst is ODD (correct sign).
#   abs on output → amplitude always non-negative; leading − encodes known
#   physics direction (rising RH suppresses, falling enhances).
#   Multiplicative zero at dr_norm=0: dry anchor preserved by construction.
NN_hyst, NN_h0 = SimpleNeuralNetwork(1, 1; hidden = 8)

@inline function g_correction(x, p)
    dr_norm = x[5] / DRH_SCALE
    g_flow  = p.a_flow * Float64(x[3] > W_THRESH)          # binary; no AD through indicator
    g_hyst  = -dr_norm * abs(NN_hyst([abs(dr_norm)], p.NN_hyst)[1])
    return g_flow + g_hyst
end

# -----------------------------------------------------------------------------
# 2.  UDE RHS  — relaxation toward physics target × (1 + g)
# -----------------------------------------------------------------------------
# g is covariate-only → u* = C0·f(RH,κ_eff)·(1+g) is unique and analytic.
#
# Two-stage κ_eff release:
#   Stage 1 — dudt_s1 locks κ_eff at KAPPA_INIT; gradient w.r.t. p.kappa_eff = 0
#             so ADAM leaves it unchanged.  C0, log_k, a_flow, NN_hyst converge
#             with κ_eff pinned → C0 is well-determined before κ_eff is free.
#   Stage 2 — dudt_s2 uses p.kappa_eff; warm-starts from stage-1 params.
const KAPPA_INIT = 0.20
const dudt_s1    = make_dudt_locked(PP, g_correction, KAPPA_INIT)
const dudt_s2    = make_dudt(PP, g_correction)

init_parameters = (NN_hyst   = NN_h0,
                   log_k     = 1.0,
                   C0        = 1.0,   # in units of C0_REF; true ≈ 1.0
                   kappa_eff = KAPPA_INIT,
                   a_flow    = 0.0)   # signed scalar; init 0 (no effect at start)

# -----------------------------------------------------------------------------
# 3.  SYNTHETIC GROUND TRUTH
# -----------------------------------------------------------------------------
data, X, truth = synthetic_deployment(PP)
data.CH1 ./= C0_REF   # normalize to O(1) so ADAM gradients are O(1-100)

# -----------------------------------------------------------------------------
# 4.  BUILD + TRAIN
# -----------------------------------------------------------------------------
ude_kwargs = (time_column_name = "time",
              proc_weight      = 0.01,
              obs_weight       = 1.0,
              reg_weight       = 5e-2,
              reg_type         = "L2")

train_kwargs = (loss_function = "conditional likelihood",
                loss_options  = (observation_error = 0.05, process_error = 0.02),
                optimizer     = "ADAM",
                verbose       = true)

# Stage 1 — κ_eff locked; fit C0, log_k, a_flow, NN_hyst
println("\n--- Stage 1: κ_eff locked at $(KAPPA_INIT) ---")
model_s1 = CustomDerivatives(data, X, dudt_s1, init_parameters; ude_kwargs...)
train!(model_s1; train_kwargs..., optim_options = (maxiter = 1000,))
pars_s1 = get_parameters(model_s1)

# Stage 2 — κ_eff free; warm-start from stage 1
println("\n--- Stage 2: κ_eff released ---")
model_s2 = CustomDerivatives(data, X, dudt_s2, pars_s1; ude_kwargs...)
train!(model_s2; train_kwargs..., optim_options = (maxiter = 1000,))

# -----------------------------------------------------------------------------
# 5.  VALIDATION — does the NN recover the injected effects without being told?
# -----------------------------------------------------------------------------
pars       = get_parameters(model_s2)
k_hat      = max(abs(pars.log_k), 1.0) + 1e-3
C0_hat     = abs(pars.C0) * C0_REF
kappa_hat  = abs(pars.kappa_eff) + 1e-3   # lumped RH-response param (NOT aerosol κ)
a_flow_hat = pars.a_flow                   # signed scalar
println("\nfitted:  k = $(round(k_hat, digits=3)) /day   C0 = $(round(C0_hat, digits=1)) counts   kappa_eff = $(round(kappa_hat, digits=4))   a_flow = $(round(a_flow_hat, digits=4))")
println("         (true C0 = $(truth.C0_true), true kappa_eff = $(truth.kappa_true), true a_flow = $(truth.a_flow_true))")

# Fixed point is analytic: u*(x) = C0 · f(RH, kappa_eff) · (1 + g(x, p)) — no bisection.
# Unique because g is covariate-only (u dropped from NN inputs).
function fixed_point_u(x)
    return C0_hat * f_RH(Float64(x[1]), kappa_hat, PP) * (1.0 + g_correction(x, pars))
end

# CRITERION 2: C0 within ±15% of true value.
c0_rel = abs(C0_hat / truth.C0_true - 1.0)
@printf "\nCRITERION 2 — C0 recovery: %.1f%%  (limit ≤15%%)  %s\n" (c0_rel * 100) (c0_rel ≤ 0.15 ? "PASS" : "FAIL")

# CRITERION 3: kappa_eff recovery — within ±20% of kappa_true.
# Non-blocking for a residual miss attributable to the κ_eff/C0 level trade;
# that degeneracy is broken by the BAM absolute-level anchor on real data.
# Two-stage training should shrink the error; report honestly either way.
kappa_rel = abs(kappa_hat / truth.kappa_true - 1.0)
kappa_c3  = kappa_rel ≤ 0.20
@printf "\nCRITERION 3 — kappa_eff recovery: %.4f  (true %.4f, init %.4f)  %.1f%%  %s\n" kappa_hat truth.kappa_true KAPPA_INIT (kappa_rel * 100) (kappa_c3 ? "PASS" : "FAIL (non-blocking: κ/C0 ridge, deferred to BAM anchor)")

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

# CRITERION 5: a_flow recovery — recovered scalar within ±30% of a_flow_true.
# Binary gate: g_flow = a_flow · 𝟙(wind > W_THRESH).  Single amplitude, not a curve.
a_flow_rel = abs(a_flow_hat / truth.a_flow_true - 1.0)
flow_c5    = a_flow_rel ≤ 0.30
@printf "\nCRITERION 5 — a_flow recovery (binary gate, w_thresh=%.1f m/s): %.4f  (true %.4f)  %.1f%%  %s\n" W_THRESH a_flow_hat truth.a_flow_true (a_flow_rel * 100) (flow_c5 ? "PASS" : "FAIL")
@printf "  samples above threshold: %.1f%%  (if small, a_flow is weakly identified — expected)\n" (truth.frac_above * 100)

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

# SECONDARY B: Flow identifiability — fraction of samples above threshold.
# Binary gate has no continuous shape to fit; a_flow is identified only by above-threshold obs.
# If frac_above is small (<10%), a_flow is weakly identified even with correct architecture.
@printf "\nSECONDARY B — flow identifiability: %.1f%% of samples above w_thresh=%.1f m/s\n" (truth.frac_above * 100) W_THRESH
println("  (not pass/fail; low frac_above → a_flow weakly identified; fleet collocation helps)")

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
@printf "CRITERION 5  a_flow within ±30%%    %s  (%.1f%% rel err, frac_above=%.1f%%)\n" (flow_c5 ? "PASS" : "FAIL") (a_flow_rel * 100) (truth.frac_above * 100)
@printf "CRITERION 6  dry-state anchor      %s  (%.1f%% vs C0_true)\n" (dry_err ≤ 0.10 ? "PASS" : "FAIL") (dry_err * 100)
println("="^52)

# ---- optional figures ----
# using Plots
# plt = plot_predictions(model);       savefig(plt, "fit.png")
# plt2 = plot_state_estimates(model);  savefig(plt2, "state.png")

# =============================================================================
# 7.  STAGE 3 — Mie-physics f(RH)
#
#  Replaces the p_scat power-law with Mie-integrated f(RH):
#      f_RH_mie(RH, κ) = ∫ Q_sca(x_wet(D,κ)) D_wet² n(D) dD
#                        ─────────────────────────────────────
#                        ∫ Q_sca(x_ref(D,κ)) D_ref² n(D) dD
#
#  ∂f_RH/∂kappa_eff now flows through Mie scattering physics via
#  mie_forward_zyg (non-mutating, Zygote-compatible).
#  Warm-start from Stage 1 parameters — only kappa_eff and fine-tuning needed.
# =============================================================================
const MIE_NPZ  = joinpath(@__DIR__, "..", "python_modules", "mie_emulator", "mie_emulator_frozen.npz")
const MIE_JLD2 = joinpath(@__DIR__, "..", "python_modules", "mie_emulator", "mie_emulator_best.jld2")

_mie_src = isfile(MIE_JLD2) ? MIE_JLD2 : (isfile(MIE_NPZ) ? MIE_NPZ : nothing)

if !isnothing(_mie_src)
    println("\n" * "="^52)
    println("STAGE 3 — Mie-physics f(RH)  [$(basename(_mie_src))]")
    println("="^52)

    emu = load_mie_emulator(_mie_src)
    psd = LognormalPSD(rh_dry = PP.rh_dry)   # align reference RH with PhysicsParams

    # Diagnostic: compare power-law vs Mie f(RH) at the stage-2 kappa_hat
    println("\nf(RH) comparison at kappa_eff = $(round(kappa_hat, digits=4)):")
    compare_f_RH(PP, kappa_hat, emu, psd)

    # Build f(RH) table once — Zygote differentiates through bilinear interp,
    # not through the 9-layer Mie MLP, making Stages 3a and 3b fast.
    println("\nBuilding Mie f(RH) table (80×80 grid)...")
    mie_tbl = build_mie_frh_table(emu, psd)
    println("  done.")

    # Stage 3a — kappa_eff locked (same discipline as Stage 1)
    println("\n--- Stage 3a: Mie f(RH), κ locked at $(KAPPA_INIT) ---")
    dudt_mie_s1 = make_dudt_mie_locked(PP, g_correction, emu, psd, KAPPA_INIT;
                                        rh_data = X.RH)
    model_mie_s1 = CustomDerivatives(data, X, dudt_mie_s1, init_parameters; ude_kwargs...)
    train!(model_mie_s1; train_kwargs..., optim_options = (maxiter = 1000,))
    pars_mie_s1 = get_parameters(model_mie_s1)

    # Stage 3b — κ released; warm-start from 3a; gradient via table interp
    println("\n--- Stage 3b: Mie f(RH), κ free (table lookup) ---")
    dudt_mie_s2 = make_dudt_mie(PP, g_correction, mie_tbl)
    model_mie_s2 = CustomDerivatives(data, X, dudt_mie_s2, pars_mie_s1; ude_kwargs...)
    train!(model_mie_s2; train_kwargs..., optim_options = (maxiter = 1000,))

    pars_mie     = get_parameters(model_mie_s2)
    kappa_mie    = abs(pars_mie.kappa_eff) + 1e-3
    C0_mie       = abs(pars_mie.C0) * C0_REF
    kappa_rel_mie = abs(kappa_mie / truth.kappa_true - 1.0)
    c0_rel_mie    = abs(C0_mie / truth.C0_true - 1.0)

    println("\n" * "="^52)
    println("STAGE 3 RESULTS  (data generated with power-law f(RH))")
    println("="^52)
    @printf "  kappa_eff  Mie-physics: %.4f   power-law: %.4f   true: %.4f\n" kappa_mie kappa_hat truth.kappa_true
    @printf "  kappa_eff  Mie rel-err: %.1f%%  power-law rel-err: %.1f%%\n" (kappa_rel_mie*100) (kappa_rel*100)
    @printf "  C0         Mie-physics: %.1f    power-law: %.1f    true: %.1f\n" C0_mie C0_hat truth.C0_true
    @printf "  C0         Mie rel-err: %.1f%%  power-law rel-err: %.1f%%\n" (c0_rel_mie*100) (c0_rel*100)
    println("  (κ bias expected: Mie f(RH) has different shape than power-law truth)")
    println("="^52)

    # =========================================================================
    # STAGE 4 — Fair comparison: both models on Mie-generated data
    #
    #  Regenerate synthetic data using mie_f_RH (via table) as the truth.
    #  Now the ground truth IS Mie physics, so we can fairly ask:
    #  does Mie-UDE recover κ better than power-law-UDE?
    # =========================================================================
    println("\n" * "="^52)
    println("STAGE 4 — κ recovery when data IS Mie-generated")
    println("="^52)

    data_m, X_m, truth_m = synthetic_deployment(PP;
        f_rh_fn = (rh, k) -> interp_frh(mie_tbl, rh, k))
    data_m.CH1 ./= C0_REF

    # 4a: power-law model on Mie data
    println("\n--- Stage 4a: power-law f(RH) on Mie data, κ locked ---")
    model_4a = CustomDerivatives(data_m, X_m, dudt_s1, init_parameters; ude_kwargs...)
    train!(model_4a; train_kwargs..., optim_options = (maxiter = 1000,))
    pars_4a = get_parameters(model_4a)

    println("\n--- Stage 4b: power-law f(RH) on Mie data, κ free ---")
    model_4b = CustomDerivatives(data_m, X_m, dudt_s2, pars_4a; ude_kwargs...)
    train!(model_4b; train_kwargs..., optim_options = (maxiter = 1000,))
    pars_4b  = get_parameters(model_4b)
    kappa_4b = abs(pars_4b.kappa_eff) + 1e-3
    C0_4b    = abs(pars_4b.C0) * C0_REF

    # 4c: Mie model on Mie data, κ locked
    dudt_mie_4a = make_dudt_mie_locked(PP, g_correction, emu, psd, KAPPA_INIT;
                                        rh_data = X_m.RH)
    println("\n--- Stage 4c: Mie f(RH) on Mie data, κ locked ---")
    model_4c = CustomDerivatives(data_m, X_m, dudt_mie_4a, init_parameters; ude_kwargs...)
    train!(model_4c; train_kwargs..., optim_options = (maxiter = 1000,))
    pars_4c = get_parameters(model_4c)

    # 4d: Mie model on Mie data, κ free (table lookup)
    println("\n--- Stage 4d: Mie f(RH) on Mie data, κ free (table) ---")
    model_4d = CustomDerivatives(data_m, X_m, dudt_mie_s2, pars_4c; ude_kwargs...)
    train!(model_4d; train_kwargs..., optim_options = (maxiter = 1000,))
    pars_4d  = get_parameters(model_4d)
    kappa_4d = abs(pars_4d.kappa_eff) + 1e-3
    C0_4d    = abs(pars_4d.C0) * C0_REF

    kappa_rel_4b = abs(kappa_4b / truth_m.kappa_true - 1.0)
    kappa_rel_4d = abs(kappa_4d / truth_m.kappa_true - 1.0)
    c0_rel_4b    = abs(C0_4b / truth_m.C0_true - 1.0)
    c0_rel_4d    = abs(C0_4d / truth_m.C0_true - 1.0)

    println("\n" * "="^52)
    println("STAGE 4 RESULTS  (data generated with Mie f(RH))")
    println("="^52)
    @printf "  true: κ=%.4f  C0=%.1f\n" truth_m.kappa_true truth_m.C0_true
    println("-"^52)
    @printf "  power-law UDE:  κ=%.4f (%.1f%% err)  C0=%.1f (%.1f%% err)\n" kappa_4b (kappa_rel_4b*100) C0_4b (c0_rel_4b*100)
    @printf "  Mie-physics UDE: κ=%.4f (%.1f%% err)  C0=%.1f (%.1f%% err)\n" kappa_4d (kappa_rel_4d*100) C0_4d (c0_rel_4d*100)
    println("-"^52)
    @printf "  κ improvement from Mie physics: %.1f pp  (%s)\n" ((kappa_rel_4b - kappa_rel_4d)*100) (kappa_rel_4d < kappa_rel_4b ? "Mie better" : "power-law better")
    @printf "  C0 improvement from Mie physics: %.1f pp\n" ((c0_rel_4b - c0_rel_4d)*100)
    println("  CRITERION 3 power-law: $(kappa_rel_4b ≤ 0.20 ? "PASS" : "FAIL")  ($(round(kappa_rel_4b*100,digits=1))%)")
    println("  CRITERION 3 Mie:       $(kappa_rel_4d ≤ 0.20 ? "PASS" : "FAIL")  ($(round(kappa_rel_4d*100,digits=1))%)")
    println("="^52)

else
    println("\nStage 3 skipped — no Mie emulator found at:")
    println("  JLD2: $MIE_JLD2")
    println("  NPZ:  $MIE_NPZ")
    println("Run src/train_mie_emulator.jl to generate the Julia checkpoint.")
end

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
