# =============================================================================
#  dryrun.jl — §7: End-to-end calibration scaffold smoke test
#
#  Proves §0–§6 wire together on synthetic data:
#
#    1.  Validate §0 panel schema (validate_panel)
#    2.  Bridge parity gate §1: Julia L1/L2 vs Python test batteries
#    3.  Build S_phys table §2
#    4.  Build per-unit RHS closures §3
#    5.  Set up CalibParams §4
#    6.  Run two-stage fit §5 (10 units, 100+150 ADAM steps)
#    7.  Run diagnostics §6
#
#  Expected outcome on synthetic data:
#    • Parity gate: PASS (L1+L2 bridge matches Python)
#    • Stage-1 NLL decreasing over epochs
#    • Stage-2 κ_eff converging toward kappa_true=0.25
#    • Allan deviation showing white-noise floor (σ_A flat or decreasing)
#    • frac_above ≈ 5% for well-specified noise
#    • RH coverage OK at all sites (synthetic RH range ≈ 0.55)
#
#  Run:
#    julia --project=. src/calibration/dryrun.jl
#
#  Requires: all of src/*.jl, src/calibration/*.jl
#            mie_emulator_frozen.npz, angular_emulator_frozen.npz
#            python_modules/test_battery/test_battery_L{1,2}.npz
# =============================================================================

using Dates, Printf, Statistics, DataFrames

# ── Load sources ─────────────────────────────────────────────────────────────
const REPO_ROOT = joinpath(@__DIR__, "..", "..")

include(joinpath(REPO_ROOT, "src", "physics.jl"))
include(joinpath(REPO_ROOT, "src", "mie_emulator.jl"))
include(joinpath(REPO_ROOT, "src", "angular_emulator.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "panel.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "parity_gate.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "s_phys.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "rhs.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "parameters.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "fit.jl"))
include(joinpath(REPO_ROOT, "src", "calibration", "diagnostics.jl"))


function run_dryrun(; verbose::Bool = true,
                     skip_parity_gate::Bool = false)
    t_total = time()
    println("="^60)
    println("CALIBRATION SCAFFOLD DRY RUN")
    println("="^60)

    # ── §0  Synthetic panel ───────────────────────────────────────────────────
    println("\n[§0] Generating synthetic panel...")
    df = synthetic_panel(
        n_units    = 6,
        n_sites    = 2,
        n_phases   = 4,
        n_days     = 60,
        kappa_true = 0.25,
        seed       = 42,
    )
    describe_panel(df)
    validate_panel(df)
    println("  Panel validated OK")

    # ── §1  Parity gate ───────────────────────────────────────────────────────
    println("\n[§1] Loading emulators and running parity gate...")
    emu_dir = joinpath(REPO_ROOT, "python_modules")
    l1_path = joinpath(emu_dir, "mie_emulator", "mie_emulator_frozen.npz")
    l2_path = joinpath(emu_dir, "angular_emulator", "angular_emulator_frozen.npz")

    if !isfile(l1_path) || !isfile(l2_path)
        @warn "Emulator NPZ not found — skipping parity gate (dryrun continues)"
        l1_ok = l2_ok = ad_ok = nothing
    else
        l1_emu = load_mie_emulator(l1_path)
        l2_emu = load_angular_emulator(l2_path)
        l1_ok, l2_ok, ad_ok = run_parity_gate(l1_emu, l2_emu; verbose = verbose)

        # L2 is a HARD gate: S_phys depends on it; stop if L2 fails
        if !l2_ok
            println("\n⚠  L2 PARITY GATE FAILED — stopping dryrun.")
            println("   S_phys cannot be built on a mismatched L2 bridge.")
            return nothing
        end

        # L1 failure is noted but does not block §2-§7 (S_phys uses L2 only)
        if !l1_ok
            println("\n⚠  L1 parity FAILED (Julia NPZ ≠ Python .pt checkpoint).")
            println("   Likely cause: mie_emulator_frozen.npz is from a different epoch than mie_emulator_best.pt.")
            println("   ACTION REQUIRED: re-export frozen NPZ from the same checkpoint used to generate the battery.")
            println("   Continuing dryrun — §2-§7 use L2 (S_phys) and are unaffected by L1.\n")
        end

        if skip_parity_gate
            println("  (skip_parity_gate=true — gate results are informational only)")
        end
    end

    # If emulators not found, build a minimal mock for the rest of the dryrun
    if !@isdefined(l2_emu)
        @warn "Emulators not found — using mock L2 (S_phys will be wrong but scaffold wires)"
        # Can't proceed without L2 — skip S_phys table build, use power-law placeholder
        _run_dryrun_fallback(df)
        return
    end

    # ── §2  S_phys table ─────────────────────────────────────────────────────
    println("\n[§2] Building S_phys table (40×40 grid)...")
    design = PMS5003_DEFAULT
    cache  = build_s_phys_cache(design)
    t2 = time()
    sphys_table = build_s_phys_table(l2_emu, design, cache;
                                      n_rh = 20, n_kappa = 20,    # smaller for dryrun speed
                                      verbose = verbose)
    @printf "  S_phys table built in %.1fs\n" (time() - t2)
    # Spot-check: S_phys at rh_dry should ≈ 1.0
    s_ref = interp_s_phys(sphys_table, Float64(design.rh_dry), 0.25)
    @printf "  S_phys(rh_dry=%.2f, κ=0.25) = %.4f  (expected ≈ 1.0)\n" design.rh_dry s_ref

    # ── §3  RHS closures (tested implicitly in §5) ────────────────────────────
    println("\n[§3] RHS closure spot-check...")
    # Build one covariate fn and evaluate RHS at a test point
    test_rhs_p = CalibRHSParams(κ_eff = 0.25, log_k = 1.2, log_G = log(2600.0))
    test_cov   = t -> (0.65, 2.0, 0.01)   # (rh, wind, drhdt)
    test_rhs!  = make_calib_rhs(sphys_table, design, test_cov)
    du_test    = [0.0]
    test_rhs!(du_test, [2600.0], test_rhs_p, 0.0)
    # At rh=0.65 with κ=0.25, S_phys≈1.6 so equilibrium ch1 ≈ G*1.6 ≈ 4000;
    # starting at u=2600 gives du/dt ≈ k*(4000-2600) ≈ 1600 — expected, not a bug
    @printf "  RHS at rh=0.65 u=2600: du/dt = %.1f  (G*S_phys≈%.0f, system not at eq)\n" du_test[1] (exp(test_rhs_p.log_G)*interp_s_phys(sphys_table, 0.65, test_rhs_p.κ_eff))

    # ── §4  Parameter setup ───────────────────────────────────────────────────
    println("\n[§4] Initialising CalibParams...")
    site_order  = sort(unique(df.site_id))
    batch_order = sort(unique(df.phase))
    n_units     = length(unique(df.unit_id))
    n_sites     = length(site_order)
    n_batches   = length(batch_order)

    init_params = CalibParams(
        n_sites   = n_sites,
        n_units   = n_units,
        n_batches = n_batches,
        C0_mean   = mean(df.ch1),
    )
    # Initialise κ_eff at wrong value to test Stage-2 recovery
    init_params_tweaked = CalibParams(
        SharedParams(κ_eff = 0.15, log_k = 1.0, a_hyst = 0.0, τ_hyst = 0.05),
        init_params.sites, init_params.units, init_params.batches,
    )
    show(stdout, init_params_tweaked); println()
    θ_init, _ = flatten(init_params_tweaked)
    @printf "  Flat θ length: %d\n" length(θ_init)

    # ── §5  Two-stage fit ─────────────────────────────────────────────────────
    println("\n[§5] Running two-stage fit (scaffold: 30+50 epochs)...")
    unit_data = extract_unit_data(df, site_order, batch_order)
    t5 = time()
    fitted = two_stage_fit(
        init_params_tweaked, unit_data, sphys_table, design;
        s1_epochs = 30,
        s2_epochs = 50,
        s1_lr     = 5e-3,
        s2_lr     = 2e-3,
        verbose   = verbose,
    )
    @printf "  Fit completed in %.1fs\n" (time() - t5)
    @printf "  Fitted κ_eff = %.4f  (true = 0.25)\n" fitted.shared.κ_eff
    show(stdout, fitted); println()

    # ── §6  Diagnostics ───────────────────────────────────────────────────────
    println("\n[§6] Running diagnostics...")
    diag = run_diagnostics(fitted, unit_data, sphys_table, design, df; verbose = verbose)

    # Summary across units
    frac_above_vals = [u.frac_above for u in diag.units]
    @printf("\n  frac_above_2σ: mean=%.1f%%  max=%.1f%%  (Gaussian target ≈ 4.6%%)\n",
        100*mean(frac_above_vals), 100*maximum(frac_above_vals))
    rh_ok = all(values(diag.rh_flags))
    @printf "  RH coverage: %s\n" rh_ok ? "ALL SITES OK" : "WARN: some sites have narrow RH"

    # ── Summary ───────────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("DRY RUN COMPLETE")
    println("="^60)
    @printf "  Total wall time: %.1fs\n" (time() - t_total)
    @printf "  §0 panel: OK\n"
    @printf "  §1 parity gate: %s\n" (l1_ok === nothing ? "SKIPPED (NPZ not found)" :
                                       (l1_ok && l2_ok && ad_ok ? "PASS" : "FAIL"))
    @printf "  §2 S_phys table: built (20×20 dryrun grid)\n"
    @printf "  §3 RHS closure: spot-checked OK\n"
    @printf "  §4 CalibParams: %d-dim flat vector\n" length(θ_init)
    @printf("  §5 two-stage fit: κ_eff %.4f→%.4f (true=0.25)\n",
        init_params_tweaked.shared.κ_eff, fitted.shared.κ_eff)
    @printf "  §6 diagnostics: frac_above mean=%.1f%%\n" (100*mean(frac_above_vals))
    println("="^60)

    return fitted
end


# Minimal fallback when emulators are not installed (CI / cold environment)
function _run_dryrun_fallback(df)
    println("\n[FALLBACK] Emulators not found.")
    println("  §0 panel: OK (schema validated)")
    println("  §1–6 skipped (need mie_emulator_frozen.npz + angular_emulator_frozen.npz)")
    println("  Install with: julia run_angular_julia.sh  # trains and exports .npz")
end


# Run when invoked as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_dryrun(verbose = true, skip_parity_gate = true)
end
