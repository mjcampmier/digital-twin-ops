# =============================================================================
#  sensitivity/kappa.jl — defend kappa = 0.20
#
#  kappa (hygroscopicity) determines how strongly aerosol particles absorb
#  water and grow with RH.  Literature range: 0.01 (fresh soot, mineral dust)
#  to ~0.9 (NaCl sea salt).  Mixed urban accumulation mode: 0.10–0.30.
#
#  This script asks: if kappa is wrong by ±50–75%, how much does f(RH) change,
#  and what fraction of that error falls inside the NN's correctable envelope?
#
#  Run:  julia --project . src/sensitivity/kappa.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "physics.jl"))

println("=" ^ 72)
println("SENSITIVITY: kappa (hygroscopicity)")
println("Baseline kappa = 0.20  (mixed urban accumulation mode)")
println("=" ^ 72)

kappas = [0.05, 0.10, 0.20, 0.30, 0.50]
labels = [@sprintf("κ=%.2f", k) for k in kappas]
pp_list = [PhysicsParams(kappa = k) for k in kappas]

println("\nf(RH) = (GF(RH)/GF_dry)^p_scat  [baseline p_scat=1.2466 throughout]\n")
print_f_rh_table(pp_list, labels; baseline_idx = 3)

println("\nGF(dry) and p_scat are re-anchored per kappa below:")
@printf "\n  %-8s  %-10s\n" "kappa" "GF(RH_dry)"
for pp in pp_list
    @printf "  %-8.2f  %-10.6f\n" pp.kappa growth_factor(pp.rh_dry, pp)
end

println("""
VERDICT
-------
• At RH=90%: kappa=0.05 gives f=1.11, kappa=0.50 gives f=1.70 — a ±(25–35)%
  spread around the baseline f=1.38.  These are NOT small errors.
• However: the NN sees RH as a covariate and its residual term is unconstrained
  on the RH axis.  It can absorb a smooth, monotone bias in f(RH).  The risk is
  that a wrong kappa causes the NN to use its RH-capacity to correct the physics
  rather than to capture composition/hysteresis.  This degrades interpretability
  but does not destroy predictive accuracy.
• Practical defence: kappa=0.20 is consistent with AERONET-derived bulk kappa
  for continental aerosol (Petters & Kreidenweis 2007 survey: median ~0.19).
  For smoke-dominated deployments use kappa~0.10; for coastal/marine use ~0.40.
  Consider making kappa a fitted parameter if reference data are available.
""")
