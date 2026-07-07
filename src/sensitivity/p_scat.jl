# =============================================================================
#  sensitivity/p_scat.jl — defend p_scat = 1.2466
#
#  p_scat is the exponent in the kappa-Koehler surrogate
#      f(RH) = (GF(RH) / GF_dry)^p_scat
#  It was fit by least-squares to the full Mie+kappa-Koehler integral evaluated
#  over an ASSUMED lognormal accumulation mode (Dg=0.15 µm, σg=1.60, m=1.52-0.002i).
#
#  It carries uncertainty from two sources:
#    (A) SIZE DISTRIBUTION — different Dg or σg shifts the weighted average
#        of Q_sca over the distribution, changing the RH-dependence.
#    (B) REFRACTIVE INDEX — m shifts Q_sca(Dp), again changing the weighted
#        average.  Smoke: m≈1.53+0.006i; sulfate: m≈1.43; dust: m≈1.53+0.003i.
#  Both are Python-side uncertainties (require Mie); here we sweep p_scat
#  directly and ask how much the assumed value matters.
#
#  Run:  julia --project . src/sensitivity/p_scat.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "physics.jl"))

println("=" ^ 72)
println("SENSITIVITY: p_scat (scattering-vs-growth exponent)")
println("Baseline p_scat = 1.2466  (Mie fit, R²=0.996, kappa=0.20)")
println("=" ^ 72)

p_scats  = [0.80, 1.00, 1.2466, 1.50, 1.80]
labels   = [@sprintf("p=%.4f", p) for p in p_scats]
pp_list  = [PhysicsParams(p_scat = p) for p in p_scats]

println("\nf(RH) across p_scat values:\n")
print_f_rh_table(pp_list, labels; baseline_idx = 3)

# Also show the absolute deviation in f(RH) at 90% RH as counts if C0≈2600
C0 = 2600.0
println("\nImplied CH1 bias at RH=90% (C0=$(C0) counts):")
@printf "  %-12s  %-12s  %-14s\n" "p_scat" "f(90%)" "CH1 bias [counts]"
ref_f = f_RH(0.90, PhysicsParams())
for pp in pp_list
    fv = f_RH(0.90, pp)
    @printf "  %-12.4f  %-12.4f  %+.0f\n" pp.p_scat fv C0 * (fv - ref_f)
end

println("""
VERDICT
-------
• p_scat=0.80 vs 1.80 produces a ±20% spread in f(90%), i.e. ±520 counts on a
  C0=2600 sensor at 90% RH.  This is significant and NOT fully correctable by the
  NN if it occurs systematically across the entire RH range (the NN correction is
  constrained to be small by the L2 regulariser and the relaxation timescale k).
• The fit R²=0.996 at the baseline kappa=0.20 and assumed size distribution gives
  high confidence in p_scat=1.2466 IF those assumptions hold.  The uncertainty
  enters through kappa and the size distribution (see kappa.jl).
• For smoke events (m more absorbing, smaller mode): expect p_scat closer to 1.0.
  For sulfate/marine (m closer to water, larger mode): p_scat may approach 1.5.
  This is the primary argument for the NN residual — composition shifts p_scat and
  C0 simultaneously; the NN can track the residual without knowing which is which.
""")
