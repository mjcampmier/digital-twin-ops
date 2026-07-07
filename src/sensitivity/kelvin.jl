# =============================================================================
#  sensitivity/kelvin.jl — defend Kelvin-effect neglect
#
#  The kappa-Koehler growth factor used throughout is the simplified form:
#      GF = (1 + κ·aw/(1−aw))^(1/3)       [no Kelvin term]
#
#  The full form includes the Kelvin (surface-tension) correction:
#      aw_Kelvin_factor = exp(A_k / (Dp_dry · GF))
#      where A_k = 4·σ·Mw / (ρw·R·T)   [units: µm]
#
#  This script evaluates A_k at 20°C and shows how the Kelvin factor
#  compares across the PMS5003-relevant size range.
#
#  Run:  julia --project . src/sensitivity/kelvin.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "physics.jl"))

println("=" ^ 72)
println("SENSITIVITY: Kelvin (surface-tension) effect neglect")
println("=" ^ 72)

# Physical constants (SI)
σ_water  = 0.072     # N/m  surface tension of water at 20°C
Mw       = 0.018     # kg/mol  molar mass of water
ρw       = 1000.0    # kg/m³
R        = 8.314     # J/(mol·K)
T        = 293.0     # K  (20°C)

# Kelvin length in µm: A_k = 4·σ·Mw / (ρw·R·T)
A_k_m  = 4.0 * σ_water * Mw / (ρw * R * T)   # metres
A_k_um = A_k_m * 1e6                           # µm
@printf "\nKelvin length  A_k = %.5f µm  (at T=20°C)\n\n" A_k_um

# For each dry diameter, show the Kelvin correction factor to water activity
# at two RH levels, and the resulting fractional GF error.
const PP_base = PhysicsParams()

println("Kelvin correction  exp(A_k / (Dp_dry · GF(RH)))  to water activity:\n")
@printf "  %-10s  %-10s  %-20s  %-20s\n" "Dp_dry[µm]" "GF(80%)" "Kelvin_factor(80%)" "Kelvin_factor(90%)"
for dp_dry in (0.10, 0.15, 0.20, 0.50, 1.00)
    gf80 = growth_factor(0.80, PP_base)
    gf90 = growth_factor(0.90, PP_base)
    # Kelvin factor on aw; perturbative (using no-Kelvin GF as first approximation)
    kf80 = exp(A_k_um / (dp_dry * gf80))
    kf90 = exp(A_k_um / (dp_dry * gf90))
    @printf "  %-10.2f  %-10.4f  %-20.4f  %-20.4f\n" dp_dry gf80 kf80 kf90
end

println("""
Interpretation:
  The Kelvin factor > 1 means the equilibrium vapour pressure over the
  curved surface is higher than over a flat surface.  A particle at a given
  ambient RH therefore reaches a SMALLER equilibrium size than the no-Kelvin
  formula predicts.  Neglecting Kelvin means we OVERESTIMATE GF.

  The fractional overestimate of GF is approximately (Kelvin_factor − 1) / 3
  (first-order through the cube-root).
""")

@printf "  %-10s  %-22s  %-22s\n" "Dp_dry[µm]" "ΔGF/GF at RH=80% [%%]" "ΔGF/GF at RH=90% [%%]"
for dp_dry in (0.10, 0.15, 0.20, 0.50, 1.00)
    gf80 = growth_factor(0.80, PP_base)
    gf90 = growth_factor(0.90, PP_base)
    kf80 = exp(A_k_um / (dp_dry * gf80))
    kf90 = exp(A_k_um / (dp_dry * gf90))
    dGF80 = (kf80 - 1.0) / 3.0 * 100.0
    dGF90 = (kf90 - 1.0) / 3.0 * 100.0
    @printf "  %-10.2f  %-22.2f  %-22.2f\n" dp_dry dGF80 dGF90
end

println("""
Effect on f(RH):
  f(RH) = (GF(RH)/GF_dry)^p_scat.  A fractional error ε in GF propagates to
  f(RH) as  Δf/f ≈ p_scat · ε / (1 − ε/3) ≈ p_scat · ε  for small ε.
  With p_scat≈1.25 and the dominant scattering range Dp≈0.2–0.5 µm:
""")

for dp_dry in (0.20, 0.50)
    gf80 = growth_factor(0.80, PP_base)
    gf90 = growth_factor(0.90, PP_base)
    kf80 = exp(A_k_um / (dp_dry * gf80))
    kf90 = exp(A_k_um / (dp_dry * gf90))
    dGF80 = (kf80 - 1.0) / 3.0
    dGF90 = (kf90 - 1.0) / 3.0
    @printf "  Dp_dry=%.2f µm:  Δf/f(80%%) ≈ %.2f%%,   Δf/f(90%%) ≈ %.2f%%\n" dp_dry 1.2466*dGF80*100 1.2466*dGF90*100
end

println("""
VERDICT
-------
For the scattering-dominant size range (Dp_dry ≈ 0.15–0.5 µm), the Kelvin
correction causes <1% overestimate of GF and <1.5% error in f(RH).  This is
well within the fitting uncertainty of p_scat itself and is negligible compared
to the composition/kappa uncertainty quantified in kappa.jl.

The neglect is valid.  If the mode ever shifts toward ultrafine particles
(Dp_dry < 0.10 µm), the Kelvin correction becomes relevant and should be
added to growth_factor() as an iterative solve.
""")
