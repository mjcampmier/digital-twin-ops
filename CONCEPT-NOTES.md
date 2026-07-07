# PMS5003 Digital Twin via UniversalDiffEq.jl — Concept Notes

## Goal
Build a state-space universal differential equation (UDE) digital twin of the Plantower
PMS5003 aerosol sensor: known physics for the mechanistic transfer function, a neural
network for the residual (aging, composition uncertainty, RH hysteresis, flow-impedance
effects). Implementation target: `UniversalDiffEq.jl` (Buckner et al.), likely via
`CustomDerivatives` or `MultiCustomDerivatives` if fitting across several collocated units.

## Source material (two papers, and they disagree with each other — important)

1. **Ouimette et al. 2022, AMT** — "Evaluating the PurpleAir monitor as an aerosol light
   scattering instrument." Treats the PMS5003 as a **cell-reciprocal nephelometer**:
   ensemble measurement, laser modeled as a constant ~1 mm diameter beam, CH1 linearly
   proportional to bulk scattering coefficient:
   `b_sp1 = 0.015 × CH1` (Mm⁻¹), r²=0.97 over 4 orders of magnitude, RH < 40%.
   This is a **useful empirical baseline**, not the mechanistic ground truth.

2. **Ouimette et al. 2024, Aerosol Sci. Technol.** — "Fundamentals of low-cost aerosol
   sensor design and operation." Overturns the 2022 assumption using direct oscilloscope
   measurements of the photodiode. Key finding: the PMS5003 is actually an **imperfect
   optical particle counter**, not a nephelometer:
   - Laser is a **focused Gaussian beam**, waist radius w₀ = 17.5 µm, Rayleigh length
     z₀ = 0.288 mm (not a uniform 1 mm cylinder).
   - Photodiode output is discrete single-particle pulses (20–800 µs wide), not a
     slowly-varying ensemble signal.
   - >99% of particles miss the laser entirely; particles that do intersect it usually
     miss the narrow focal point and are **systematically undersized** because pulse
     amplitude depends on *where* the particle crossed the beam, not just its diameter.
   - Particle sizing efficiency (probability of being assigned the correct size bin):
     `PSE(Dp) = exp[−3.22 · log(Dp / 0.30 µm)]`  (Eq. 16)
   - Mass scattering efficiency is systematically truncated relative to the ideal
     nephelometer value by a factor equal to PSE(Dp) (Eq. 14).
   - CFD shows aspiration efficiency into the PurpleAir housing itself is ~100% for
     particles < 1 µm at all tested wind speeds (0.4–20 m/s), degrading for larger
     particles at higher wind speed (50% cut size ~15 µm at 0.4 m/s, ~2.5 µm at 20 m/s).

**Recommendation:** use the 2024 mechanistic model (Mie scattering integral + Gaussian
beam geometry + PSE(Dp)) as the "known physics" term. Use the 2022 linear CH1↔b_sp1
relationship only as a sanity-check baseline, since it's empirical and RH/composition-
dependent, not a first-principles description.

## What the known physics term should encode
- Mie scattering amplitude functions S₁(θ), S₂(θ) as a function of particle diameter Dp,
  refractive index m, wavelength λ=657 nm (2024 paper, Eqs. 3–11 give the full geometric
  integral over beam position; this is the generative single-particle transfer function).
- Given an assumed/estimated aerosol size distribution and refractive index, the model
  predicts CH1 via the mass-scattering-efficiency transfer function (Eqs. 13–14).
- RH-driven hygroscopic growth: static κ-Köhler / Petters-Kreidenweis parameterization
  gives density, diameter growth, and refractive index shift as a function of RH — this
  is a *known, static* relationship, good candidate for the deterministic part of the ODE.

## What should be left to the neural network (the actual unknowns)
1. **Aging/drift** — Fig. S15 (2022 paper) shows ~10% sensitivity degradation over one
   year in the field. Plausible as a slowly-evolving latent state.
2. **Composition/refractive-index uncertainty** — the model assumes a fixed m
   (e.g., 1.52+0.002i); real ambient aerosol (dust vs. smoke vs. sulfate) shifts both
   true scattering and the PSE(Dp) curve. Not observed directly in the field — this is
   probably the single largest source of unmodeled nonlinearity. Could be covariate-driven
   using indirect proxies (season, back-trajectory cluster, satellite AOD speciation, fire
   activity flags).
3. **RH hysteresis** — the static κ-Köhler formula misses deliquescence/efflorescence
   path-dependence; a real hygroscopic aerosol takes a different growth curve depending on
   whether RH is rising or falling.
4. **Flow-impedance / fan effects** — CH1 was empirically very sensitive to inlet pressure
   (30% reduction at 1 Pa impedance, 83% at 2.5 Pa). In the field this could map onto wind
   loading on the inlet, temperature-dependent fan curve, or partial inlet clogging —
   a plausible slow multiplicative bias state.
5. **Wind-speed-dependent aspiration** — there's already a semi-empirical CFD curve
   (Fig. 8, 2024 paper) for this; could fold into the known term if wind speed is
   available as a covariate, or let the NN learn/correct it if the CFD geometry doesn't
   match your actual deployment orientation/housing.

## Sketch of state/covariate structure
```julia
# state u = [latent CH1 calibration factor] or [CH1_predicted] depending on
# whether continuous reference data is available
function dudt(u, p, t, X)
    # X = covariates at time t: RH, T, wind_speed, reference_PM_or_bsp (if available),
    #     time_since_deployment, season/back-trajectory proxy
    known_physics = mie_transfer_function(X.size_dist_proxy, X.RH, p.refractive_index) - u[1]
    residual = NN(u, X, p.NN)[1]   # aging, composition, RH hysteresis, flow impedance
    return [known_physics + residual]
end
```
If a measured aerosol size distribution isn't available in the field (which is somewhat
the point — the PMS5003's own six size bins are known to be unreliable, see Table S4/S5
in the 2022 paper, r²=0.997 between CH1 and CH2, meaning the other bins carry almost no
independent information), `mie_transfer_function` can collapse to the RH/composition-
driven mass-scattering-efficiency curve (Fig. 5 / Eq. 14 in the 2024 paper) evaluated
against an assumed background composition, with the NN correcting for that assumption
being wrong.

## Validation checks once trained
- Use `get_right_hand_side` (or equivalent introspection) to extract the NN's learned
  function of RH and time-since-deployment, and compare its shape against:
  - the independently-measured aging curve (Fig. S15, 2022 paper, ~10%/year)
  - the RH degradation curve (Fig. S54, 2024 paper, ~60% MSE drop from 21% to 89% RH)
  If the fitted residual reproduces those independently-measured effects without being
  told to, that's good evidence the decomposition is doing real work rather than just
  absorbing noise.
- Cross-check predicted CH1 against the simple empirical b_sp1 = 0.015 × CH1 relationship
  from the 2022 paper as an outside sanity check, especially at RH < 40% where that
  relationship was validated.

## Open questions / caveats to flag to Fable
- Refractive index and true size distribution are not directly observable in most field
  deployments — the whole exercise is partly about how much can be recovered/corrected
  without them, using only covariates.
- The two papers' models disagree on mechanism (nephelometer vs. OPC) even though both
  fit their own field/lab data well — worth deciding explicitly which one is the
  "known physics" backbone before writing any ODE, rather than blending them ad hoc.
- Sensor-to-sensor variability is large even nominally-identical units (Table S1, 2022
  paper: CH1 in filtered air ranged 0.10 to 377 across 42 sensors) — if fitting across
  multiple collocated units, per-sensor offset/scale parameters are probably necessary
  in addition to the shared nonlinear correction (this is what `MultiCustomDerivatives`
  is for).
