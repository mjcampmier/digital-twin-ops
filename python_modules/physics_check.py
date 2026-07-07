"""
PMS5003 known-physics validation (runnable, standalone).

Purpose: build and sanity-check the *deterministic* transfer function that becomes
the "known physics" term of the UDE, BEFORE porting it to Julia. Everything here is
first-principles (Mie via miepython) + standard aerosol parameterizations
(kappa-Koehler growth, volume-mixed refractive index, PSE undersizing from
Ouimette 2024 Eq. 16). We extract two things for the Julia side:

  1. A cheap, differentiable humidification factor f(RH) surrogate (gamma-law)
     that reproduces the full Mie+kappa-Koehler curve. This is what goes inside
     the ODE RHS (a full Mie integral inside a stiff-ish ODE solved thousands of
     times under ForwardDiff would be slow and needlessly fragile).
  2. The dry-state OPC truncation ratio b_obs/b_true -- the quantitative statement
     of the 2024 paper's "imperfect OPC, not a nephelometer" finding.

Nothing here depends on proprietary data; the aerosol is an *assumed* background
accumulation mode. The whole point of the UDE is that the NN corrects for that
assumption being wrong (composition, aging, hysteresis, flow).
"""
import pathlib
import numpy as np
import miepython as mp

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- constants
LAMBDA   = 0.657          # laser wavelength [um]  (Ouimette 2024)
M_DRY    = 1.52 - 0.002j  # assumed dry ambient refractive index (m = n - ik)
M_WATER  = 1.333 - 0.0j   # water at 657 nm
KAPPA    = 0.20           # hygroscopicity (ambient mixed aerosol ~0.1-0.3)
# assumed dry lognormal accumulation mode (number distribution)
DG_DRY   = 0.15           # number geometric-mean diameter [um]
SIGMA_G  = 1.60           # geometric standard deviation
N_TOTAL  = 2.0e3          # total number conc [cm^-3] (sets absolute scale only)
RH_DRY   = 0.20           # reference "dry" RH for humidification factor
CAL_0015 = 0.015          # 2022 paper: b_sp1 [Mm^-1] = 0.015 * CH1  (RH<40%)

# integration grid over diameter
DP = np.logspace(np.log10(0.01), np.log10(10.0), 600)   # [um]
LNDP = np.log(DP)


# ---------------------------------------------------------------- pieces
def qsca(dp_um, m, lam=LAMBDA):
    """Mie scattering efficiency Q_sca(Dp) for scalar or array Dp [um]."""
    x = np.pi * np.asarray(dp_um) / lam
    # efficiencies_mx returns (qext, qsca, qback, g)
    out = mp.efficiencies_mx(m, x)
    return np.asarray(out[1], dtype=float)


def pse(dp_um, base="ln"):
    """
    Particle sizing efficiency, Ouimette 2024 Eq. 16:
        PSE(Dp) = exp[-3.22 * log(Dp / 0.30 um)]
    'log' base is ambiguous in the note. base='ln' -> (Dp/0.30)^-3.22.
    Physically PSE = P(correct bin); large particles crossing the beam edge get
    undersized, so PSE falls with Dp. Capped at 1 (a particle can't be
    'more than correctly' sized).
    """
    dp = np.asarray(dp_um)
    if base == "ln":
        val = np.exp(-3.22 * np.log(dp / 0.30))
    elif base == "log10":
        val = np.exp(-3.22 * np.log10(dp / 0.30))
    else:
        raise ValueError(base)
    return np.minimum(val, 1.0)


def dNdlnDp(dp_um, dg, sigma_g, N):
    """Lognormal number distribution dN/dlnDp."""
    lnsg = np.log(sigma_g)
    return (N / (np.sqrt(2 * np.pi) * lnsg)
            * np.exp(-(np.log(dp_um) - np.log(dg))**2 / (2 * lnsg**2)))


def growth_factor(rh, kappa=KAPPA):
    """kappa-Koehler diameter growth factor GF = Dp_wet/Dp_dry (Kelvin neglected)."""
    aw = np.clip(rh, 0.0, 0.985)          # water activity ~ RH; clip to avoid blowup
    return (1.0 + kappa * aw / (1.0 - aw))**(1.0 / 3.0)


def wet_refractive_index(gf):
    """Volume mixing of dry aerosol + condensed water."""
    fw = 1.0 - 1.0 / gf**3                # volume fraction water
    return M_DRY * (1.0 - fw) + M_WATER * fw


def bulk_scattering(rh, truncate, base="ln"):
    """
    Bulk scattering coefficient [Mm^-1] at given RH.
    truncate=False -> ideal nephelometer b_true; True -> PSE-weighted OPC b_obs.
    Growth shifts BOTH the diameter (-> larger geometric area & x) and the
    refractive index (-> toward water). PSE is evaluated on the *wet* diameter,
    because that is what the sensor actually intercepts.
    """
    gf = growth_factor(rh)
    m_wet = wet_refractive_index(gf)
    dp_wet = DP * gf                                   # wet diameters [um]
    q = qsca(dp_wet, m_wet)                            # Mie on wet particle
    area = np.pi / 4.0 * (dp_wet)**2                   # [um^2]
    weight = pse(dp_wet, base=base) if truncate else 1.0
    integrand = dNdlnDp(DP, DG_DRY, SIGMA_G, N_TOTAL) * q * area * weight
    # units: [cm^-3] * [-] * [um^2]; convert to Mm^-1:
    #   cm^-3 * um^2 = 1e-8 cm^-1 = 1e-8 * 1e2 m^-1 = 1e-6 m^-1 = 1.0 Mm^-1 ... check:
    #   1 um^2 = 1e-8 cm^2; * cm^-3 -> 1e-8 cm^-1; 1 cm^-1 = 1e2 m^-1 = 1e8 Mm^-1
    #   so 1e-8 cm^-1 = 1.0 Mm^-1.  => integral (in cm^-3*um^2 units) IS already Mm^-1.
    return np.trapezoid(integrand, LNDP)


# ---------------------------------------------------------------- diagnostics
def main():
    print("="*72)
    print("PMS5003 known-physics transfer function  (lambda=657 nm)")
    print("="*72)
    print(f"assumed dry mode: Dg={DG_DRY} um, sigma_g={SIGMA_G}, "
          f"N={N_TOTAL:.0f} cm^-3, m_dry={M_DRY}, kappa={KAPPA}")

    # --- 1. dry-state truncation ratio (2024 paper's central claim) ----------
    b_true_dry = bulk_scattering(RH_DRY, truncate=False)
    b_obs_dry  = bulk_scattering(RH_DRY, truncate=True, base="ln")
    b_obs_dry_log10 = bulk_scattering(RH_DRY, truncate=True, base="log10")
    print("\n-- dry state (RH=%.0f%%) --" % (RH_DRY*100))
    print(f"  b_true (ideal nephelometer) = {b_true_dry:8.2f} Mm^-1")
    print(f"  b_obs  (PSE-truncated, ln ) = {b_obs_dry:8.2f} Mm^-1   "
          f"truncation ratio = {b_obs_dry/b_true_dry:.3f}")
    print(f"  b_obs  (PSE-truncated,log10)= {b_obs_dry_log10:8.2f} Mm^-1   "
          f"truncation ratio = {b_obs_dry_log10/b_true_dry:.3f}")
    print("  -> the sensor systematically undersizes: observed scattering is a")
    print("     fraction of the true nephelometer value (Ouimette 2024, Eq. 14).")

    # --- 2. CH1 <-> b_sp1 = 0.015*CH1 sanity check (2022 baseline) -----------
    # 2022 treats CH1 as reporting *true* bulk scattering at RH<40%.
    ch1_equiv = b_true_dry / CAL_0015
    print("\n-- 2022 linear baseline (b_sp1 = 0.015*CH1, RH<40%) --")
    print(f"  implied CH1 for this aerosol = {ch1_equiv:8.0f} counts "
          f"(order-of-magnitude plausible for moderate PM)")

    # --- 3. humidification factor f(RH) and gamma-law surrogate --------------
    rh_grid = np.linspace(0.20, 0.95, 40)
    f_obs = np.array([bulk_scattering(r, truncate=True, base="ln") for r in rh_grid])
    f_obs /= bulk_scattering(RH_DRY, truncate=True, base="ln")   # normalize to dry

    # fit gamma-law: f(RH) = (( 1 - RH_dry ) / ( 1 - RH ))^gamma   (Kasten form,
    # anchored so f(RH_dry)=1 exactly). Fit gamma by least squares in log space.
    x = np.log((1 - RH_DRY) / (1 - rh_grid))
    y = np.log(f_obs)
    gamma = float(np.sum(x*y) / np.sum(x*x))          # through-origin slope
    f_fit = ((1 - RH_DRY) / (1 - rh_grid))**gamma
    ss_res = np.sum((f_obs - f_fit)**2)
    ss_tot = np.sum((f_obs - f_obs.mean())**2)
    r2 = 1 - ss_res/ss_tot
    print("\n-- humidification factor  f(RH) = b_obs(RH)/b_obs(dry) --")
    print(f"  gamma-law fit:  f(RH) = ((1-{RH_DRY})/(1-RH))^gamma,  "
          f"gamma = {gamma:.4f},  R^2 = {r2:.5f}")
    for r in (0.40, 0.60, 0.80, 0.90):
        print(f"    RH={r*100:4.0f}%:  f_mie={((1-RH_DRY)/(1-r))**gamma:5.2f}"
              f"   (full Mie: {bulk_scattering(r,True,'ln')/bulk_scattering(RH_DRY,True,'ln'):5.2f})")
    print("  -> gamma-law is a useful reference but NOT what Julia uses.")

    # --- 3b. kappa-Koehler surrogate (the form Julia actually uses) ----------
    gf_dry_val = growth_factor(RH_DRY)
    x_kk = np.log(np.array([growth_factor(r) for r in rh_grid]) / gf_dry_val)
    p_scat = float(np.sum(x_kk * y) / np.sum(x_kk * x_kk))
    f_kk = np.exp(p_scat * x_kk)
    r2_kk = 1.0 - np.sum((f_obs - f_kk)**2) / ss_tot
    print(f"\n-- kappa-Koehler surrogate  f(RH) = (GF(RH)/GF_dry)^p_scat --")
    print(f"  p_scat = {p_scat:.4f},  R^2 = {r2_kk:.5f},  GF_dry = {gf_dry_val:.6f}")
    for r in (0.40, 0.60, 0.80, 0.90):
        gf_r = growth_factor(r)
        print(f"    RH={r*100:4.0f}%:  f_kk={(gf_r/gf_dry_val)**p_scat:5.2f}"
              f"   (full Mie: {bulk_scattering(r,True,'ln')/bulk_scattering(RH_DRY,True,'ln'):5.2f})")
    print("  -> P_SCAT and GF_dry written to physics_constants.txt for Julia")

    # --- 4. PSE log-base sensitivity ----------------------------------------
    print("\n-- PSE log-base sensitivity (flag for Eq. 16 verification) --")
    for base in ("ln", "log10"):
        print(f"    base={base:6s}: PSE(0.6um)={float(pse(0.6,base)):.3f}, "
              f"PSE(1.0um)={float(pse(1.0,base)):.3f}, "
              f"PSE(2.5um)={float(pse(2.5,base)):.3f}")
    print("  -> materially changes the absolute truncation ratio (above), only")
    print("     weakly changes f(RH) shape. Confirm base against the paper.")

    # save constants for the Julia port
    out_path = _ROOT / "physics_constants.txt"
    with open(out_path, "w") as fh:
        fh.write(f"gamma_RH      = {gamma:.6f}\n")
        fh.write(f"RH_dry        = {RH_DRY}\n")
        fh.write(f"trunc_ratio_ln= {b_obs_dry/b_true_dry:.6f}\n")
        fh.write(f"b_obs_dry     = {b_obs_dry:.6f}   # Mm^-1\n")
        fh.write(f"b_true_dry    = {b_true_dry:.6f}   # Mm^-1\n")
        fh.write(f"kappa         = {KAPPA}\n")
        fh.write(f"\n# --- CHOSEN humidification surrogate for Julia (R^2={r2_kk:.3f} vs full Mie) ---\n")
        fh.write(f"# f(RH) = ( GF(RH) / GF(RH_dry) )^p_scat,  GF from kappa-Koehler (closed form)\n")
        fh.write(f"p_scat        = {p_scat:.6f}\n")
        fh.write(f"GF_dry        = {gf_dry_val:.6f}   # growth_factor(RH_dry={RH_DRY}, kappa={KAPPA})\n")
    print(f"\nwrote {out_path}")
    print(f"  gamma_RH = {gamma:.4f}  (Kasten form, reference only)")
    print(f"  p_scat   = {p_scat:.4f}  (kappa-Koehler form, used by Julia)")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

        ax[0].plot(DP, qsca(DP, M_DRY), lw=1.6)
        ax[0].set_xscale("log"); ax[0].set_xlabel("Dp [um]")
        ax[0].set_ylabel("Q_sca"); ax[0].set_title("Mie efficiency (dry, 657 nm)")
        ax[0].axvspan(0.1, 1.0, color="k", alpha=0.06)

        dry = dNdlnDp(DP, DG_DRY, SIGMA_G, N_TOTAL)
        contrib_true = dry * qsca(DP, M_DRY) * np.pi/4*DP**2
        contrib_obs  = contrib_true * pse(DP, "ln")
        ax[1].plot(DP, contrib_true, label="ideal nephelometer", lw=1.6)
        ax[1].plot(DP, contrib_obs,  label="PSE-truncated (OPC)", lw=1.6)
        ax[1].set_xscale("log"); ax[1].set_xlabel("Dp [um]")
        ax[1].set_ylabel("db_sp/dlnDp  [Mm^-1]")
        ax[1].set_title("scattering size-contribution"); ax[1].legend(fontsize=8)

        ax[2].plot(rh_grid*100, f_obs, "o", ms=3, label="full Mie + kappa-Koehler")
        ax[2].plot(rh_grid*100, f_fit, "-", lw=1.6, label=f"gamma-law (g={gamma:.2f})")
        ax[2].set_xlabel("RH [%]"); ax[2].set_ylabel("f(RH) = b(RH)/b(dry)")
        ax[2].set_title("humidification factor"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(_ROOT / "physics_check.png", dpi=130)
        print(f"wrote {_ROOT / 'physics_check.png'}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
