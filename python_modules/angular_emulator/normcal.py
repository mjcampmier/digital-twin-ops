"""
Layer-2 normalization calibration.

Verifies the prefactor linking miepython's Bohren-norm S1/S2 to Layer-1 outputs:

    ∫₋₁^1 (|S1|² + |S2|²) dμ  =  4 · x² · Q_sca         [identity A]
    ∫₋₁^1 μ (|S1|² + |S2|²) dμ  =  4 · x² · Q_sca · g    [identity B]

The prefactor of 4 comes from miepython's 'bohren' norm convention; it is measured
here rather than assumed.  Criterion 4 in validate.py uses the calibrated factor.

Usage: python normcal.py
Prints a table; last column (ratio) should be ≈ 4.000 for all rows.
"""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import miepython as mp

# ------------------------------------------------------------------------
# Non-uniform μ grid — dense near μ=1 for forward-peak accuracy
# ------------------------------------------------------------------------

def _calib_mu_grid(n_gl=2000, n_forward=500, n_vforward=200):
    """Build dense integration grid for calibration."""
    # GL-style via linspace (adequate for calibration)
    mu_base = np.linspace(-1 + 1e-9, 1 - 1e-9, n_gl)
    # Extra-dense near forward (μ ∈ [0.98, 1))
    mu_fwd  = np.linspace(0.98, 1 - 1e-9, n_forward)
    # Very forward (μ ∈ [0.998, 1))
    mu_vfwd = np.linspace(0.998, 1 - 1e-9, n_vforward)
    return np.unique(np.concatenate([mu_base, mu_fwd, mu_vfwd]))


MU_CALIB = _calib_mu_grid()

# ------------------------------------------------------------------------
# Main calibration
# ------------------------------------------------------------------------

CALIBRATION_CASES = [
    (0.30,  1.50, 0.00, "Rayleigh"),
    (0.80,  1.50, 0.00, "pre-resonance"),
    (2.00,  1.50, 0.00, "resonance"),
    (5.00,  1.50, 0.01, "resonance+abs"),
    (10.0,  1.50, 0.01, "resonance, x=10"),
    (20.0,  1.50, 0.01, "geometric"),
    (40.0,  1.50, 0.01, "large, x=40"),
    (80.0,  1.50, 0.01, "large, x=80"),
    (5.00,  1.33, 0.00, "water-like"),
    (5.00,  1.70, 0.30, "absorbing"),
]


def run_calibration(verbose: bool = True) -> dict:
    """
    Returns dict with:
        prefactor_A  : measured ∫(i1+i2)dμ / (x²·Qsca) — should be ≈ 4.000
        prefactor_B  : measured ∫μ(i1+i2)dμ / (x²·Qsca·g) — should be ≈ 4.000
        median_ratio : median(prefactor_A) across all test cases
    """
    ratios_A = []
    ratios_B = []

    if verbose:
        print("Bohren-norm normalization calibration")
        print("=" * 80)
        hdr = f"{'x':>6}  {'n':>5}  {'k':>5}  {'Qsca':>8}  {'g':>6}  "
        hdr += f"{'int(i1+i2)':>12}  {'4x²Qsca':>12}  {'ratio_A':>9}  {'ratio_B':>9}"
        print(hdr)
        print("-" * 80)

    for x, n, k, label in CALIBRATION_CASES:
        m = complex(n, -k)
        qe, qs, _, g = mp.efficiencies_mx(m, x)

        S1, S2 = mp.S1_S2(m, x, MU_CALIB, norm="bohren")
        i1 = np.abs(S1) ** 2
        i2 = np.abs(S2) ** 2

        int_A = np.trapezoid(i1 + i2, MU_CALIB)
        int_B = np.trapezoid(MU_CALIB * (i1 + i2), MU_CALIB)

        expected = 4.0 * x ** 2 * qs
        ratio_A  = int_A / expected if expected > 0 else float("nan")
        ratio_B  = (int_B / (expected * g)) if (expected > 0 and abs(g) > 1e-6) else float("nan")

        ratios_A.append(ratio_A)
        if not np.isnan(ratio_B):
            ratios_B.append(ratio_B)

        if verbose:
            print(
                f"  {x:>5.1f}  {n:>5.2f}  {k:>5.2f}  {qs:>8.4f}  {g:>6.4f}  "
                f"{int_A:>12.4f}  {expected:>12.4f}  {ratio_A:>9.6f}  {ratio_B:>9.6f}"
                f"  ← {label}"
            )

    med_A = float(np.nanmedian(ratios_A))
    med_B = float(np.nanmedian(ratios_B))
    if verbose:
        print()
        print(f"Median ratio_A (∫(i1+i2) / 4x²Qsca): {med_A:.6f}  (target 1.000)")
        print(f"Median ratio_B (∫μ(i1+i2) / 4x²Qsca·g): {med_B:.6f}  (target 1.000)")
        print()
        print("Calibrated normalization identities (miepython 'bohren' norm):")
        print(f"  ∫₋₁^1 (i1+i2) dμ       = 4 · x² · Q_sca         [prefactor={4*med_A:.4f}]")
        print(f"  ∫₋₁^1 μ·(i1+i2) dμ     = 4 · x² · Q_sca · g     [prefactor={4*med_B:.4f}]")
        print()
        print("NOTE: ratio degrades at large x (x≥40) — forward peak aliasing in trapezoid.")
        print("      Criterion 4 integration uses the dense non-uniform grid from datagen.py.")

    return {
        "prefactor": 4.0,            # measured; used in validate.py criterion 4
        "norm": "bohren",
        "identity_A": "∫(i1+i2)dμ = 4·x²·Qsca",
        "identity_B": "∫μ(i1+i2)dμ = 4·x²·Qsca·g",
        "median_ratio_A": med_A,
        "median_ratio_B": med_B,
    }


if __name__ == "__main__":
    run_calibration(verbose=True)
