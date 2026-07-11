"""
Layer-1 Mie emulator — data generation.

Generates ground-truth (x, n, k) -> (Q_sca, Q_ext, g) from miepython
and caches to .npz on disk.  Training and validation grids use different
x-nodes (stratified split) so ripple interpolation error is measured honestly.

Domain
------
x  : [0.03, 80]  log-spaced   — covers Rayleigh→resonance→geometric
n  : [1.33, 1.80]
k  : [0.00, 0.80]  (sulfate/water → black carbon)
"""

import pathlib
import numpy as np
import miepython as mp

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Grid sizes — generous for the overnight run
N_X_TOTAL  = 1200   # x nodes (split 80/20 train/val at different positions)
N_N        = 30     # n nodes
N_K        = 30     # k nodes
CACHE_PATH = ROOT / "python_modules" / "mie_emulator" / "mie_cache.npz"


def _compute_efficiencies(x_arr, n_arr, k_arr):
    """
    Returns arrays shaped (N,) for Q_sca, Q_ext, g
    given flat arrays x_arr, n_arr, k_arr (same length).
    """
    Q_sca = np.empty(len(x_arr))
    Q_ext = np.empty(len(x_arr))
    g_asy = np.empty(len(x_arr))

    # miepython.efficiencies_mx: (m, x) → (qext, qsca, qback, g)
    for i, (x, n, k) in enumerate(zip(x_arr, n_arr, k_arr)):
        m = complex(n, -k)            # convention: m = n - ik
        qe, qs, _, gi = mp.efficiencies_mx(m, float(x))
        Q_ext[i] = float(qe)
        Q_sca[i] = float(qs)
        g_asy[i]  = float(gi)

    return Q_sca, Q_ext, g_asy


def generate_or_load(force=False):
    """
    Returns dict with keys: x_tr, n_tr, k_tr, Qsca_tr, Qext_tr, g_tr
                            x_va, n_va, k_va, Qsca_va, Qext_va, g_va
    """
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached data from {CACHE_PATH}")
        d = np.load(CACHE_PATH)
        return dict(d)

    rng = np.random.default_rng(42)

    # --- build x grid with denser sampling in the ripple regime ---
    # Interleave: split the full log-range, then alternate train/val by
    # position so the two sets never share an x-node.
    x_all = np.logspace(np.log10(0.03), np.log10(80.0), N_X_TOTAL)
    idx   = np.arange(N_X_TOTAL)
    # even indices → train, odd → val  (interleaved so both cover ripple regime)
    x_tr  = x_all[idx % 2 == 0]
    x_va  = x_all[idx % 2 == 1]

    n_pts = np.linspace(1.33, 1.80, N_N)
    k_pts = np.linspace(0.00, 0.80, N_K)

    def make_full_grid(x_nodes):
        xx, nn, kk = np.meshgrid(x_nodes, n_pts, k_pts, indexing='ij')
        return xx.ravel(), nn.ravel(), kk.ravel()

    x_tr_flat, n_tr_flat, k_tr_flat = make_full_grid(x_tr)
    x_va_flat, n_va_flat, k_va_flat = make_full_grid(x_va)

    print(f"Train set: {len(x_tr_flat):,} pts   Val set: {len(x_va_flat):,} pts")

    print("Computing training efficiencies...")
    Qs_tr, Qe_tr, g_tr = _compute_efficiencies(x_tr_flat, n_tr_flat, k_tr_flat)

    print("Computing validation efficiencies...")
    Qs_va, Qe_va, g_va = _compute_efficiencies(x_va_flat, n_va_flat, k_va_flat)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        x_tr=x_tr_flat,   n_tr=n_tr_flat,   k_tr=k_tr_flat,
        Qsca_tr=Qs_tr,    Qext_tr=Qe_tr,    g_tr=g_tr,
        x_va=x_va_flat,   n_va=n_va_flat,   k_va=k_va_flat,
        Qsca_va=Qs_va,    Qext_va=Qe_va,    g_va=g_va,
    )
    print(f"Saved → {CACHE_PATH}")
    return dict(np.load(CACHE_PATH))


if __name__ == "__main__":
    d = generate_or_load(force=True)
    print("Qsca_tr range:", d["Qsca_tr"].min(), "–", d["Qsca_tr"].max())
    print("g_tr    range:", d["g_tr"].min(),    "–", d["g_tr"].max())
