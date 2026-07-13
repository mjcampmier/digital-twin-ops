"""
Layer-2 angular Mie emulator — data generation.

Generates (x, n, k, μ) → (i₁=|S₁|², i₂=|S₂|²) from miepython and caches
to .npz on disk.

Domain (identical to Layer 1 for drop-in composability):
  x  : [0.03, 80]   (size parameter)
  n  : [1.33, 1.80] (real refractive index)
  k  : [0.00, 0.80] (imaginary RI — sulfate/water to absorbing soot)
  μ  : [-1, 1]      (cos θ), sampled non-uniformly with forward emphasis

Normalization (from normcal.py):
  ∫₋₁^1 (i₁ + i₂) dμ = 4 · x² · Q_sca   [miepython 'bohren' norm]

The training set uses N_PARTICLES_TR particles × N_ANGLES_TR angles per particle.
Validation set uses N_PARTICLES_VA particles × N_ANGLES_VA angles (denser, for
integration accuracy in criterion 4/5 checks).  Train/val particle sets are
drawn from different (non-overlapping) random seeds so they cover different
(x, n, k) positions — an honest interpolation test.
"""

import pathlib
import time
import numpy as np
import miepython as mp

ROOT       = pathlib.Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT / "python_modules" / "angular_emulator" / "angular_cache.npz"

# Particles per set — 5k×200 = 1M pts (same scale as Layer 1, dense angular coverage)
# 200 angles/particle resolves Δμ ≈ 0.01, Nyquist-sufficient for x≤100
N_PARTICLES_TR = 5_000
N_PARTICLES_VA = 5_000

# Angles per particle — both sets use 200 angles for consistent angular density
N_ANGLES_TR = 200
N_ANGLES_VA = 200

# Domain
X_MIN, X_MAX  = 0.03, 80.0
N_MIN, N_MAX  = 1.33, 1.80
K_MIN, K_MAX  = 0.00, 0.80

# Floor for log(intensity) — avoids -inf at true Mie nulls
I_FLOOR = 1e-15


# ---------------------------------------------------------------------------
# Angle grids
# ---------------------------------------------------------------------------

def _make_angle_grid(n_total: int, forward_fraction: float = 0.45) -> np.ndarray:
    """
    Non-uniform μ grid with heavy forward (μ→1) emphasis.

    Composition:
      - ~50%  uniform in θ ∈ [0°, 180°]  →  density ∝ 1/sin(θ), naturally denser
              near μ = ±1
      - ~25%  uniform in θ ∈ [0°, 20°]   →  forward region
      - ~15%  uniform in μ ∈ [0.98, 1)   →  near-forward
      - ~10%  uniform in μ ∈ [0.99, 1)   →  very near-forward
    Blend, deduplicate, sort, clip to avoid exact poles.
    """
    n_uniform_theta  = int(0.50 * n_total)
    n_fwd_theta      = int(0.25 * n_total)
    n_near_fwd       = int(0.15 * n_total)
    n_vfwd           = n_total - n_uniform_theta - n_fwd_theta - n_near_fwd

    theta_u  = np.linspace(0.0, np.pi, n_uniform_theta)
    theta_fw = np.linspace(0.0, np.deg2rad(20.0), n_fwd_theta)
    mu_near  = np.linspace(0.98, 1 - 1e-9, n_near_fwd)
    mu_vfwd  = np.linspace(0.999, 1 - 1e-9, n_vfwd)

    mu_all = np.concatenate([
        np.cos(theta_u),
        np.cos(theta_fw),
        mu_near,
        mu_vfwd,
    ])
    mu_all = np.clip(mu_all, -1 + 1e-9, 1 - 1e-9)
    mu_all = np.unique(mu_all)
    # Thin or pad to exactly n_total
    if len(mu_all) > n_total:
        idx    = np.round(np.linspace(0, len(mu_all) - 1, n_total)).astype(int)
        mu_all = mu_all[idx]
    return mu_all


MU_TRAIN = _make_angle_grid(N_ANGLES_TR)


def _make_val_angle_grid(n_total: int = 200) -> np.ndarray:
    """Denser validation grid for integration accuracy."""
    # Use Gauss-Legendre nodes via numpy (accurate quadrature)
    # Supplement with extra forward nodes
    n_gl    = int(0.55 * n_total)
    n_fwd   = int(0.25 * n_total)
    n_vfwd  = int(0.10 * n_total)
    n_rest  = n_total - n_gl - n_fwd - n_vfwd

    from numpy.polynomial.legendre import leggauss
    mu_gl, _ = leggauss(n_gl)        # in (-1, 1), already no poles

    mu_fwd  = np.linspace(0.97, 1 - 1e-9, n_fwd)
    mu_vfwd = np.linspace(0.998, 1 - 1e-9, n_vfwd)
    mu_back = np.linspace(-1 + 1e-9, -0.90, n_rest)

    mu_all = np.clip(
        np.unique(np.concatenate([mu_gl, mu_fwd, mu_vfwd, mu_back])),
        -1 + 1e-9, 1 - 1e-9,
    )
    if len(mu_all) > n_total:
        idx    = np.round(np.linspace(0, len(mu_all) - 1, n_total)).astype(int)
        mu_all = mu_all[idx]
    return mu_all


MU_VAL = _make_val_angle_grid(N_ANGLES_VA)


# ---------------------------------------------------------------------------
# Particle sampling
# ---------------------------------------------------------------------------

def _sample_particles(n: int, rng: np.random.Generator) -> tuple:
    """
    Sample (x, n_ri, k_ri) in the Layer-1 domain.
    x is log-uniform; n and k are uniform.
    Returns arrays of shape (n,).
    """
    logx = rng.uniform(np.log(X_MIN), np.log(X_MAX), n)
    x    = np.exp(logx)
    n_ri = rng.uniform(N_MIN, N_MAX, n)
    k_ri = rng.uniform(K_MIN, K_MAX, n)
    return x, n_ri, k_ri


# ---------------------------------------------------------------------------
# Per-particle intensity computation
# ---------------------------------------------------------------------------

def _particle_intensities(
    x: float,
    n_ri: float,
    k_ri: float,
    mu_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute i₁=|S₁|², i₂=|S₂|² at each angle in mu_arr for a single particle.
    Returns (i1, i2) each of shape (len(mu_arr),).
    """
    m   = complex(n_ri, -k_ri)
    S1, S2 = mp.S1_S2(m, float(x), mu_arr, norm="bohren")
    i1  = np.abs(S1) ** 2
    i2  = np.abs(S2) ** 2
    return i1, i2


# ---------------------------------------------------------------------------
# Main data generation
# ---------------------------------------------------------------------------

def _generate_set(
    n_particles: int,
    mu_grid: np.ndarray,
    rng: np.random.Generator,
    label: str,
) -> dict:
    """Generate a flat (N_particles × N_angles,) dataset."""
    x_p, n_p, k_p = _sample_particles(n_particles, rng)
    n_ang     = len(mu_grid)
    n_total   = n_particles * n_ang

    x_flat  = np.empty(n_total, dtype=np.float32)
    n_flat  = np.empty(n_total, dtype=np.float32)
    k_flat  = np.empty(n_total, dtype=np.float32)
    mu_flat = np.empty(n_total, dtype=np.float32)
    i1_flat = np.empty(n_total, dtype=np.float32)
    i2_flat = np.empty(n_total, dtype=np.float32)

    t0 = time.time()
    for ip in range(n_particles):
        if ip % 10_000 == 0 and ip > 0:
            elapsed = time.time() - t0
            rate    = ip / elapsed
            eta     = (n_particles - ip) / rate if rate > 0 else 0
            print(f"  {label}: {ip}/{n_particles}  ({rate:.0f} part/s, ETA {eta:.0f}s)")

        i1, i2 = _particle_intensities(x_p[ip], n_p[ip], k_p[ip], mu_grid)
        sl = slice(ip * n_ang, (ip + 1) * n_ang)
        x_flat[sl]  = x_p[ip]
        n_flat[sl]  = n_p[ip]
        k_flat[sl]  = k_p[ip]
        mu_flat[sl] = mu_grid.astype(np.float32)
        i1_flat[sl] = i1.astype(np.float32)
        i2_flat[sl] = i2.astype(np.float32)

    elapsed = time.time() - t0
    print(f"  {label}: done — {n_particles} particles × {n_ang} angles = "
          f"{n_total:,} pts in {elapsed:.1f}s")
    return {"x": x_flat, "n": n_flat, "k": k_flat, "mu": mu_flat,
            "i1": i1_flat, "i2": i2_flat}


def generate_or_load(force: bool = False) -> dict:
    """
    Returns dict with keys:
        x_tr, n_tr, k_tr, mu_tr, i1_tr, i2_tr   — training set (flat)
        x_va, n_va, k_va, mu_va, i1_va, i2_va   — validation set (flat)
        mu_train_grid, mu_val_grid                — the angle grids (for reference)
    """
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached data from {CACHE_PATH}")
        d = dict(np.load(CACHE_PATH))
        return d

    print("Generating Layer-2 angular intensity data...")
    print(f"  Train: {N_PARTICLES_TR:,} particles × {N_ANGLES_TR} angles = "
          f"{N_PARTICLES_TR * N_ANGLES_TR:,} pts")
    print(f"  Val  : {N_PARTICLES_VA:,} particles × {N_ANGLES_VA} angles = "
          f"{N_PARTICLES_VA * N_ANGLES_VA:,} pts")
    print(f"  Angle grid (train): forward pts (μ>0.98) = "
          f"{(MU_TRAIN > 0.98).sum()}/{len(MU_TRAIN)}")

    rng_tr = np.random.default_rng(42)
    rng_va = np.random.default_rng(99)     # different seed → different (x,n,k) positions

    tr = _generate_set(N_PARTICLES_TR, MU_TRAIN, rng_tr, "train")
    va = _generate_set(N_PARTICLES_VA, MU_VAL,   rng_va, "val  ")

    out = {
        "x_tr":  tr["x"],  "n_tr":  tr["n"],  "k_tr":  tr["k"],
        "mu_tr": tr["mu"], "i1_tr": tr["i1"], "i2_tr": tr["i2"],
        "x_va":  va["x"],  "n_va":  va["n"],  "k_va":  va["k"],
        "mu_va": va["mu"], "i1_va": va["i1"], "i2_va": va["i2"],
        "mu_train_grid": MU_TRAIN.astype(np.float32),
        "mu_val_grid":   MU_VAL.astype(np.float32),
        "i_floor": np.float32(I_FLOOR),
    }
    np.savez_compressed(CACHE_PATH, **out)
    print(f"Saved to {CACHE_PATH}")
    return out


if __name__ == "__main__":
    generate_or_load(force=True)
