"""
Generate L1 + L2 parity-gate test battery.

Saves two NPZ files for the Julia bridge parity check (§1):
  test_battery_L1.npz  —  (x,n,k) → (Q_sca, Q_ext, g)   via Python emulator
  test_battery_L2.npz  —  (x,n,k,μ) → (i1, i2)          via Python emulator

The Julia bridge must match these outputs to ≤1e-5 max-absolute-error.

Run:
    python python_modules/test_battery/generate_battery.py
"""

import pathlib, sys
import numpy as np

ROOT   = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python_modules"))

import torch
from mie_emulator.model import MieEmulator

# ── Load L1 Python emulator ──────────────────────────────────────────────────
L1_CKPT = ROOT / "python_modules" / "mie_emulator" / "mie_emulator_best.pt"
ck1     = torch.load(str(L1_CKPT), map_location="cpu", weights_only=False)
args1   = ck1["args"]
mie_emu = MieEmulator(
    hidden_dim     = args1.get("hidden", 512),
    n_layers       = args1.get("layers", 8),
    n_fourier      = args1.get("fourier", 256),
    fourier_scale  = args1.get("fourier_scale", 7.0),
    include_logx   = True,
    normalize_logx = True,
)
mie_emu.load_state_dict(ck1["model_state"])
mie_emu.eval()
print(f"L1 loaded: epoch={ck1.get('epoch','?')}  val_med={ck1.get('val_med', float('nan')):.3f}%")

# ── Load L2 Python emulator (NPZ) ────────────────────────────────────────────
from angular_emulator.validate import NpzAngularEmulator
L2_NPZ = ROOT / "python_modules" / "angular_emulator" / "angular_emulator_frozen.npz"
ang_emu = NpzAngularEmulator(str(L2_NPZ))
ang_emu.eval()

# ── Random test inputs (fixed seed for reproducibility) ─────────────────────
rng  = np.random.default_rng(20260101)
N    = 500

# L1 inputs: x ∈ [0.05, 80], n ∈ [1.33, 1.80], k ∈ [0.00, 0.80]
x1   = np.exp(rng.uniform(np.log(0.05), np.log(80.0), N)).astype(np.float32)
n1   = rng.uniform(1.33, 1.80, N).astype(np.float32)
k1   = rng.uniform(0.00, 0.80, N).astype(np.float32)

with torch.no_grad():
    xt = torch.tensor(x1); nt = torch.tensor(n1); kt = torch.tensor(k1)
    Q_sca_t, Q_ext_t, g_asy_t = mie_emu(xt, nt, kt)
    Q_sca = Q_sca_t.numpy().astype(np.float64)
    Q_ext = Q_ext_t.numpy().astype(np.float64)
    g_asy = g_asy_t.numpy().astype(np.float64)

out_dir = ROOT / "python_modules" / "test_battery"
out_dir.mkdir(exist_ok=True)

np.savez(out_dir / "test_battery_L1.npz",
         x=x1, n=n1, k=k1, Q_sca=Q_sca, Q_ext=Q_ext, g=g_asy,
         N=np.array([N]))
print(f"L1 battery: {N} points → {out_dir/'test_battery_L1.npz'}")
print(f"  Q_sca range: [{Q_sca.min():.4f}, {Q_sca.max():.4f}]")
print(f"  Q_ext range: [{Q_ext.min():.4f}, {Q_ext.max():.4f}]")
print(f"  g     range: [{g_asy.min():.4f}, {g_asy.max():.4f}]")

# L2 inputs: x ∈ [0.05, 80], n, k same range, μ ∈ [-1, 1]
x2   = np.exp(rng.uniform(np.log(0.05), np.log(80.0), N)).astype(np.float32)
n2   = rng.uniform(1.33, 1.80, N).astype(np.float32)
k2   = rng.uniform(0.00, 0.80, N).astype(np.float32)
mu2  = rng.uniform(-0.9999, 0.9999, N).astype(np.float32)

with torch.no_grad():
    i1_out, i2_out = ang_emu(
        torch.tensor(x2), torch.tensor(n2),
        torch.tensor(k2), torch.tensor(mu2),
    )
    i1 = i1_out.numpy().astype(np.float64)
    i2 = i2_out.numpy().astype(np.float64)

np.savez(out_dir / "test_battery_L2.npz",
         x=x2, n=n2, k=k2, mu=mu2, i1=i1, i2=i2, N=np.array([N]))
print(f"\nL2 battery: {N} points → {out_dir/'test_battery_L2.npz'}")
print(f"  i1 range: [{i1.min():.4e}, {i1.max():.4e}]")
print(f"  i2 range: [{i2.min():.4e}, {i2.max():.4e}]")

# ── Print spec sheet ──────────────────────────────────────────────────────────
print("""
===========================================================
FEATURE ENCODING SPEC (for Julia bridge, §1)
===========================================================

L1 — Mie efficiency emulator (mie_emulator_frozen.npz)
  Inputs:  x (size param), n (real RI), k (imag RI)
  Features (513-dim):
    [0]      lx_norm = (log(x) − 0.437) / 3.945
    [1:256]  cos(Bⱼ·log(x))   j=1..256  (from fourier/B)
    [257:512] sin(Bⱼ·log(x))
    [513]    n   (raw, not normalised)
    [514]    k   (raw, not normalised)
  Feature total: 1 + 2×256 + 2 = 515
  MLP: Dense(515→512,SiLU)×8 → Dense(512→3,linear)
  Output head: [exp(out[0]), out[1], out[2]] = (Q_sca, Q_ext, g)

L2 — Angular emulator (angular_emulator_frozen.npz)
  Inputs:  x, n, k, μ=cosθ
  Constants: LOGX_CENTER=0.4365  LOGX_HALF=3.9383
             N_CENTER=1.565       N_HALF=0.235
             K_CENTER=0.40        K_HALF=0.40
  Features (616-dim):
    [0]        lx_norm = (log(x)−0.4365)/3.9383
    [1:256]    cos(Bⱼ·log(x))   from fourier/B  (n_fourier=256)
    [257:512]  sin(Bⱼ·log(x))
    [513:613]  P₀(μ)…P₁₀₀(μ)  (Legendre, three-term recurrence, n_legendre=100+1=101)
    [614]      (n−1.565)/0.235
    [615]      (k−0.40)/0.40
  Feature total: 1 + 2×256 + 101 + 2 = 616
  MLP: Dense(616→512,SiLU)×8 → Dense(512→2,linear)
  Output: [exp(out[0]), exp(out[1])] = (i1, i2)
  Note: Julia convention is column-major — features shape (616,N);
        Dense layer: h ← W·h + b where W is (out,in).
===========================================================
""")
