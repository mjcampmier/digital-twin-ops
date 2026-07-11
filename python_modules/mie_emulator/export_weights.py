"""
Export frozen Mie emulator weights to a language-agnostic format (npz).

The npz contains:
  arch/          — architecture hyperparameters (scalar arrays)
  fourier/B      — Fourier embedding frequencies (n_freq,)
  layers/i/W     — weight matrix for layer i
  layers/i/b     — bias vector for layer i

The forward function is:
  1. logx = log(x)
  2. if n_fourier > 0:
       feat = [cos(B * logx), sin(B * logx), n, k]   (2*n_freq + 2)
     else:
       feat = [logx, n, k]
  3. For each hidden layer i:  feat = silu(W_i @ feat + b_i)
  4. Final layer:  out = W_last @ feat + b_last
  5. Q_sca = exp(out[0]),  Q_ext = out[1],  g = out[2]

SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))  — trivially portable to Julia.
"""

import pathlib
import numpy as np
import torch
from model import MieEmulator

ROOT = pathlib.Path(__file__).resolve().parent
CKPT = ROOT / "mie_emulator_best.pt"
OUT  = ROOT / "mie_emulator_frozen.npz"


def export(ckpt_path=CKPT, out_path=OUT):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    args  = ckpt["args"]
    model = MieEmulator(
        hidden_dim=args["hidden"],
        n_layers=args["layers"],
        n_fourier=args["fourier"],
        fourier_scale=args["fourier_scale"],
        include_logx=args.get("include_logx", False),
        normalize_logx=args.get("normalize_logx", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    arrays = {}
    # architecture
    arrays["arch/hidden_dim"]    = np.array(args["hidden"],        dtype=np.int32)
    arrays["arch/n_layers"]      = np.array(args["layers"],        dtype=np.int32)
    arrays["arch/n_fourier"]     = np.array(args["fourier"],       dtype=np.int32)
    arrays["arch/fourier_scale"] = np.array(args["fourier_scale"], dtype=np.float32)
    arrays["arch/val_med_err_pct"] = np.array(ckpt["val_med_err_pct"], dtype=np.float32)
    arrays["arch/epoch"]         = np.array(ckpt["epoch"],         dtype=np.int32)
    arrays["arch/include_logx"]   = np.array(int(args.get("include_logx", False)),   dtype=np.int32)
    arrays["arch/normalize_logx"] = np.array(int(args.get("normalize_logx", False)), dtype=np.int32)

    # Fourier embedding
    if args["fourier"] > 0:
        arrays["fourier/B"] = model.fourier.B.numpy().astype(np.float32)

    # MLP layers
    for i, mod in enumerate(model.net):
        if isinstance(mod, torch.nn.Linear):
            arrays[f"layers/{i}/W"] = mod.weight.detach().numpy().astype(np.float32)
            arrays[f"layers/{i}/b"] = mod.bias.detach().numpy().astype(np.float32)

    np.savez(out_path, **arrays)
    print(f"Exported → {out_path}")
    print(f"  Epoch {ckpt['epoch']}, val median Q_sca rel-err: {ckpt['val_med_err_pct']:.3f}%")
    print(f"  Layers: {sorted(k for k in arrays if k.startswith('layers/'))}")

    # Verify round-trip: compare numpy forward vs torch forward on a few points
    x_np = np.array([0.1, 1.0, 10.0, 50.0], dtype=np.float32)
    n_np = np.array([1.4, 1.5, 1.6, 1.7],   dtype=np.float32)
    k_np = np.array([0.0, 0.01, 0.1, 0.5],  dtype=np.float32)

    with torch.no_grad():
        Qs_t, _, _ = model(torch.tensor(x_np), torch.tensor(n_np), torch.tensor(k_np))
    Qs_t = Qs_t.numpy()

    Qs_np = _numpy_forward(arrays, x_np, n_np, k_np, args)
    max_re = np.max(np.abs(Qs_np - Qs_t) / (np.abs(Qs_t) + 1e-12))
    print(f"  Round-trip numerical check (max rel-err): {max_re:.2e}  (should be ~1e-7)")
    return arrays


def _numpy_forward(arrays, x_np, n_np, k_np, args):
    """Pure-numpy reference forward pass for round-trip verification."""
    n_fourier = int(args["fourier"])
    logx = np.log(x_np)

    LOGX_CENTER    = 0.437
    LOGX_HALFRANGE = 3.945
    include_logx   = bool(int(arrays.get("arch/include_logx",   np.array(0))))
    normalize_logx = bool(int(arrays.get("arch/normalize_logx", np.array(0))))
    if n_fourier > 0:
        B    = arrays["fourier/B"]          # (n_freq,)
        proj = logx[:, None] * B[None, :]  # (N, n_freq)
        if include_logx:
            lx_in = (logx - LOGX_CENTER) / LOGX_HALFRANGE if normalize_logx else logx
            feat = np.concatenate([lx_in[:, None], np.cos(proj), np.sin(proj),
                                   n_np[:, None], k_np[:, None]], axis=1)
        else:
            feat = np.concatenate([np.cos(proj), np.sin(proj),
                                   n_np[:, None], k_np[:, None]], axis=1)
    else:
        feat = np.stack([logx, n_np, k_np], axis=1)

    # walk through linear layers; SiLU between hidden layers
    hidden_count = 0
    layer_keys = sorted(
        [k for k in arrays if k.startswith("layers/") and k.endswith("/W")],
        key=lambda s: int(s.split("/")[1])
    )
    total_linear = len(layer_keys)

    for idx, wkey in enumerate(layer_keys):
        i = wkey.split("/")[1]
        W = arrays[f"layers/{i}/W"]
        b = arrays[f"layers/{i}/b"]
        feat = feat @ W.T + b
        if idx < total_linear - 1:  # SiLU on all but last
            feat = feat * (1.0 / (1.0 + np.exp(-feat)))

    # output head: [log(Qsca), Qext, g]
    Qsca = np.exp(feat[:, 0])
    return Qsca


if __name__ == "__main__":
    export()
