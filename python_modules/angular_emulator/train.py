"""
Layer-2 angular Mie emulator — training loop.

Trains AngularEmulator on log(i₁) + log(i₂) MSE loss (equal weight).
Saves best checkpoint (by val median relative error on i₁) to
angular_emulator_best.pt; also saves periodic snapshots every --save_every epochs.

Usage
-----
    python train.py [--epochs 400] [--batch 65536] [--hidden 512] [--n_hidden 8]
                    [--n_fx 256] [--scale_x 7.0] [--n_fmu 128] [--scale_mu 30.0]
                    [--lr 3e-4] [--device cpu]
"""

import argparse
import pathlib
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datagen import generate_or_load, I_FLOOR
from model   import AngularEmulator

ROOT  = pathlib.Path(__file__).resolve().parent
CKPT  = ROOT / "angular_emulator_best.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr.astype(np.float32), device=device)


def rel_err(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Element-wise relative error |pred - true| / true."""
    return (pred - true).abs() / (true + 1e-15)


def median_rel_err(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(rel_err(pred, true).median())


def log_loss(log_pred: torch.Tensor, log_true: torch.Tensor) -> torch.Tensor:
    """MSE in log space — equivalent to minimizing (log relative error)²."""
    return (log_pred - log_true).pow(2).mean()


# ---------------------------------------------------------------------------
# Validation (stratified by x and θ regime)
# ---------------------------------------------------------------------------

def validate(model: AngularEmulator, tensors: dict, device: str, batch: int = 131072) -> dict:
    model.eval()
    x, n, k, mu = tensors["x_va"], tensors["n_va"], tensors["k_va"], tensors["mu_va"]
    i1_true, i2_true = tensors["i1_va"], tensors["i2_va"]

    n_va = len(x)
    i1_pred_all = torch.empty(n_va, device=device)
    i2_pred_all = torch.empty(n_va, device=device)

    with torch.no_grad():
        for start in range(0, n_va, batch):
            sl = slice(start, start + batch)
            i1p, i2p = model(x[sl], n[sl], k[sl], mu[sl])
            i1_pred_all[sl] = i1p
            i2_pred_all[sl] = i2p

    # Global
    re_i1 = rel_err(i1_pred_all, i1_true)
    re_i2 = rel_err(i2_pred_all, i2_true)
    med_i1 = float(re_i1.median()) * 100
    p99_i1 = float(re_i1.kthvalue(int(0.99 * n_va))[0]) * 100
    med_i2 = float(re_i2.median()) * 100
    p99_i2 = float(re_i2.kthvalue(int(0.99 * n_va))[0]) * 100

    # Stratified by μ regime (forward / side / back)
    logx   = torch.log(x)
    mu_np  = mu.cpu().numpy()
    re1_np = re_i1.cpu().numpy()
    strats = {
        "forward (μ>0.94, θ<20°)": mu_np > 0.94,
        "side (μ∈[-0.64,0.64], 50°–130°)": (mu_np > -0.64) & (mu_np < 0.64),
        "back (μ<-0.77, θ>140°)": mu_np < -0.77,
    }
    strat_meds = {}
    for label, mask in strats.items():
        if mask.sum() > 0:
            strat_meds[label] = float(np.median(re1_np[mask])) * 100

    model.train()
    return {
        "med_i1": med_i1, "p99_i1": p99_i1,
        "med_i2": med_i2, "p99_i2": p99_i2,
        "strat": strat_meds,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = args.device

    print("Loading / generating data...")
    data = generate_or_load()

    def t(key): return to_tensor(data[key], device)

    # Build training DataLoader
    x_tr   = t("x_tr");  n_tr = t("n_tr");   k_tr = t("k_tr")
    mu_tr  = t("mu_tr"); i1_tr = t("i1_tr"); i2_tr = t("i2_tr")
    # Floor intensities before log
    i1_tr  = torch.clamp(i1_tr, min=I_FLOOR)
    i2_tr  = torch.clamp(i2_tr, min=I_FLOOR)
    log_i1_tr = torch.log(i1_tr)
    log_i2_tr = torch.log(i2_tr)

    tensors_va = {
        "x_va": t("x_va"), "n_va": t("n_va"), "k_va": t("k_va"),
        "mu_va": t("mu_va"),
        "i1_va": torch.clamp(t("i1_va"), min=I_FLOOR),
        "i2_va": torch.clamp(t("i2_va"), min=I_FLOOR),
    }

    ds     = TensorDataset(x_tr, n_tr, k_tr, mu_tr, log_i1_tr, log_i2_tr)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    # Build model
    model = AngularEmulator(
        hidden_dim       = args.hidden,
        n_hidden         = args.n_hidden,
        n_fourier_x      = args.n_fx,
        fourier_scale_x  = args.scale_x,
        n_legendre       = args.n_legendre,
        n_fourier_mu     = args.n_fmu,
        fourier_scale_mu = args.scale_mu,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: hidden={args.hidden}×{args.n_hidden}  "
          f"n_fx={args.n_fx}/scale={args.scale_x}  "
          f"n_legendre={args.n_legendre}  "
          f"n_fmu={args.n_fmu}/scale={args.scale_mu}  "
          f"params={param_count:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_med = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for xb, nb, kb, mub, l1b, l2b in loader:
            opt.zero_grad(set_to_none=True)
            lp1, lp2 = model.forward_log(xb, nb, kb, mub)
            loss = 0.5 * log_loss(lp1, l1b) + 0.5 * log_loss(lp2, l2b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(loader)

        if epoch % 10 == 0 or epoch == 1:
            v = validate(model, tensors_va, device)
            print(
                f"ep {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}  "
                f"val i1: med={v['med_i1']:.3f}% p99={v['p99_i1']:.2f}%  "
                f"i2: med={v['med_i2']:.3f}%  "
                f"| fwd={v['strat'].get('forward (μ>0.94, θ<20°)', float('nan')):.2f}%  "
                f"side={v['strat'].get('side (μ∈[-0.64,0.64], 50°–130°)', float('nan')):.2f}%  "
                f"back={v['strat'].get('back (μ<-0.77, θ>140°)', float('nan')):.2f}%  "
                f"  t={time.time()-t0:.1f}s"
            )
            if v["med_i1"] < best_med:
                best_med   = v["med_i1"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model_state": best_state,
                        "epoch": epoch,
                        "val_med_i1": best_med,
                        "args": vars(args),
                    },
                    CKPT,
                )
                print(f"  → saved best (med i1 {best_med:.3f}%)")

        if args.save_every > 0 and epoch % args.save_every == 0:
            snap = ROOT / f"angular_emulator_ep{epoch:04d}.pt"
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "args": vars(args)}, snap)

    print(f"\nTraining complete. Best val med i1 rel-err: {best_med:.3f}%")
    print(f"Checkpoint: {CKPT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=400)
    p.add_argument("--batch",     type=int,   default=65536)
    p.add_argument("--hidden",    type=int,   default=512)
    p.add_argument("--n_hidden",  type=int,   default=8)
    p.add_argument("--n_fx",        type=int,   default=256)
    p.add_argument("--scale_x",    type=float, default=7.0)
    p.add_argument("--n_legendre", type=int,   default=100)
    p.add_argument("--n_fmu",      type=int,   default=0)
    p.add_argument("--scale_mu",   type=float, default=30.0)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--save_every", type=int,  default=50)
    p.add_argument("--device",    type=str,   default="cpu")
    args = p.parse_args()

    train(args)
