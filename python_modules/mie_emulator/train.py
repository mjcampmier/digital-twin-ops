"""
Layer-1 Mie emulator — training loop.

Trains on asinh(Q_sca) + Q_ext + g simultaneously.
Saves best checkpoint (by val Q_sca rel-err) to mie_emulator_best.pt.

Usage
-----
    python train.py [--epochs 300] [--batch 65536] [--hidden 256] [--layers 6]
                    [--fourier 64] [--lr 3e-4] [--device cpu]
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

from datagen import generate_or_load
from model import MieEmulator

ROOT   = pathlib.Path(__file__).resolve().parent
CKPT   = ROOT / "mie_emulator_best.pt"


def to_tensors(data: dict, device: str):
    def t(key):
        return torch.tensor(data[key].astype(np.float32), device=device)

    Qsca_tr = t("Qsca_tr")
    return {
        "x_tr":        t("x_tr"),
        "n_tr":        t("n_tr"),
        "k_tr":        t("k_tr"),
        "Qsca_tr":     Qsca_tr,
        "Qext_tr":     t("Qext_tr"),
        "g_tr":        t("g_tr"),
        "logQsca_tr":  torch.log(Qsca_tr.clamp(min=1e-30)),   # log target — even Rayleigh in range
        "x_va":        t("x_va"),
        "n_va":        t("n_va"),
        "k_va":        t("k_va"),
        "Qsca_va":     t("Qsca_va"),
        "Qext_va":     t("Qext_va"),
        "g_va":        t("g_va"),
    }


def rel_err(pred, true, eps=1e-30):
    return torch.abs(pred - true) / (torch.abs(true) + eps)


def train(args):
    device = args.device
    data   = generate_or_load()
    T      = to_tensors(data, device)

    model = MieEmulator(
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_fourier=args.fourier,
        fourier_scale=args.fourier_scale,
        include_logx=args.include_logx,
        normalize_logx=args.normalize_logx,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # dataset: (x, n, k, log_Qsca, Qext, g)
    ds   = TensorDataset(T["x_tr"], T["n_tr"], T["k_tr"],
                         T["logQsca_tr"], T["Qext_tr"], T["g_tr"])
    dl   = DataLoader(ds, batch_size=args.batch, shuffle=True, pin_memory=False)

    best_val_err = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, nb, kb, lQb, Qeb, gb in dl:
            Qs_pred, Qe_pred, g_pred = model(xb, nb, kb)
            # Q_sca: net outputs log(Q_sca) directly — MSE in log space = relative error loss
            log_Qs_pred = torch.log(Qs_pred.clamp(min=1e-30))
            l_sca = ((log_Qs_pred - lQb) ** 2).mean()
            l_ext = ((Qe_pred - Qeb) ** 2).mean()
            l_g   = ((g_pred - gb) ** 2).mean()
            loss  = l_sca + 0.5 * l_ext + 0.5 * l_g
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                Qs_v, _, _ = model(T["x_va"], T["n_va"], T["k_va"])
                re = rel_err(Qs_v, T["Qsca_va"])
                med = re.median().item() * 100
                p99 = re.quantile(0.99).item() * 100

                # stratified by x-regime
                x_va = T["x_va"]
                reg  = [(x_va < 0.5, "Rayleigh"),
                        ((x_va >= 0.5) & (x_va <= 30.0), "resonance"),
                        (x_va > 30.0, "geometric")]
                strat = []
                for mask, name in reg:
                    m_re = re[mask].median().item() * 100 if mask.any() else float("nan")
                    strat.append(f"{name}={m_re:.2f}%")

            if med < best_val_err:
                best_val_err = med
                torch.save({"model_state": model.state_dict(),
                            "args": vars(args),
                            "epoch": epoch,
                            "val_med_err_pct": med,
                            "val_p99_err_pct": p99}, CKPT)

            elapsed = time.time() - t0
            lr_now  = sched.get_last_lr()[0]
            print(f"[{epoch:4d}/{args.epochs}] loss={epoch_loss/len(dl):.4f}  "
                  f"val Q_sca med={med:.2f}% p99={p99:.2f}%  "
                  f"[{' | '.join(strat)}]  "
                  f"lr={lr_now:.2e}  t={elapsed:.0f}s  best={best_val_err:.2f}%")

    print(f"\nBest val med rel-err: {best_val_err:.3f}%")
    print(f"Checkpoint: {CKPT}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=400)
    p.add_argument("--batch",        type=int,   default=65536)
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--layers",       type=int,   default=6)
    p.add_argument("--fourier",      type=int,   default=64)
    p.add_argument("--fourier-scale",type=float, default=3.0, dest="fourier_scale")
    p.add_argument("--include-logx",    action="store_true", default=False, dest="include_logx")
    p.add_argument("--normalize-logx",  action="store_true", default=False, dest="normalize_logx")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--log-every",    type=int,   default=20, dest="log_every")
    p.add_argument("--device",       type=str,   default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
