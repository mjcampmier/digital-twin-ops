"""
Layer-1 Mie emulator — MLP with optional Fourier features on x.

Architecture
------------
Input: [log(x), n, k] (3-dim) optionally augmented with Fourier features on x.
Hidden: configurable depth × width, SiLU activations (smooth, no ReLU).
Output: [asinh(Q_sca), Q_ext, g]  — asinh handles Q_sca dynamic range; Q_ext
        and g are well-behaved raw.

Differentiability is structural: SiLU is C∞, so ∂Q_sca/∂x is smooth and
correct under torch.autograd.
"""

import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    """Random Fourier features on a single scalar input (here log(x)).

    Embedding: [cos(B·v), sin(B·v)] where B ~ N(0, scale²).
    Dimension doubles: scalar → 2*n_freq features.
    """
    def __init__(self, n_freq: int = 64, scale: float = 3.0, seed: int = 0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        B = torch.randn(n_freq, generator=rng) * scale   # (n_freq,)
        self.register_buffer("B", B)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (...,)  →  (..., 2*n_freq)
        proj = v.unsqueeze(-1) * self.B          # (..., n_freq)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MieEmulator(nn.Module):
    """
    MLP Mie emulator.

    Parameters
    ----------
    hidden_dim   : width of each hidden layer
    n_layers     : number of hidden layers
    n_fourier    : Fourier features on log(x); 0 = disable
    fourier_scale: bandwidth of random Fourier features
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_fourier: int = 64,
        fourier_scale: float = 3.0,
        include_logx: bool = True,
        normalize_logx: bool = False,   # True → maps logx to [-1,1]; False → raw log(x) (round-3 compat)
    ):
        super().__init__()
        self.n_fourier = n_fourier
        self.include_logx = include_logx
        self.normalize_logx = normalize_logx

        if n_fourier > 0:
            self.fourier = FourierEmbedding(n_fourier, fourier_scale)
            in_dim = 2 * n_fourier + (3 if include_logx else 2)  # optional log(x) feature + [n, k]
        else:
            in_dim = 3                   # [log(x), n, k]

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 3))   # [asinh_Qsca, Qext, g]

        self.net = nn.Sequential(*layers)

    # log(x) domain: x ∈ [0.03, 80] → logx ∈ [-3.51, 4.38]; center/half-range for normalisation
    _LOGX_CENTER    = (torch.tensor(-3.507) + torch.tensor(4.382)) / 2   # ≈ 0.437
    _LOGX_HALFRANGE = (torch.tensor(4.382)  - torch.tensor(-3.507)) / 2  # ≈ 3.945

    def _encode(self, x: torch.Tensor, n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        logx = torch.log(x)
        if self.n_fourier > 0:
            feat = self.fourier(logx)         # (..., 2*n_fourier)
            if self.include_logx:
                if self.normalize_logx:
                    # Normalize to [-1, 1] — same scale as cos/sin features (round 4+)
                    lx_in = (logx - self._LOGX_CENTER.to(logx)) / self._LOGX_HALFRANGE.to(logx)
                else:
                    # Raw log(x) — backward compat for round-3 checkpoint
                    lx_in = logx
                return torch.cat([lx_in.unsqueeze(-1), feat, n.unsqueeze(-1), k.unsqueeze(-1)], dim=-1)
            return torch.cat([feat, n.unsqueeze(-1), k.unsqueeze(-1)], dim=-1)
        return torch.stack([logx, n, k], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Q_sca, Q_ext, g) — physical-space outputs.
        x, n, k: 1-D tensors of same length.
        """
        inp  = self._encode(x, n, k)
        out  = self.net(inp)                  # (..., 3)
        Qsca = torch.exp(out[..., 0])         # net predicts log(Q_sca); exp handles 8-decade dynamic range
        Qext = out[..., 1]
        g    = out[..., 2]
        return Qsca, Qext, g

    def forward_Qsca_only(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Single-output forward for gradient checks (retains graph on x)."""
        inp  = self._encode(x, n, k)
        out  = self.net(inp)
        return torch.exp(out[..., 0])


def input_stats(data: dict) -> dict:
    """Compute normalisation constants from training data."""
    x_tr = data["x_tr"].astype(np.float32)
    n_tr = data["n_tr"].astype(np.float32)
    k_tr = data["k_tr"].astype(np.float32)
    logx = np.log(x_tr)
    return {
        "logx_mean": float(logx.mean()),
        "logx_std":  float(logx.std()),
        "n_mean":    float(n_tr.mean()),
        "n_std":     float(n_tr.std()),
        "k_mean":    float(k_tr.mean()),
        "k_std":     float(k_tr.std()),
    }
