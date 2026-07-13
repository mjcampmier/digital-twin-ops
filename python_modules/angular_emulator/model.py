"""
Layer-2 angular Mie emulator — AngularEmulator model.

Architecture
------------
Inputs: (x, n, k, μ)  where μ = cos(θ) ∈ [-1, 1].
  - log(x) with Fourier embedding (captures size-parameter ripples; same as Layer 1)
  - μ with Legendre polynomial embedding P_0…P_n_legendre(μ)  [S3 default]
    OR Fourier embedding (S1/S2 option, kept for comparison)
  - Raw normalized (logx, μ, n, k) also concatenated

Outputs: (log(i₁), log(i₂)) where iₚ = |Sₚ|² (miepython bohren norm).
  - exp() recovers physical intensities
  - log target handles the ~10-decade dynamic range of the angular pattern

Activations: SiLU throughout (C∞, smooth gradients — required for ∂/∂θ, ∂/∂x).

Why Legendre over Fourier on μ (S3 motivation):
  The Mie angular functions S1(cos θ), S2(cos θ) are finite sums of angular momentum
  harmonics, which expand exactly in Legendre polynomials P_n(μ).  For x≤80 the
  series truncates at N_max ≈ x+4x^(1/3)+2 ≈ 100 terms.  Legendre features are:
    • Deterministic — no random frequency sampling, no per-particle overfitting
    • Bounded: |P_n(μ)| ≤ 1 for all μ ∈ [-1,1], no feature explosion
    • Physics-aligned: exactly the basis Mie uses for angular expansion
  Random Fourier features on μ (S1/S2) overfit to the specific training angles
  per particle and fail to generalize; Legendre features are globally defined.

Normalization identity (for criterion 4):
  ∫₋₁^1 (i₁ + i₂) dμ = 4 · x² · Q_sca
  ∫₋₁^1 μ (i₁ + i₂) dμ = 4 · x² · Q_sca · g
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Fourier embedding (same implementation as Layer 1 — used for log(x))
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Random Fourier features on a single scalar.

    forward(v) : (...,) → (..., 2·n_freq)   [cos(B·v), sin(B·v)]
    """
    def __init__(self, n_freq: int, scale: float, seed: int = 0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        B   = torch.randn(n_freq, generator=rng) * scale
        self.register_buffer("B", B)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        proj = v.unsqueeze(-1) * self.B          # (..., n_freq)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# ---------------------------------------------------------------------------
# Legendre polynomial embedding — physics-aligned angular basis
# ---------------------------------------------------------------------------

class LegendreEmbedding(nn.Module):
    """
    Legendre polynomial features P_0(μ), P_1(μ), …, P_n_max(μ).

    Computed via the three-term recurrence (differentiable, O(n_max)):
        P_{k+1}(μ) = ((2k+1)·μ·P_k(μ) − k·P_{k-1}(μ)) / (k+1)

    For x_max=80: N_max ≈ 80 + 4·80^(1/3) + 2 ≈ 100 — use n_max=100.
    Output shape: (..., n_max+1).  All values in [-1, 1].
    """
    def __init__(self, n_max: int = 100):
        super().__init__()
        self.n_max = n_max

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        # mu: (...,)  →  (..., n_max+1)
        P_prev = torch.ones_like(mu)      # P_0 = 1
        P_curr = mu.clone()               # P_1 = μ
        polys  = [P_prev, P_curr]
        for k in range(1, self.n_max):
            P_next = ((2 * k + 1) * mu * P_curr - k * P_prev) / (k + 1)
            P_prev = P_curr
            P_curr = P_next
            polys.append(P_curr)
        return torch.stack(polys, dim=-1)  # (..., n_max+1)


# ---------------------------------------------------------------------------
# Angular emulator
# ---------------------------------------------------------------------------

# Layer-1 logx normalisation constants — kept identical for composability
_LOGX_CENTER    = (np.log(0.03) + np.log(80.0)) / 2          # ≈ 0.4365
_LOGX_HALFRANGE = (np.log(80.0) - np.log(0.03)) / 2          # ≈ 3.9383

# n and k normalization constants (Layer-1 domain)
_N_CENTER, _N_HALF = (1.33 + 1.80) / 2, (1.80 - 1.33) / 2   # 1.565, 0.235
_K_CENTER, _K_HALF = (0.00 + 0.80) / 2, (0.80 - 0.00) / 2   # 0.40,  0.40


class AngularEmulator(nn.Module):
    """
    Layer-2 angular scattering emulator.

    Parameters
    ----------
    hidden_dim      : width of each hidden layer
    n_hidden        : number of hidden layers
    n_fourier_x     : Fourier features on log(x) — captures size-parameter ripples
    fourier_scale_x : bandwidth of Fourier features on log(x) (same as Layer 1)
    n_legendre      : Legendre poly features P_0..P_n on μ (S3 default; 0 = disable)
    n_fourier_mu    : Fourier features on μ (S1/S2; 0 = disable, recommended for S3)
    fourier_scale_mu: bandwidth of Fourier features on μ
    """

    def __init__(
        self,
        hidden_dim: int        = 512,
        n_hidden: int          = 8,
        n_fourier_x: int       = 256,
        fourier_scale_x: float = 7.0,
        n_legendre: int        = 100,   # Legendre on μ (S3 default)
        n_fourier_mu: int      = 0,     # Fourier on μ (S1/S2; off by default)
        fourier_scale_mu: float = 30.0,
    ):
        super().__init__()
        self.n_fourier_x  = n_fourier_x
        self.n_fourier_mu = n_fourier_mu
        self.n_legendre   = n_legendre

        if n_fourier_x > 0:
            self.fourier_x  = FourierEmbedding(n_fourier_x, fourier_scale_x, seed=0)
        if n_fourier_mu > 0:
            self.fourier_mu = FourierEmbedding(n_fourier_mu, fourier_scale_mu, seed=1)
        if n_legendre > 0:
            self.legendre   = LegendreEmbedding(n_legendre)

        # Input dimension:
        # logx_norm(1) + Fourier_x(2*n_fx) + mu_raw(1) + Legendre(n_leg+1) + Fourier_mu(2*n_fmu) + n,k(2)
        in_dim = (1 + 2 * n_fourier_x) + 1 + (n_legendre + 1) + (2 * n_fourier_mu) + 2

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))  # [log(i1), log(i2)]

        self.net = nn.Sequential(*layers)

    def _encode(
        self,
        x: torch.Tensor,    # size parameter, shape (N,)
        n: torch.Tensor,    # real RI
        k: torch.Tensor,    # imaginary RI
        mu: torch.Tensor,   # cos(theta), in [-1, 1]
    ) -> torch.Tensor:
        logx   = torch.log(x)
        logx_n = (logx - _LOGX_CENTER) / _LOGX_HALFRANGE
        n_n    = (n - _N_CENTER) / _N_HALF
        k_n    = (k - _K_CENTER) / _K_HALF

        feats = [logx_n.unsqueeze(-1)]
        if self.n_fourier_x > 0:
            feats.append(self.fourier_x(logx))
        feats.append(mu.unsqueeze(-1))
        if self.n_legendre > 0:
            feats.append(self.legendre(mu))
        if self.n_fourier_mu > 0:
            feats.append(self.fourier_mu(mu))
        feats.append(n_n.unsqueeze(-1))
        feats.append(k_n.unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        k: torch.Tensor,
        mu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (i₁, i₂) — physical intensities (positive).
        x, n, k, mu: 1-D tensors of the same length.
        """
        inp = self._encode(x, n, k, mu)
        out = self.net(inp)
        return torch.exp(out[..., 0]), torch.exp(out[..., 1])

    def forward_log(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        k: torch.Tensor,
        mu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_i1, log_i2) — for loss computation without exp/log round-trip."""
        inp = self._encode(x, n, k, mu)
        out = self.net(inp)
        return out[..., 0], out[..., 1]
