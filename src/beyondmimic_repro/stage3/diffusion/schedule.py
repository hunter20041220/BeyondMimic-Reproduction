"""Diffusion schedules for state-latent trajectories."""

from __future__ import annotations

import math

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-3 diffusion schedules") from exc


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return a linear beta schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)


def cosine_alpha_bar_schedule(steps: int, s: float = 0.008, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return cosine cumulative alpha bars."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float32, device=device)
    alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar[1:].clamp(min=1e-6, max=1.0)


def alpha_bars_from_betas(betas: torch.Tensor) -> torch.Tensor:
    """Convert betas to cumulative alpha bars."""
    return torch.cumprod(1.0 - betas, dim=0)
