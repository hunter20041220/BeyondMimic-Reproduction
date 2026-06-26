"""DDPM-style forward and oracle reverse helpers for audits."""

from __future__ import annotations

import math

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def q_sample(x0: np.ndarray, eps: np.ndarray, alpha_bar: float) -> np.ndarray:
    """Sample noisy trajectory tokens ``x_t[T,D]`` from clean ``x0[T,D]``."""
    x0 = ensure_finite("x0", x0)
    eps = ensure_finite("eps", eps)
    if x0.shape != eps.shape:
        raise ValueError(f"x0 shape {x0.shape} != eps shape {eps.shape}")
    if not 0.0 <= alpha_bar <= 1.0 or not math.isfinite(alpha_bar):
        raise ValueError(f"alpha_bar must be finite in [0,1], got {alpha_bar}")
    return math.sqrt(alpha_bar) * x0 + math.sqrt(1.0 - alpha_bar) * eps


def denoise_one_step_with_oracle_eps(x_t: np.ndarray, x0: np.ndarray, alpha_bar_t: float, alpha_bar_prev: float) -> np.ndarray:
    """Move one reverse step for trajectory tokens ``[T,D]`` when clean ``x0`` is known."""
    x_t = ensure_finite("x_t", x_t)
    x0 = ensure_finite("x0", x0)
    if x_t.shape != x0.shape:
        raise ValueError(f"x_t shape {x_t.shape} != x0 shape {x0.shape}")
    for name, value in {"alpha_bar_t": alpha_bar_t, "alpha_bar_prev": alpha_bar_prev}.items():
        if not 0.0 < value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite in (0,1], got {value}")
    eps = (x_t - math.sqrt(alpha_bar_t) * x0) / math.sqrt(1.0 - alpha_bar_t)
    return math.sqrt(alpha_bar_prev) * x0 + math.sqrt(1.0 - alpha_bar_prev) * eps


def apply_observation_mask(noisy: np.ndarray, clean: np.ndarray, mask_observed: np.ndarray) -> np.ndarray:
    """Clamp observed history/keyframe tokens in trajectory tensors ``[T,D]``."""
    noisy = ensure_finite("noisy", noisy)
    clean = ensure_finite("clean", clean)
    if noisy.shape != clean.shape or noisy.shape != mask_observed.shape:
        raise ValueError(f"mask shapes must match noisy/clean, got {noisy.shape}, {clean.shape}, {mask_observed.shape}")
    out = noisy.copy()
    out[mask_observed] = clean[mask_observed]
    return out
