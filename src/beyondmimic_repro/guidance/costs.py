"""Reward and guidance cost primitives used by local probes."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def gaussian_reward(error: np.ndarray, sigma: float) -> float:
    """Exponentiated squared-error reward over finite error vector ``[D]``."""
    error = ensure_finite("error", error)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return float(math.exp(-float(np.dot(error, error)) / (sigma * sigma)))


def finite_difference_grad(fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient for finite scalar cost inputs ``[D]``."""
    x = ensure_finite("x", x)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in range(x.size):
        step = np.zeros_like(x, dtype=np.float64)
        step[idx] = eps
        grad[idx] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
    return grad


def sdf_barrier(distance: np.ndarray, delta: float = 0.1) -> float:
    """BeyondMimic relaxed SDF barrier summed over finite distances ``[D]``.

    The paper defines ``B(x, delta)`` as ``-log(x)`` for ``x >= delta`` and a
    quadratic relaxation otherwise:

    ``-log(delta) + 0.5 * (((x - 2 * delta) / delta) ** 2 - 1)``.
    """
    distance = ensure_finite("distance", distance)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    value = np.empty_like(distance, dtype=np.float64)
    smooth_region = distance < delta
    if np.any(~smooth_region):
        value[~smooth_region] = -np.log(distance[~smooth_region])
    if np.any(smooth_region):
        value[smooth_region] = -math.log(delta) + 0.5 * (
            ((distance[smooth_region] - 2.0 * delta) / delta) ** 2 - 1.0
        )
    return float(np.sum(value))
