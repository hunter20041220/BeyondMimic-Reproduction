"""Sampling and augmentation helpers for debug reproduction gates."""

from __future__ import annotations

import math

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def adaptive_distribution(failure_bin: int, bin_count: int, kernel_size: int, floor_mass: float = 0.05) -> np.ndarray:
    """Build a normalized failure-window distribution with shape ``[bin_count]``."""
    if bin_count <= 0 or kernel_size <= 0:
        raise ValueError("bin_count and kernel_size must be positive")
    if not 0 <= failure_bin < bin_count:
        raise ValueError(f"failure_bin {failure_bin} outside [0,{bin_count})")
    if not math.isfinite(floor_mass) or floor_mass < 0.0:
        raise ValueError("floor_mass must be finite and nonnegative")
    weights = np.full(bin_count, floor_mass / bin_count, dtype=np.float64)
    for u in range(kernel_size):
        idx = failure_bin - u
        if 0 <= idx < bin_count:
            weights[idx] += math.exp(-u)
    return weights / weights.sum()


def ou_noise(seed: int, steps: int, dim: int, theta: float = 0.35, sigma: float = 0.15) -> np.ndarray:
    """Generate finite OU perturbations with shape ``[steps, dim]`` from a fixed seed."""
    if steps <= 0 or dim <= 0:
        raise ValueError("steps and dim must be positive")
    if not all(math.isfinite(v) for v in [theta, sigma]):
        raise ValueError("theta and sigma must be finite")
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, dim), dtype=np.float64)
    for i in range(1, steps):
        x[i] = x[i - 1] + theta * (0.0 - x[i - 1]) + sigma * rng.normal(size=dim)
    return x


def mirror_state_29d(vec: np.ndarray) -> np.ndarray:
    """Mirror finite Unitree G1 action/state vectors in sagittal frame ``[29]``."""
    vec = ensure_finite("vec", vec)
    if vec.shape != (29,):
        raise ValueError(f"expected 29D vector, got {vec.shape}")
    mirrored = vec.copy()
    pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (12, 18), (13, 19)]
    sign_flip = {2, 5, 8, 11, 14, 17, 20, 23, 26}
    for left, right in pairs:
        mirrored[left], mirrored[right] = vec[right], vec[left]
    for idx in sign_flip:
        mirrored[idx] *= -1.0
    return mirrored
