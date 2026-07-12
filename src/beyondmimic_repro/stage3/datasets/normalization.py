"""Mean/std normalization metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.mean) / np.maximum(self.std, self.eps)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * np.maximum(self.std, self.eps) + self.mean


def fit_normalization(values: np.ndarray, eps: float = 1e-6) -> NormalizationStats:
    """Fit per-channel statistics from finite arrays."""
    arr = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contain NaN or Inf")
    return NormalizationStats(mean=arr.mean(axis=tuple(range(arr.ndim - 1))), std=arr.std(axis=tuple(range(arr.ndim - 1))), eps=eps)
