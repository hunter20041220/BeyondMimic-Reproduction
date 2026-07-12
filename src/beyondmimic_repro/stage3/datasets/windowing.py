"""Contiguous window builders."""

from __future__ import annotations

import numpy as np


def build_contiguous_windows(array: np.ndarray, sequence_length: int) -> np.ndarray:
    """Build sliding windows from [T,D] or [E,T,D] arrays."""
    arr = np.asarray(array)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if arr.ndim == 2:
        if arr.shape[0] < sequence_length:
            return np.zeros((0, sequence_length, arr.shape[-1]), dtype=arr.dtype)
        return np.stack([arr[start : start + sequence_length] for start in range(arr.shape[0] - sequence_length + 1)], axis=0)
    if arr.ndim == 3:
        windows = [build_contiguous_windows(episode, sequence_length) for episode in arr]
        return np.concatenate(windows, axis=0) if windows else np.zeros((0, sequence_length, arr.shape[-1]), dtype=arr.dtype)
    raise ValueError(f"array must be [T,D] or [E,T,D], got {arr.shape}")
