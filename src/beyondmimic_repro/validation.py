"""Validation helpers shared by lightweight formula implementations."""

from __future__ import annotations

from typing import Any

import numpy as np


def ensure_finite(name: str, value: Any) -> np.ndarray:
    """Return ``value`` as float64 and reject NaN/Inf entries.

    The returned array keeps the input shape. Callers use this for paper-formula
    tensors such as Rot6D vectors ``[6]``, trajectory windows ``[T, D]``, and
    anchor/world-frame points ``[..., 3]`` before applying coordinate transforms
    or loss terms.
    """
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array
