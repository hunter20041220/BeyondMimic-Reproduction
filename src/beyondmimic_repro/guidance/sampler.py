"""Dependency-light guided trajectory adjustment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from beyondmimic_repro.guidance.costs import finite_difference_grad
from beyondmimic_repro.validation import ensure_finite


def gradient_guidance_step(
    tokens: np.ndarray,
    cost_fn: Callable[[np.ndarray], float],
    *,
    step_size: float = 1e-2,
    guided_dims: int = 2,
    finite_difference_eps: float = 1e-4,
) -> tuple[np.ndarray, float, float]:
    """Apply one finite-difference guidance step to trajectory tokens."""
    arr = ensure_finite("tokens", tokens)
    original_cost = float(cost_fn(arr))
    if guided_dims <= 0 or guided_dims > arr.shape[-1]:
        raise ValueError(f"guided_dims must be in [1,{arr.shape[-1]}], got {guided_dims}")

    guided_slice = arr[..., :guided_dims]
    base = arr.copy()

    def wrapped(flat_guided: np.ndarray) -> float:
        candidate = base.copy()
        candidate[..., :guided_dims] = flat_guided.reshape(guided_slice.shape)
        return float(cost_fn(candidate))

    grad = finite_difference_grad(wrapped, guided_slice.reshape(-1), eps=finite_difference_eps)
    updated = arr.copy()
    updated[..., :guided_dims] = guided_slice - step_size * grad.reshape(guided_slice.shape)
    updated_cost = float(cost_fn(updated))
    return updated.astype(np.float32), original_cost, updated_cost
