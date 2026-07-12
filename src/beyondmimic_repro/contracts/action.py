"""Normalized action contract shared by Stage-2/3 frontends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DIM = 29


@dataclass(frozen=True)
class NormalizedActionSpec:
    """A policy output before action scale, PD targets, or torque clipping."""

    dim: int = ACTION_DIM
    frame: str = "robot_joint_order"
    units: str = "normalized action"
    semantic_meaning: str = "decoder output used by backend as default_joint_pos + action_scale * action"


def validate_normalized_action(action: np.ndarray, *, action_dim: int = ACTION_DIM) -> np.ndarray:
    """Validate finite normalized actions with trailing dimension 29."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != action_dim:
        raise ValueError(f"normalized action last dimension must be {action_dim}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("normalized action contains NaN or Inf")
    return arr
