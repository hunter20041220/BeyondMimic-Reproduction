"""Character-yaw-centric transforms for Stage-3 state tokens."""

from __future__ import annotations

import numpy as np

from beyondmimic_repro.state import build_paper_hybrid_state_window
from beyondmimic_repro.validation import ensure_finite


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """Return a world-z yaw rotation matrix."""
    c = np.cos(float(yaw))
    s = np.sin(float(yaw))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def world_to_character_yaw(points_w: np.ndarray, origin_w: np.ndarray, yaw_w: float) -> np.ndarray:
    """Express world-frame points in the character yaw frame."""
    points = ensure_finite("points_w", points_w)
    origin = ensure_finite("origin_w", origin_w)
    if points.shape[-1] != 3 or origin.shape[-1] != 3:
        raise ValueError("points_w and origin_w must have trailing dimension 3")
    return (points - origin) @ yaw_rotation_matrix(-yaw_w).T


def character_yaw_to_world(points_c: np.ndarray, origin_w: np.ndarray, yaw_w: float) -> np.ndarray:
    """Map character-frame points back to world coordinates."""
    points = ensure_finite("points_c", points_c)
    origin = ensure_finite("origin_w", origin_w)
    if points.shape[-1] != 3 or origin.shape[-1] != 3:
        raise ValueError("points_c and origin_w must have trailing dimension 3")
    return points @ yaw_rotation_matrix(yaw_w).T + origin


def build_character_yaw_state_window(*args: object, **kwargs: object) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Build the paper hybrid state using the shared audited implementation."""
    return build_paper_hybrid_state_window(*args, **kwargs)
