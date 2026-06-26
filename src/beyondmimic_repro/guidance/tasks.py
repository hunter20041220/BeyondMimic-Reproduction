"""Test-time guidance costs over state-latent trajectories."""

from __future__ import annotations

import numpy as np

from beyondmimic_repro.guidance.costs import sdf_barrier
from beyondmimic_repro.validation import ensure_finite


def root_xy(tokens: np.ndarray) -> np.ndarray:
    """Return root XY positions from token/state trajectories."""
    arr = ensure_finite("tokens", tokens)
    if arr.shape[-1] < 2:
        raise ValueError("tokens must contain root x/y in the first two channels")
    return arr[..., :2]


def joystick_cost(tokens: np.ndarray, target_velocity_xy: np.ndarray) -> float:
    """Velocity-tracking cost from finite-difference root XY motion."""
    xy = root_xy(tokens)
    target = ensure_finite("target_velocity_xy", target_velocity_xy)
    if target.shape != (2,):
        raise ValueError(f"target_velocity_xy must have shape [2], got {target.shape}")
    velocity = np.diff(xy, axis=-2)
    return float(np.mean((velocity - target) ** 2))


def waypoint_cost(tokens: np.ndarray, waypoint_xy: np.ndarray) -> float:
    """Final-position waypoint cost."""
    xy = root_xy(tokens)
    waypoint = ensure_finite("waypoint_xy", waypoint_xy)
    if waypoint.shape != (2,):
        raise ValueError(f"waypoint_xy must have shape [2], got {waypoint.shape}")
    return float(np.sum((xy[..., -1, :] - waypoint) ** 2))


def obstacle_cost(tokens: np.ndarray, obstacle_xy: np.ndarray, radius: float, delta: float = 0.1) -> float:
    """SDF-style obstacle barrier along root XY path."""
    xy = root_xy(tokens)
    center = ensure_finite("obstacle_xy", obstacle_xy)
    if center.shape != (2,):
        raise ValueError(f"obstacle_xy must have shape [2], got {center.shape}")
    distance = np.linalg.norm(xy - center, axis=-1) - float(radius)
    return sdf_barrier(distance, delta=delta)


def inpainting_cost(tokens: np.ndarray, keyframe_index: int, target_token: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Keyframe inpainting reconstruction cost."""
    arr = ensure_finite("tokens", tokens)
    target = ensure_finite("target_token", target_token)
    if not 0 <= keyframe_index < arr.shape[-2]:
        raise ValueError("keyframe_index outside trajectory")
    if target.shape != (arr.shape[-1],):
        raise ValueError(f"target_token must have shape [{arr.shape[-1]}], got {target.shape}")
    diff = arr[..., keyframe_index, :] - target
    if mask is not None:
        mask_arr = ensure_finite("mask", mask)
        if mask_arr.shape != target.shape:
            raise ValueError(f"mask must have shape {target.shape}, got {mask_arr.shape}")
        diff = diff * mask_arr
    return float(np.mean(diff**2))


def evaluate_guidance_suite(tokens: np.ndarray) -> dict[str, float]:
    """Compute a small guided-vs-unguided metric suite for smoke tests."""
    arr = ensure_finite("tokens", tokens)
    return {
        "joystick_cost": joystick_cost(arr, np.array([0.03, 0.0], dtype=np.float64)),
        "waypoint_cost": waypoint_cost(arr, np.array([1.0, 0.0], dtype=np.float64)),
        "obstacle_cost": obstacle_cost(arr, np.array([0.5, 0.0], dtype=np.float64), radius=0.2),
        "inpainting_cost": inpainting_cost(arr, arr.shape[-2] // 2, arr[..., arr.shape[-2] // 2, :].mean(axis=0)),
    }
