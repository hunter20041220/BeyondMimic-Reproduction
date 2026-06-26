"""Rotation and anchor-frame transforms used by paper-state audits."""

from __future__ import annotations

import math

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a finite vector while preserving its one-dimensional shape."""
    vec = ensure_finite("vec", vec)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("cannot normalize zero vector")
    return vec / norm


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert finite Rot6D ``[6]`` orientation features to a matrix ``[3,3]``."""
    rot6d = ensure_finite("rot6d", rot6d)
    if rot6d.shape[0] < 6:
        raise ValueError(f"rot6d shape {rot6d.shape} does not contain 6 values")
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def yaw_matrix(yaw: float) -> np.ndarray:
    """Build a finite world-z yaw rotation matrix with shape ``[3,3]``."""
    if not math.isfinite(yaw):
        raise ValueError("yaw contains NaN or Inf")
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def world_to_anchor(points: np.ndarray, anchor_pos: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Express finite world-frame points ``[...,3]`` in the anchor yaw frame."""
    points = ensure_finite("points", points)
    anchor_pos = ensure_finite("anchor_pos", anchor_pos)
    if points.shape[-1] != 3 or anchor_pos.shape[-1] != 3:
        raise ValueError(f"expected points/anchor_pos last dimension 3, got {points.shape}, {anchor_pos.shape}")
    return (points - anchor_pos) @ yaw_matrix(anchor_yaw)


def anchor_to_world(local: np.ndarray, anchor_pos: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Map finite anchor yaw-frame points ``[...,3]`` back to world frame."""
    local = ensure_finite("local", local)
    anchor_pos = ensure_finite("anchor_pos", anchor_pos)
    if local.shape[-1] != 3 or anchor_pos.shape[-1] != 3:
        raise ValueError(f"expected local/anchor_pos last dimension 3, got {local.shape}, {anchor_pos.shape}")
    return local @ yaw_matrix(anchor_yaw).T + anchor_pos
