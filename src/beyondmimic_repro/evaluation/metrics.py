"""Basic metric helpers used by DAgger and trajectory audits."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def action_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean squared action error for finite action arrays ``[..., action_dim]``."""
    pred = ensure_finite("predicted", predicted)
    tgt = ensure_finite("target", target)
    if pred.shape != tgt.shape:
        raise ValueError(f"predicted shape {pred.shape} != target shape {tgt.shape}")
    return float(np.mean((pred - tgt) ** 2))


def tracking_error(predicted_positions: np.ndarray, target_positions: np.ndarray) -> dict[str, float]:
    """Mean/max Euclidean tracking error for position trajectories ``[T,J,3]``."""
    pred = ensure_finite("predicted_positions", predicted_positions)
    tgt = ensure_finite("target_positions", target_positions)
    if pred.shape != tgt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"position trajectories must both have shape [T,J,3], got {pred.shape}, {tgt.shape}")
    distances = np.linalg.norm(pred - tgt, axis=-1)
    return {"mean_error": float(np.mean(distances)), "max_error": float(np.max(distances))}


def survival_rate(episode_lengths: np.ndarray, horizon: int) -> float:
    """Fraction of finite trajectory episode lengths ``[N]`` that survive horizon."""
    lengths = ensure_finite("episode_lengths", episode_lengths)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if lengths.ndim != 1:
        raise ValueError(f"episode_lengths must have shape [N], got {lengths.shape}")
    return float(np.mean(lengths >= horizon))


def success_rate(success_flags: np.ndarray) -> float:
    """Fraction of finite binary success flags ``[N]`` that are successful."""
    flags = ensure_finite("success_flags", success_flags)
    if flags.ndim != 1:
        raise ValueError(f"success_flags must have shape [N], got {flags.shape}")
    if flags.size == 0:
        raise ValueError("success_flags must be nonempty")
    if not np.all((flags == 0) | (flags == 1)):
        raise ValueError("success_flags must be binary 0/1 values")
    return float(np.mean(flags))


def fall_rate(fall_flags: np.ndarray) -> float:
    """Fraction of finite binary fall flags ``[N]`` that record a fall."""
    return success_rate(fall_flags)


def velocity_tracking_error(predicted_velocity: np.ndarray, target_velocity: np.ndarray) -> dict[str, float]:
    """Mean/max Euclidean velocity tracking error for trajectories ``[T,D]``."""
    pred = ensure_finite("predicted_velocity", predicted_velocity)
    tgt = ensure_finite("target_velocity", target_velocity)
    if pred.shape != tgt.shape or pred.ndim != 2:
        raise ValueError(f"velocity trajectories must both have shape [T,D], got {pred.shape}, {tgt.shape}")
    errors = np.linalg.norm(pred - tgt, axis=-1)
    return {
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean((pred - tgt) ** 2))),
    }


def split_metric_summary(values: np.ndarray, splits: Iterable[str]) -> dict[str, dict[str, float]]:
    """Summarize finite scalar metric values by split label."""
    arr = ensure_finite("values", values)
    split_list = list(splits)
    if arr.ndim != 1:
        raise ValueError(f"values must have shape [N], got {arr.shape}")
    if arr.size != len(split_list):
        raise ValueError(f"values length {arr.size} != splits length {len(split_list)}")
    summary: dict[str, dict[str, float]] = {}
    for split in sorted(set(split_list)):
        mask = np.array([item == split for item in split_list], dtype=bool)
        subset = arr[mask]
        summary[split] = {
            "count": float(subset.size),
            "mean": float(np.mean(subset)),
            "std": float(np.std(subset)),
            "min": float(np.min(subset)),
            "max": float(np.max(subset)),
        }
    return summary
