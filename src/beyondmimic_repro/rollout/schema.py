"""Teacher rollout dataset schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_teacher_rollout(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite rollout arrays with shapes `[N,T,S]` and `[N,T,A]`."""
    state_arr = np.asarray(states, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    if state_arr.ndim != 3 or action_arr.ndim != 3:
        raise ValueError(f"states/actions must be [N,T,D], got {state_arr.shape}, {action_arr.shape}")
    if state_arr.shape[:2] != action_arr.shape[:2]:
        raise ValueError(f"states/actions must share [N,T], got {state_arr.shape} and {action_arr.shape}")
    if not np.all(np.isfinite(state_arr)) or not np.all(np.isfinite(action_arr)):
        raise ValueError("teacher rollout contains NaN or Inf")
    return state_arr, action_arr


def save_teacher_rollout(path: str | Path, states: np.ndarray, actions: np.ndarray, names: np.ndarray | None = None) -> dict[str, object]:
    """Save a compact teacher rollout NPZ file."""
    state_arr, action_arr = validate_teacher_rollout(states, actions)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = np.array([f"rollout_{idx:04d}" for idx in range(state_arr.shape[0])])
    np.savez_compressed(output, states=state_arr, actions=action_arr, names=names)
    return {
        "status": "ok",
        "output": str(output),
        "rollout_count": int(state_arr.shape[0]),
        "horizon": int(state_arr.shape[1]),
        "state_dim": int(state_arr.shape[2]),
        "action_dim": int(action_arr.shape[2]),
    }


def load_teacher_rollout(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a teacher rollout NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        states = data["states"]
        actions = data["actions"]
        names = data["names"] if "names" in data else np.array([f"rollout_{idx:04d}" for idx in range(states.shape[0])])
    state_arr, action_arr = validate_teacher_rollout(states, actions)
    return state_arr, action_arr, names
