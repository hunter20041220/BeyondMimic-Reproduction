"""Build teacher-rollout fixtures from prepared motion arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beyondmimic_repro.rollout.schema import save_teacher_rollout


def motion_npz_to_teacher_rollouts(
    input_path: str | Path,
    output_path: str | Path,
    *,
    horizon: int = 64,
    stride: int = 16,
    max_windows: int | None = None,
) -> dict[str, object]:
    """Convert prepared 36-D motion rows to state/action rollout windows.

    This is a public-data fallback for the clean repository. A full IsaacLab
    tracking policy rollout can write the same `states/actions/names` NPZ schema.
    """
    with np.load(input_path, allow_pickle=False) as data:
        motions = data["motions"]
        lengths = data["lengths"]
        names = data["names"] if "names" in data else np.array([f"motion_{idx:04d}" for idx in range(motions.shape[0])])

    if horizon <= 1 or stride <= 0:
        raise ValueError("horizon must be > 1 and stride must be positive")

    state_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    window_names: list[str] = []
    for motion_idx in range(motions.shape[0]):
        length = int(lengths[motion_idx])
        motion = motions[motion_idx, :length]
        for start in range(0, max(0, length - horizon + 1), stride):
            window = motion[start : start + horizon]
            root = window[:, :7]
            action = window[:, 7:]
            velocity = np.vstack([np.zeros((1, action.shape[1]), dtype=action.dtype), np.diff(action, axis=0)])
            state_windows.append(np.concatenate([root, action, velocity], axis=1).astype(np.float32))
            action_windows.append(action.astype(np.float32))
            window_names.append(f"{names[motion_idx]}:{start:06d}")
            if max_windows is not None and len(state_windows) >= max_windows:
                break
        if max_windows is not None and len(state_windows) >= max_windows:
            break

    if not state_windows:
        raise ValueError("no rollout windows generated; reduce horizon or provide longer motions")

    summary = save_teacher_rollout(
        output_path,
        states=np.stack(state_windows, axis=0),
        actions=np.stack(action_windows, axis=0),
        names=np.array(window_names),
    )
    summary.update({"source": str(input_path), "stride": stride, "public_motion_fallback": True})
    return summary
