"""Teacher closed-loop rollout contract for D0 warm starts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


TEACHER_ROLLOUT_SCHEMA_VERSION = "stage2-teacher-rollout-v1"


@dataclass(frozen=True)
class TeacherRolloutMetadata:
    """Metadata that distinguishes teacher rollout data from reference motion."""

    schema_version: str = TEACHER_ROLLOUT_SCHEMA_VERSION
    frequency_hz: float = 50.0
    source: str = "teacher closed-loop rollout"
    action_semantics: str = "normalized 29-D teacher policy action"
    intended_use: str = "D0 BC warm start only; not a completed DAgger dataset"


def validate_teacher_rollout_arrays(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate a teacher rollout NPZ-like payload."""
    required = ["policy_observation", "teacher_action"]
    for key in required:
        if key not in payload:
            raise ValueError(f"teacher rollout is missing {key!r}")
    obs = np.asarray(payload["policy_observation"], dtype=np.float32)
    act = np.asarray(payload["teacher_action"], dtype=np.float32)
    if obs.ndim != 3 or act.ndim != 3:
        raise ValueError(f"policy_observation/teacher_action must be [T,E,D], got {obs.shape}, {act.shape}")
    if obs.shape[:2] != act.shape[:2]:
        raise ValueError(f"policy_observation/teacher_action [T,E] mismatch: {obs.shape}, {act.shape}")
    if act.shape[-1] != 29:
        raise ValueError(f"teacher_action last dimension must be 29, got {act.shape}")
    for key, value in payload.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains NaN or Inf")
    return {"policy_observation": obs, "teacher_action": act}


def save_teacher_rollout_contract(
    path: str | Path,
    payload: dict[str, np.ndarray],
    metadata: TeacherRolloutMetadata | None = None,
) -> dict[str, Any]:
    """Save a teacher rollout with contract metadata."""
    checked = validate_teacher_rollout_arrays(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata or TeacherRolloutMetadata()
    np.savez_compressed(output, **payload, metadata_json=json.dumps(asdict(meta), sort_keys=True))
    return {
        "schema_version": meta.schema_version,
        "output": str(output),
        "step_count": int(checked["policy_observation"].shape[0]),
        "env_count": int(checked["policy_observation"].shape[1]),
        "obs_dim": int(checked["policy_observation"].shape[-1]),
        "action_dim": int(checked["teacher_action"].shape[-1]),
    }


def load_teacher_rollout_contract(path: str | Path) -> tuple[dict[str, np.ndarray], TeacherRolloutMetadata]:
    """Load a teacher rollout contract file."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata_raw = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
    validate_teacher_rollout_arrays(payload)
    meta_dict = json.loads(metadata_raw) if metadata_raw else {}
    return payload, TeacherRolloutMetadata(**{**asdict(TeacherRolloutMetadata()), **meta_dict})
