"""Paper-semantics DAgger dataset schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DAGGER_SCHEMA_VERSION = "stage2-dagger-v1"


DAGGER_REQUIRED_ARRAYS = (
    "encoder_reference_input",
    "decoder_proprio_input",
    "student_mu",
    "student_logvar",
    "student_latent",
    "student_action",
    "teacher_action",
    "policy_observation",
    "root_state",
    "joint_position",
    "joint_velocity",
    "previous_action",
    "reward",
    "done",
    "body_position_error",
    "body_orientation_error",
    "joint_position_error",
    "joint_velocity_error",
)


@dataclass(frozen=True)
class DAggerDatasetMetadata:
    """Metadata for an aggregated student-state/teacher-label dataset."""

    schema_version: str = DAGGER_SCHEMA_VERSION
    frequency_hz: float = 50.0
    joint_position_semantics: str = "relative_to_default"
    source: str = "student closed-loop states labeled by teacher"
    round_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["round_ids"] = list(self.round_ids)
        return data


def array_field_specs() -> dict[str, dict[str, object]]:
    """Return JSON-safe shape/dtype/frame docs for each DAgger array."""
    return {
        "encoder_reference_input": {
            "shape": ["N", 67],
            "dtype": "float32",
            "frame": "reference/current-anchor",
            "units": "rad, rad/s, m, Rot6D",
            "normalization": "dataset mean/std",
            "semantic_meaning": "reference-motion information for VAE encoder",
        },
        "decoder_proprio_input": {
            "shape": ["N", 96],
            "dtype": "float32",
            "frame": "robot proprioception",
            "units": "gravity/twist/rad/rad/s/action",
            "normalization": "dataset mean/std",
            "semantic_meaning": "student-state conditioning for VAE decoder",
        },
        "student_action": {"shape": ["N", 29], "dtype": "float32", "frame": "robot_joint_order", "units": "normalized", "normalization": "none", "semantic_meaning": "action actually executed by student"},
        "teacher_action": {"shape": ["N", 29], "dtype": "float32", "frame": "robot_joint_order", "units": "normalized", "normalization": "none", "semantic_meaning": "teacher label for the student state"},
        "student_latent": {"shape": ["N", 32], "dtype": "float32", "frame": "VAE latent", "units": "standard normal", "normalization": "none", "semantic_meaning": "latent sampled/chosen by student"},
    }


def _validate_1d(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {arr.shape}")
    return arr


def validate_dagger_dataset(
    payload: dict[str, np.ndarray],
    *,
    schema_version: str = DAGGER_SCHEMA_VERSION,
) -> dict[str, np.ndarray]:
    """Validate arrays for an aggregated DAgger dataset."""
    missing = [key for key in DAGGER_REQUIRED_ARRAYS if key not in payload]
    if missing:
        raise ValueError(f"DAgger dataset missing required arrays: {missing}")
    first = np.asarray(payload["encoder_reference_input"])
    if first.ndim != 2:
        raise ValueError(f"encoder_reference_input must be [N,67], got {first.shape}")
    sample_count = first.shape[0]
    checked: dict[str, np.ndarray] = {}
    expected_dims = {
        "encoder_reference_input": 67,
        "decoder_proprio_input": 96,
        "student_mu": 32,
        "student_logvar": 32,
        "student_latent": 32,
        "student_action": 29,
        "teacher_action": 29,
        "joint_position": 29,
        "joint_velocity": 29,
        "previous_action": 29,
    }
    for key in DAGGER_REQUIRED_ARRAYS:
        arr = np.asarray(payload[key])
        if arr.shape[0] != sample_count:
            raise ValueError(f"{key} sample count {arr.shape[0]} != {sample_count}")
        if key in expected_dims and (arr.ndim != 2 or arr.shape[-1] != expected_dims[key]):
            raise ValueError(f"{key} must be [N,{expected_dims[key]}], got {arr.shape}")
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains NaN or Inf")
        checked[key] = arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.floating) else arr
    for key in ["motion_name", "environment_id", "episode_id", "step_index", "reference_frame_index", "frequency_hz", "termination_reason"]:
        if key in payload:
            checked[key] = _validate_1d(key, np.asarray(payload[key]), sample_count)
    if schema_version != DAGGER_SCHEMA_VERSION:
        raise ValueError(f"unsupported DAgger schema_version {schema_version!r}")
    return checked


def save_dagger_dataset(
    path: str | Path,
    payload: dict[str, np.ndarray],
    metadata: DAggerDatasetMetadata | None = None,
) -> dict[str, object]:
    """Save a validated DAgger dataset as NPZ plus JSON metadata."""
    meta = metadata or DAggerDatasetMetadata()
    checked = validate_dagger_dataset(payload, schema_version=meta.schema_version)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **checked, metadata_json=json.dumps(meta.to_dict(), sort_keys=True))
    return {"output": str(output), "schema_version": meta.schema_version, "sample_count": int(next(iter(checked.values())).shape[0])}


def load_dagger_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], DAggerDatasetMetadata]:
    """Load and validate a DAgger dataset."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        meta_raw = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
    meta_dict = json.loads(meta_raw) if meta_raw else {}
    metadata = DAggerDatasetMetadata(**{**asdict(DAggerDatasetMetadata()), **meta_dict})
    return validate_dagger_dataset(payload, schema_version=metadata.schema_version), metadata


def merge_dagger_rounds(paths: list[str | Path], output_path: str | Path) -> dict[str, object]:
    """Merge D0/D1/... DAgger rounds by concatenating sample arrays."""
    if not paths:
        raise ValueError("paths must be nonempty")
    rounds = [load_dagger_dataset(path) for path in paths]
    keys = [key for key in rounds[0][0] if key in rounds[0][0]]
    merged = {key: np.concatenate([payload[key] for payload, _ in rounds], axis=0) for key in keys}
    round_ids: list[str] = []
    for path, (_, metadata) in zip(paths, rounds, strict=True):
        round_ids.extend(metadata.round_ids or (Path(path).stem,))
    metadata = DAggerDatasetMetadata(round_ids=tuple(round_ids), frequency_hz=rounds[0][1].frequency_hz)
    return save_dagger_dataset(output_path, merged, metadata)
