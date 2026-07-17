"""VAE closed-loop rollout schema used to build Stage-3 data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


VAE_ROLLOUT_SCHEMA_VERSION = "stage2-vae-rollout-v1"


@dataclass(frozen=True)
class VAERolloutMetadata:
    """Metadata for accepted/rejected VAE student closed-loop episodes."""

    schema_version: str = VAE_ROLLOUT_SCHEMA_VERSION
    frequency_hz: float = 50.0
    source: str = "trained VAE student closed-loop rollout"
    isaac_validation_status: str = "requires Isaac Sim / Isaac Lab runtime validation"


def validate_vae_rollout(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the rollout source required by paper-faithful Stage-3."""
    required = [
        "actual_state",
        "latent",
        "clean_action",
        "executed_action",
        "accepted",
        "episode_id",
        "time_index",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"VAE rollout missing arrays: {missing}")
    state = np.asarray(payload["actual_state"], dtype=np.float32)
    latent = np.asarray(payload["latent"], dtype=np.float32)
    if state.ndim != 3 or latent.ndim != 3:
        raise ValueError(f"actual_state/latent must be [E,T,D], got {state.shape}, {latent.shape}")
    if state.shape[:2] != latent.shape[:2]:
        raise ValueError(f"actual_state/latent [E,T] mismatch: {state.shape}, {latent.shape}")
    for key in ["clean_action", "executed_action"]:
        arr = np.asarray(payload[key], dtype=np.float32)
        if arr.shape[:2] != state.shape[:2] or arr.shape[-1] != 29:
            raise ValueError(f"{key} must be [E,T,29], got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains NaN or Inf")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(latent)):
        raise ValueError("actual_state/latent contains NaN or Inf")
    return {key: np.asarray(value) for key, value in payload.items()}


def save_vae_rollout(path: str | Path, payload: dict[str, np.ndarray], metadata: VAERolloutMetadata | None = None) -> dict[str, object]:
    """Save a VAE rollout contract file."""
    checked = validate_vae_rollout(payload)
    meta = metadata or VAERolloutMetadata()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **checked, metadata_json=json.dumps(asdict(meta), sort_keys=True))
    return {"output": str(output), "schema_version": meta.schema_version, "episode_count": int(checked["actual_state"].shape[0])}


def load_vae_rollout(path: str | Path) -> tuple[dict[str, np.ndarray], VAERolloutMetadata]:
    """Load and validate VAE rollout data."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        meta_raw = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
    metadata = VAERolloutMetadata(**{**asdict(VAERolloutMetadata()), **(json.loads(meta_raw) if meta_raw else {})})
    return validate_vae_rollout(payload), metadata
