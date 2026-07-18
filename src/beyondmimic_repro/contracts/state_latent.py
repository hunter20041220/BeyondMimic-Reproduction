"""State-latent trajectory dataset contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


STATE_LATENT_SCHEMA_VERSION = "stage3-state-latent-v1"


@dataclass(frozen=True)
class TimeWindowConfig:
    """Explicit paper window semantics."""

    past_steps: int = 4
    include_current: bool = True
    future_steps: int = 16
    frequency_hz: float = 50.0

    @property
    def sequence_length(self) -> int:
        return self.past_steps + int(self.include_current) + self.future_steps

    @property
    def dt(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def future_physical_horizon_s(self) -> float:
        return self.future_steps / self.frequency_hz


@dataclass(frozen=True)
class StateLatentMetadata:
    """Metadata that prevents teacher-rollout encodings from posing as final S3 data."""

    schema_version: str = STATE_LATENT_SCHEMA_VERSION
    source: str = "vae_closed_loop_rollout"
    frequency_hz: float = 50.0
    past_steps: int = 4
    include_current: bool = True
    future_steps: int = 16
    state_schema: str = "character-yaw-centric"
    latent_source: str = "VAE latent from executed student rollout"

    @property
    def sequence_length(self) -> int:
        return self.past_steps + int(self.include_current) + self.future_steps


def validate_state_latent_dataset(
    payload: dict[str, np.ndarray],
    *,
    metadata: StateLatentMetadata | None = None,
    allow_legacy_teacher_source: bool = False,
) -> dict[str, np.ndarray]:
    """Validate Stage-3 state/latent windows."""
    meta = metadata or StateLatentMetadata()
    if meta.source != "vae_closed_loop_rollout" and not allow_legacy_teacher_source:
        raise ValueError("paper-faithful Stage-3 state-latent data must come from VAE closed-loop rollout")
    required = ["states", "latents", "tokens", "valid_mask", "episode_id", "motion_id", "time_index", "frequency_hz"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"state-latent dataset missing arrays: {missing}")
    states = np.asarray(payload["states"], dtype=np.float32)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    tokens = np.asarray(payload["tokens"], dtype=np.float32)
    if states.ndim != 3 or latents.ndim != 3 or tokens.ndim != 3:
        raise ValueError(f"states/latents/tokens must be [N,T,D], got {states.shape}, {latents.shape}, {tokens.shape}")
    if states.shape[:2] != latents.shape[:2] or states.shape[:2] != tokens.shape[:2]:
        raise ValueError("states, latents, and tokens must share [N,T]")
    if tokens.shape[-1] != states.shape[-1] + latents.shape[-1]:
        raise ValueError("tokens last dimension must equal state_dim + latent_dim")
    if states.shape[1] != meta.sequence_length:
        raise ValueError(f"sequence_length must be {meta.sequence_length}, got {states.shape[1]}")
    for key in ["states", "latents", "tokens"]:
        if not np.all(np.isfinite(np.asarray(payload[key]))):
            raise ValueError(f"{key} contains NaN or Inf")
    return {key: np.asarray(value) for key, value in payload.items()}


def save_state_latent_dataset(
    path: str | Path,
    payload: dict[str, np.ndarray],
    metadata: StateLatentMetadata | None = None,
    *,
    compressed: bool = True,
) -> dict[str, object]:
    """Save a Stage-3 state-latent dataset."""
    meta = metadata or StateLatentMetadata()
    checked = validate_state_latent_dataset(payload, metadata=meta)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(output, **checked, metadata_json=json.dumps(asdict(meta), sort_keys=True))
    return {
        "output": str(output),
        "schema_version": meta.schema_version,
        "window_count": int(checked["tokens"].shape[0]),
        "compressed": bool(compressed),
    }


def load_state_latent_dataset(path: str | Path, *, allow_legacy_teacher_source: bool = False) -> tuple[dict[str, np.ndarray], StateLatentMetadata]:
    """Load and validate Stage-3 state-latent data."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        meta_raw = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
    metadata = StateLatentMetadata(**{**asdict(StateLatentMetadata()), **(json.loads(meta_raw) if meta_raw else {})})
    return validate_state_latent_dataset(payload, metadata=metadata, allow_legacy_teacher_source=allow_legacy_teacher_source), metadata
