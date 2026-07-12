"""Build Stage-3 state-latent data from VAE closed-loop rollout."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from beyondmimic_repro.contracts.state_latent import StateLatentMetadata, save_state_latent_dataset
from beyondmimic_repro.contracts.vae_rollout import load_vae_rollout
from beyondmimic_repro.stage3.datasets.windowing import build_contiguous_windows


LEGACY_TEACHER_WARNING = "This path is not the paper-faithful Stage-3 dataset path."


def build_from_vae_rollout(
    vae_rollout_path: str | Path,
    output_path: str | Path,
    *,
    metadata: StateLatentMetadata | None = None,
) -> dict[str, object]:
    """Build actual VAE rollout state + latent windows."""
    payload, rollout_meta = load_vae_rollout(vae_rollout_path)
    meta = metadata or StateLatentMetadata(frequency_hz=rollout_meta.frequency_hz)
    states = build_contiguous_windows(payload["actual_state"], meta.sequence_length).astype(np.float32)
    latents = build_contiguous_windows(payload["latent"], meta.sequence_length).astype(np.float32)
    tokens = np.concatenate([states, latents], axis=-1).astype(np.float32)
    window_count = tokens.shape[0]
    out_payload = {
        "states": states,
        "latents": latents,
        "tokens": tokens,
        "valid_mask": np.ones((window_count, meta.sequence_length), dtype=np.bool_),
        "episode_id": np.arange(window_count, dtype=np.int32),
        "motion_id": np.zeros(window_count, dtype=np.int32),
        "time_index": np.zeros(window_count, dtype=np.int32),
        "frequency_hz": np.full(window_count, meta.frequency_hz, dtype=np.float32),
        "normalization_mean": tokens.mean(axis=(0, 1)) if window_count else np.zeros(tokens.shape[-1], dtype=np.float32),
        "normalization_std": tokens.std(axis=(0, 1)) if window_count else np.ones(tokens.shape[-1], dtype=np.float32),
    }
    return save_state_latent_dataset(output_path, out_payload, meta)


def build_state_latent_from_teacher_legacy(*_: object, **__: object) -> None:
    """Legacy guard for teacher-rollout direct encodings."""
    warnings.warn(LEGACY_TEACHER_WARNING, stacklevel=2)
    print(LEGACY_TEACHER_WARNING)
    raise RuntimeError(LEGACY_TEACHER_WARNING)
