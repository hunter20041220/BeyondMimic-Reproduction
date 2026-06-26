"""State-latent trajectory window schema and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from beyondmimic_repro.validation import ensure_finite


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class StateLatentWindow:
    """One state-latent trajectory window with tokens ``[T, state_dim + latent_dim]``."""

    sample_id: str
    source_motion: str
    start_timestep: int
    split: SplitName
    accepted: bool
    states: np.ndarray
    latents: np.ndarray


def stack_state_latent_tokens(states: np.ndarray, latents: np.ndarray) -> np.ndarray:
    """Concatenate finite state and latent trajectories along feature axis."""
    state_arr = ensure_finite("states", states)
    latent_arr = ensure_finite("latents", latents)
    if state_arr.ndim != 2 or latent_arr.ndim != 2:
        raise ValueError(f"states/latents must have shape [T,D], got {state_arr.shape}, {latent_arr.shape}")
    if state_arr.shape[0] != latent_arr.shape[0]:
        raise ValueError(f"state length {state_arr.shape[0]} != latent length {latent_arr.shape[0]}")
    return np.concatenate([state_arr, latent_arr], axis=1)


def build_state_latent_window(
    sample_id: str,
    source_motion: str,
    start_timestep: int,
    split: SplitName,
    accepted: bool,
    states: np.ndarray,
    latents: np.ndarray,
) -> StateLatentWindow:
    """Validate a finite state-latent window with state/latent shape ``[T,D]``."""
    if not sample_id or not source_motion:
        raise ValueError("sample_id and source_motion must be nonempty")
    if start_timestep < 0:
        raise ValueError("start_timestep must be nonnegative")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid split {split!r}")
    tokens = stack_state_latent_tokens(states, latents)
    if tokens.shape[0] < 2:
        raise ValueError("trajectory window must contain at least two timesteps")
    return StateLatentWindow(
        sample_id=sample_id,
        source_motion=source_motion,
        start_timestep=start_timestep,
        split=split,
        accepted=accepted,
        states=ensure_finite("states", states),
        latents=ensure_finite("latents", latents),
    )


def split_counts(windows: list[StateLatentWindow]) -> dict[str, int]:
    """Count accepted state-latent windows per train/validation/test split."""
    counts = {"train": 0, "validation": 0, "test": 0}
    for window in windows:
        if window.accepted:
            counts[window.split] += 1
    return counts
