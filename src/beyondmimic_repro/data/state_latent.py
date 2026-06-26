"""State-latent dataset construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beyondmimic_repro.rollout.schema import load_teacher_rollout


def encode_actions_linear(actions: np.ndarray, latent_dim: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dependency-light action encoder used for fixture construction.

    The torch VAE trainer writes learned latents for real experiments. This
    deterministic projection keeps the repository smoke tests runnable without
    a GPU or pre-trained checkpoint.
    """
    action_arr = np.asarray(actions, dtype=np.float64)
    if action_arr.ndim != 3:
        raise ValueError(f"actions must be [N,T,A], got {action_arr.shape}")
    flat = action_arr.reshape(-1, action_arr.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((latent_dim, action_arr.shape[-1]), dtype=np.float64)
    rows = min(latent_dim, vh.shape[0])
    components[:rows] = vh[:rows]
    latents = centered @ components.T
    return latents.reshape(action_arr.shape[0], action_arr.shape[1], latent_dim).astype(np.float32), mean.squeeze(0), components


def build_state_latent_dataset(
    rollout_path: str | Path,
    output_path: str | Path,
    *,
    latent_dim: int = 32,
) -> dict[str, object]:
    """Build `tokens=[state, latent]` windows from teacher rollout actions."""
    states, actions, names = load_teacher_rollout(rollout_path)
    latents, action_mean, components = encode_actions_linear(actions, latent_dim=latent_dim)
    tokens = np.concatenate([states, latents], axis=-1).astype(np.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        tokens=tokens,
        states=states,
        latents=latents,
        names=names,
        action_mean=action_mean.astype(np.float32),
        linear_components=components.astype(np.float32),
    )
    return {
        "status": "ok",
        "output": str(output),
        "source_rollout": str(rollout_path),
        "window_count": int(tokens.shape[0]),
        "horizon": int(tokens.shape[1]),
        "state_dim": int(states.shape[-1]),
        "latent_dim": int(latents.shape[-1]),
        "token_dim": int(tokens.shape[-1]),
    }


def load_state_latent_tokens(path: str | Path) -> np.ndarray:
    """Load state-latent token windows."""
    with np.load(path, allow_pickle=False) as data:
        tokens = data["tokens"]
    if tokens.ndim != 3 or not np.all(np.isfinite(tokens)):
        raise ValueError(f"tokens must be finite [N,T,D], got {tokens.shape}")
    return tokens.astype(np.float32)
