"""Stage-2 VAE checkpoint format."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-2 checkpointing") from exc


REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "global_step",
    "config",
    "input_schema",
    "normalization",
    "dataset_manifest",
    "git_commit",
    "random_seed",
}


def current_git_commit(cwd: str | Path | None = None) -> str:
    """Return the current git commit or 'unknown' outside git."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_vae_checkpoint(
    *,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    input_schema: dict[str, Any],
    normalization: dict[str, Any],
    dataset_manifest: dict[str, Any],
    random_seed: int,
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Build a checkpoint payload with all required release fields."""
    payload = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "input_schema": input_schema,
        "normalization": normalization,
        "dataset_manifest": dataset_manifest,
        "git_commit": git_commit,
        "random_seed": int(random_seed),
    }
    validate_vae_checkpoint(payload)
    return payload


def validate_vae_checkpoint(payload: dict[str, Any]) -> None:
    """Validate required checkpoint keys."""
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
    if missing:
        raise ValueError(f"VAE checkpoint missing required keys: {missing}")


def save_vae_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a validated checkpoint."""
    validate_vae_checkpoint(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_vae_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a VAE checkpoint."""
    payload = torch.load(path, map_location="cpu")
    validate_vae_checkpoint(payload)
    return payload
