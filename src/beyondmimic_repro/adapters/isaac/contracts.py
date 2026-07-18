"""Isaac runtime contracts and import boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ISAAC_UNAVAILABLE_MESSAGE = "Isaac Sim runtime is unavailable. Run this entrypoint from an Isaac Lab environment."


@dataclass(frozen=True)
class IsaacRunConfig:
    """Common CLI options for Isaac Lab entrypoints."""

    task_name: str = "Tracking-Flat-G1-v0"
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    teacher_map: Path | None = None
    vae_checkpoint: Path | None = None
    diffusion_checkpoint: Path | None = None
    output_dir: Path = Path("outputs/isaac")


def require_isaac_app_launcher() -> Any:
    """Import AppLauncher only at runtime, never at package import time."""
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(ISAAC_UNAVAILABLE_MESSAGE) from exc
    return AppLauncher


def launch_isaac_app(config: IsaacRunConfig) -> Any:
    """Launch Isaac before importing tasks."""
    app_launcher = require_isaac_app_launcher()
    launcher = app_launcher({"headless": config.headless, "device": config.device})
    return launcher.app
