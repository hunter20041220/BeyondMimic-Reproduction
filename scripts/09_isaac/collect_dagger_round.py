#!/usr/bin/env python3
"""Collect a live Isaac DAgger shard.

The script follows IsaacLab's import order: launch AppLauncher before importing
torch-heavy project runtime or any Isaac task modules.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))
DEFAULT_LOOP_ISAAC_EXPERIENCE = (
    "/dev/shm/BeyondMimic_Official_Stage1_runtime/IsaacLab/apps/"
    "isaaclab.python.headless.loop_isaac.single_gpu.kit"
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-name", default="Tracking-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--teacher-map")
    parser.add_argument("--motion-name")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--agent-config")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--motion-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default="outputs/isaac")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--round-id", default="D1")
    parser.add_argument("--disable-obs-noise", dest="disable_obs_noise", action="store_true", default=True)
    parser.add_argument("--enable-obs-noise", dest="disable_obs_noise", action="store_false")
    parser.add_argument("--disable-events", action="store_true", default=False)
    parser.add_argument("--physical-only-terminations", action="store_true", default=False)
    parser.add_argument("--stochastic-vae", dest="deterministic", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=20260712)


def _ensure_file(path: str | None, label: str) -> None:
    if path and not Path(path).is_file():
        raise SystemExit(f"{label} does not exist: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    if not getattr(args, "experience", ""):
        args.experience = DEFAULT_LOOP_ISAAC_EXPERIENCE
    for label in ["teacher_map", "teacher_checkpoint", "agent_config", "vae_checkpoint", "motion_file"]:
        _ensure_file(getattr(args, label), f"--{label.replace('_', '-')}")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completed = False
    try:
        from beyondmimic_repro.adapters.isaac.live_rollout import LiveRolloutConfig, run_collect_dagger_round

        config = LiveRolloutConfig(
            task_name=args.task_name,
            num_envs=args.num_envs,
            device=args.device,
            output_path=Path(args.output),
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            frequency_hz=args.frequency_hz,
            disable_obs_noise=args.disable_obs_noise,
            disable_events=args.disable_events,
            deterministic=args.deterministic,
            physical_only_terminations=args.physical_only_terminations,
            motion_name=args.motion_name,
            motion_file=Path(args.motion_file) if args.motion_file else None,
            teacher_map=Path(args.teacher_map) if args.teacher_map else None,
            teacher_checkpoint=Path(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            agent_config=Path(args.agent_config) if args.agent_config else None,
            vae_checkpoint=Path(args.vae_checkpoint),
            round_id=args.round_id,
            seed=args.seed,
        )
        summary = run_collect_dagger_round(config)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "collect_dagger_round_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        completed = True
        return 0
    except Exception as exc:  # noqa: BLE001
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "status": "error",
            "entrypoint": "collect_dagger_round",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "collect_dagger_round_error.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(error, sort_keys=True), flush=True)
        raise
    finally:
        if completed:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
