#!/usr/bin/env python3
"""Run a live Isaac closed-loop validation rollout for a VAE checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
DEFAULT_LOOP_ISAAC_EXPERIENCE = (
    "/dev/shm/BeyondMimic_Official_Stage1_runtime/IsaacLab/apps/"
    "isaaclab.python.headless.loop_isaac.single_gpu.kit"
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-name", default="Tracking-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--teacher-map")
    parser.add_argument("--motion-name")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--motion-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default="outputs/isaac")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
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
    for label in ["teacher_map", "vae_checkpoint", "motion_file"]:
        _ensure_file(getattr(args, label), f"--{label.replace('_', '-')}")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completed = False
    try:
        from beyondmimic_repro.adapters.isaac.live_rollout import LiveRolloutConfig, run_collect_vae_rollout

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
            physical_only_terminations=args.physical_only_terminations,
            deterministic=args.deterministic,
            motion_name=args.motion_name,
            motion_file=Path(args.motion_file) if args.motion_file else None,
            teacher_map=Path(args.teacher_map) if args.teacher_map else None,
            vae_checkpoint=Path(args.vae_checkpoint),
            seed=args.seed,
        )
        summary = run_collect_vae_rollout(config)
        summary["validation_mode"] = "vae_closed_loop"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_vae_closed_loop_summary.json").write_text(
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
            "entrypoint": "eval_vae_closed_loop",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "eval_vae_closed_loop_error.json").write_text(
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
