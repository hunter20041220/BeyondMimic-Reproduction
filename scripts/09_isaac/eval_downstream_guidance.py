#!/usr/bin/env python3
"""Run live Isaac downstream guided diffusion+VAE rollout."""

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
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--motion-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default="outputs/isaac")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--guidance-mode", choices=["velocity", "speed", "turn", "waypoint", "obstacle", "inpainting"], required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-clip-norm", type=float, default=1.0)
    parser.add_argument("--disable-diffusion-ema", dest="diffusion_use_ema", action="store_false", default=True)
    parser.add_argument("--diffusion-initial-noise-scale", type=float, default=1.0)
    parser.add_argument("--walk-velocity-x", type=float, default=0.4)
    parser.add_argument("--walk-velocity-y", type=float, default=0.0)
    parser.add_argument("--run-velocity-x", type=float, default=1.2)
    parser.add_argument("--run-velocity-y", type=float, default=0.0)
    parser.add_argument("--turn-rate-z", type=float, default=0.8)
    parser.add_argument("--waypoint-x", type=float, default=1.0)
    parser.add_argument("--waypoint-y", type=float, default=0.0)
    parser.add_argument("--absolute-waypoint", dest="waypoint_relative", action="store_false", default=True)
    parser.add_argument("--waypoint-weight", type=float, default=1.0)
    parser.add_argument("--obstacle-x", type=float, default=0.5)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--absolute-obstacle", dest="obstacle_relative", action="store_false", default=True)
    parser.add_argument("--obstacle-radius", type=float, default=0.25)
    parser.add_argument("--obstacle-delta", type=float, default=0.1)
    parser.add_argument("--obstacle-weight", type=float, default=1.0)
    parser.add_argument("--inpaint-x", type=float, default=0.6)
    parser.add_argument("--inpaint-y", type=float, default=0.0)
    parser.add_argument("--absolute-inpaint", dest="inpaint_relative", action="store_false", default=True)
    parser.add_argument("--inpaint-token-index", type=int, default=-1)
    parser.add_argument("--disable-obs-noise", dest="disable_obs_noise", action="store_true", default=True)
    parser.add_argument("--enable-obs-noise", dest="disable_obs_noise", action="store_false")
    parser.add_argument("--disable-events", action="store_true", default=False)
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
    for label in ["teacher_map", "vae_checkpoint", "diffusion_checkpoint", "motion_file"]:
        _ensure_file(getattr(args, label), f"--{label.replace('_', '-')}")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completed = False
    try:
        from beyondmimic_repro.adapters.isaac.live_rollout import LiveRolloutConfig, run_collect_diffusion_rollout

        config = LiveRolloutConfig(
            task_name=args.task_name,
            num_envs=args.num_envs,
            device=args.device,
            output_path=Path(args.output),
            steps=args.steps,
            frequency_hz=args.frequency_hz,
            disable_obs_noise=args.disable_obs_noise,
            disable_events=args.disable_events,
            motion_name=args.motion_name,
            motion_file=Path(args.motion_file) if args.motion_file else None,
            teacher_map=Path(args.teacher_map) if args.teacher_map else None,
            vae_checkpoint=Path(args.vae_checkpoint),
            diffusion_checkpoint=Path(args.diffusion_checkpoint),
            diffusion_use_ema=args.diffusion_use_ema,
            diffusion_initial_noise_scale=args.diffusion_initial_noise_scale,
            guidance_mode=args.guidance_mode,
            guidance_scale=args.guidance_scale,
            guidance_clip_norm=args.guidance_clip_norm,
            walk_velocity_x=args.walk_velocity_x,
            walk_velocity_y=args.walk_velocity_y,
            run_velocity_x=args.run_velocity_x,
            run_velocity_y=args.run_velocity_y,
            turn_rate_z=args.turn_rate_z,
            waypoint_x=args.waypoint_x,
            waypoint_y=args.waypoint_y,
            waypoint_relative=args.waypoint_relative,
            waypoint_weight=args.waypoint_weight,
            obstacle_x=args.obstacle_x,
            obstacle_y=args.obstacle_y,
            obstacle_relative=args.obstacle_relative,
            obstacle_radius=args.obstacle_radius,
            obstacle_delta=args.obstacle_delta,
            obstacle_weight=args.obstacle_weight,
            inpaint_x=args.inpaint_x,
            inpaint_y=args.inpaint_y,
            inpaint_relative=args.inpaint_relative,
            inpaint_token_index=args.inpaint_token_index,
            seed=args.seed,
        )
        summary = run_collect_diffusion_rollout(config)
        summary["validation_mode"] = f"{args.guidance_mode}_guided_diffusion_closed_loop"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"eval_{args.guidance_mode}_guidance_summary.json").write_text(
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
            "entrypoint": "eval_downstream_guidance",
            "guidance_mode": args.guidance_mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / f"eval_{args.guidance_mode}_guidance_error.json").write_text(
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
