"""Smoke checks for the four BeyondMimic downstream guidance tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to run downstream guidance task checks") from exc

from beyondmimic_repro.stage3.guidance.inpainting import inpainting_guidance_cost
from beyondmimic_repro.stage3.guidance.joystick import joystick_guidance_cost, turn_rate_guidance_cost
from beyondmimic_repro.stage3.guidance.obstacle import obstacle_guidance_cost
from beyondmimic_repro.stage3.guidance.waypoint import waypoint_guidance_cost


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trajectory_from_xy(xy: torch.Tensor, state_dim: int = 79, *, dt: float = 0.02) -> torch.Tensor:
    traj = torch.zeros((1, xy.shape[0], state_dim), dtype=torch.float32)
    traj[0, :, :2] = xy
    if xy.shape[0] > 1:
        velocity = torch.zeros_like(xy)
        velocity[1:] = (xy[1:] - xy[:-1]) / dt
        velocity[0] = velocity[1]
        traj[0, :, 9:11] = velocity
    return traj


def run_downstream_guidance_smoke(*, output: str | Path, horizon: int = 21, dt: float = 0.02) -> dict[str, Any]:
    if horizon < 3:
        raise ValueError("--horizon must be at least 3")
    steps = torch.arange(horizon, dtype=torch.float32)

    target_velocity = torch.tensor([0.6, 0.0], dtype=torch.float32)
    velocity_match_xy = torch.stack([steps * target_velocity[0] * dt, steps * 0.0], dim=-1)
    velocity_bad_xy = torch.stack([steps * -target_velocity[0] * dt, steps * 0.0], dim=-1)
    velocity_match = _trajectory_from_xy(velocity_match_xy, dt=dt)
    velocity_bad = _trajectory_from_xy(velocity_bad_xy, dt=dt)
    velocity_cost, _ = joystick_guidance_cost(velocity_match, {"target_velocity_xy": target_velocity, "dt": dt})
    velocity_bad_cost, _ = joystick_guidance_cost(velocity_bad, {"target_velocity_xy": target_velocity, "dt": dt})

    target_turn_rate = torch.tensor(0.8, dtype=torch.float32)
    turn_match = torch.zeros((1, horizon, 79), dtype=torch.float32)
    turn_bad = torch.zeros_like(turn_match)
    turn_match[:, :, 14] = target_turn_rate
    turn_bad[:, :, 14] = -target_turn_rate
    turn_cost, _ = turn_rate_guidance_cost(
        turn_match,
        {"target_turn_rate_z": target_turn_rate, "angular_velocity_z_index": 14},
    )
    turn_bad_cost, _ = turn_rate_guidance_cost(
        turn_bad,
        {"target_turn_rate_z": target_turn_rate, "angular_velocity_z_index": 14},
    )

    waypoint = torch.tensor([0.1, 0.025], dtype=torch.float32)
    approach = torch.linspace(0.0, 1.0, horizon)
    approach = torch.clamp(approach * 2.0, max=1.0)
    waypoint_match_xy = torch.stack([approach * waypoint[0], approach * waypoint[1]], dim=-1)
    waypoint_bad_xy = torch.stack([torch.linspace(0.0, -0.6, horizon), torch.linspace(0.0, -0.4, horizon)], dim=-1)
    waypoint_cost, _ = waypoint_guidance_cost(_trajectory_from_xy(waypoint_match_xy, dt=dt), {"waypoint_xy": waypoint})
    waypoint_bad_cost, _ = waypoint_guidance_cost(_trajectory_from_xy(waypoint_bad_xy, dt=dt), {"waypoint_xy": waypoint})

    inpaint_traj = _trajectory_from_xy(waypoint_match_xy, dt=dt)
    target = torch.zeros_like(inpaint_traj)
    mask = torch.zeros_like(inpaint_traj)
    key_indices = torch.tensor([horizon // 3, 2 * horizon // 3, horizon - 1], dtype=torch.long)
    target[:, key_indices, :2] = inpaint_traj[:, key_indices, :2]
    mask[:, key_indices, :2] = 1.0
    inpaint_cost, _ = inpainting_guidance_cost(inpaint_traj, {"target": target, "mask": mask})
    inpaint_bad = inpaint_traj.clone()
    inpaint_bad[:, key_indices, :2] += torch.tensor([0.4, -0.3], dtype=torch.float32)
    inpaint_bad_cost, _ = inpainting_guidance_cost(inpaint_bad, {"target": target, "mask": mask})

    obstacle = torch.tensor([0.5, 0.0], dtype=torch.float32)
    collision_xy = torch.stack([torch.linspace(0.0, 1.0, horizon), torch.zeros(horizon)], dim=-1)
    safe_xy = torch.stack([torch.linspace(0.0, 1.0, horizon), torch.full((horizon,), 0.8)], dim=-1)
    obstacle_cost, obstacle_diag = obstacle_guidance_cost(
        _trajectory_from_xy(safe_xy, dt=dt),
        {"obstacle_xy": obstacle, "radius": 0.2, "delta": 0.1},
    )
    obstacle_bad_cost, obstacle_bad_diag = obstacle_guidance_cost(
        _trajectory_from_xy(collision_xy, dt=dt),
        {"obstacle_xy": obstacle, "radius": 0.2, "delta": 0.1},
    )

    checks = {
        "joystick_velocity": bool(velocity_cost.item() < velocity_bad_cost.item()),
        "joystick_turn_rate": bool(turn_cost.item() < turn_bad_cost.item()),
        "waypoint_navigation": bool(waypoint_cost.item() < waypoint_bad_cost.item()),
        "motion_inpainting": bool(inpaint_cost.item() < inpaint_bad_cost.item()),
        "obstacle_avoidance": bool(obstacle_cost.item() < obstacle_bad_cost.item()),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "validation": "beyondmimic_four_downstream_guidance_cost_smoke",
        "paper_downstream_tasks": [
            "joystick velocity/yaw-rate command",
            "waypoint navigation",
            "motion inpainting",
            "obstacle avoidance",
        ],
        "horizon": int(horizon),
        "dt": float(dt),
        "checks": checks,
        "costs": {
            "joystick_velocity_match": float(velocity_cost.item()),
            "joystick_velocity_bad": float(velocity_bad_cost.item()),
            "joystick_turn_rate_match": float(turn_cost.item()),
            "joystick_turn_rate_bad": float(turn_bad_cost.item()),
            "waypoint_match": float(waypoint_cost.item()),
            "waypoint_bad": float(waypoint_bad_cost.item()),
            "inpainting_match": float(inpaint_cost.item()),
            "inpainting_bad": float(inpaint_bad_cost.item()),
            "obstacle_safe": float(obstacle_cost.item()),
            "obstacle_collision": float(obstacle_bad_cost.item()),
        },
        "diagnostics": {
            "obstacle_safe_min_sdf": float(obstacle_diag["min_sdf"].item()),
            "obstacle_collision_min_sdf": float(obstacle_bad_diag["min_sdf"].item()),
        },
        "scope_note": "This validates task cost semantics on synthetic trajectories; closed-loop robot performance still requires Isaac/MuJoCo rollout per task.",
    }
    _write_json(Path(output), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate four BeyondMimic downstream guidance task costs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args(argv)
    summary = run_downstream_guidance_smoke(output=args.output, horizon=args.horizon, dt=args.dt)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
