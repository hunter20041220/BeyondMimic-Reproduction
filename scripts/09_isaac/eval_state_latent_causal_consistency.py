#!/usr/bin/env python3
"""Measure guided state-latent causal consistency inside live Isaac physics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
DEFAULT_LOOP_ISAAC_EXPERIENCE = (
    "/dev/shm/BeyondMimic_Official_Stage1_runtime/IsaacLab/apps/"
    "isaaclab.python.headless.loop_isaac.single_gpu.kit"
)


def _ensure_file(path: str | None, label: str) -> None:
    if path and not Path(path).is_file():
        raise SystemExit(f"{label} does not exist: {path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _snapshot(command: Any, robot: Any, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
    from beyondmimic_repro.adapters.isaac.live_rollout import _make_root_state

    return {
        "root_state": _make_root_state(robot).detach().clone(),
        "joint_pos": robot.data.joint_pos.detach().clone(),
        "joint_vel": robot.data.joint_vel.detach().clone(),
        "previous_action": previous_action.detach().clone(),
        "command_time_steps": command.time_steps.detach().clone(),
    }


def _restore(env: Any, command: Any, robot: Any, snap: dict[str, torch.Tensor]) -> torch.Tensor:
    env_ids = torch.arange(snap["root_state"].shape[0], dtype=torch.long, device=snap["root_state"].device)
    robot.write_root_state_to_sim(snap["root_state"], env_ids=env_ids)
    robot.write_joint_state_to_sim(snap["joint_pos"], snap["joint_vel"], env_ids=env_ids)
    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(snap["joint_pos"], env_ids=env_ids)
    if hasattr(robot, "set_joint_velocity_target"):
        robot.set_joint_velocity_target(torch.zeros_like(snap["joint_vel"]), env_ids=env_ids)
    if hasattr(command, "time_steps"):
        command.time_steps.copy_(snap["command_time_steps"])
    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0
    if hasattr(env.unwrapped.scene, "write_data_to_sim"):
        env.unwrapped.scene.write_data_to_sim()
    if hasattr(env.unwrapped.scene, "update"):
        env.unwrapped.scene.update(0.0)
    return snap["previous_action"].detach().clone()


def _stack_xy(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("empty metric list")
    return torch.stack(values, dim=1)


def _body_indices(command: Any) -> tuple[list[str], int | None, int | None]:
    names = [str(name) for name in command.cfg.body_names]
    left = names.index("left_ankle_roll_link") if "left_ankle_roll_link" in names else None
    right = names.index("right_ankle_roll_link") if "right_ankle_roll_link" in names else None
    return names, left, right


def _actual_foot_contact(command: Any, threshold_z: float = 0.08) -> torch.Tensor:
    _, left, right = _body_indices(command)
    feet: list[torch.Tensor] = []
    for index in (left, right):
        if index is None:
            feet.append(torch.zeros(command.time_steps.shape[0], dtype=torch.bool, device=command.time_steps.device))
        else:
            feet.append(command.robot_body_pos_w[:, index, 2] <= float(threshold_z))
    return torch.stack(feet, dim=-1)


def _actual_body_velocity_xy(command: Any, current_yaw_inv: torch.Tensor) -> torch.Tensor:
    body_velocity_w = command.robot_body_lin_vel_w.mean(dim=1)
    return torch.einsum("bij,bj->bi", current_yaw_inv, body_velocity_w)[:, :2]


def _root_displacement_xy_current_yaw(
    initial_root_pos_w: torch.Tensor,
    current_root_pos_w: torch.Tensor,
    current_yaw_inv: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bij,bj->bi", current_yaw_inv, current_root_pos_w - initial_root_pos_w)[:, :2]


def _unproject_hybrid(tokens: torch.Tensor, *, state_dim: int, projection_inverse: torch.Tensor | None) -> torch.Tensor:
    state = tokens[..., :state_dim]
    if projection_inverse is None:
        return state
    inverse = projection_inverse.to(device=state.device, dtype=state.dtype)
    return torch.matmul(state, inverse.T)


def _predicted_contact_and_body_velocity(
    tokens: torch.Tensor,
    *,
    state_dim: int,
    projection_inverse: torch.Tensor | None,
    current_root_height: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    left_foot_index: int | None,
    right_foot_index: int | None,
    threshold_z: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    hybrid = _unproject_hybrid(tokens, state_dim=state_dim, projection_inverse=projection_inverse)
    root_pos_rel = hybrid[..., 0:3]
    body_pos_local = hybrid[..., 15:57].reshape(*hybrid.shape[:-1], 14, 3)
    body_vel_local = hybrid[..., 57:99].reshape(*hybrid.shape[:-1], 14, 3)
    foot_contacts: list[torch.Tensor] = []
    for index in (left_foot_index, right_foot_index):
        if index is None:
            foot_contacts.append(torch.zeros(hybrid.shape[:2], dtype=torch.bool, device=hybrid.device))
        else:
            foot_z = current_root_height[:, None] + root_pos_rel[..., 2] + body_pos_local[..., index, 2]
            foot_contacts.append(foot_z <= float(threshold_z))
    body_velocity_xy = current_velocity_xy[:, None, :] + body_vel_local[..., :2].mean(dim=-2)
    return torch.stack(foot_contacts, dim=-1), body_velocity_xy


def _plan_once(
    *,
    diffusion: torch.nn.Module,
    diffusion_cfg: dict[str, Any],
    diffusion_metadata: dict[str, Any],
    vae: torch.nn.Module,
    command: Any,
    robot: Any,
    previous_action: torch.Tensor,
    raw_state_history: list[dict[str, torch.Tensor]],
    latent_history: list[torch.Tensor],
    config: Any,
    target_velocity_xy: tuple[float, float],
) -> dict[str, Any]:
    from beyondmimic_repro.adapters.isaac.live_rollout import (
        _build_actual_state,
        _build_paper_raw_state,
        _build_vae_tensors,
        _current_root_velocity_xy,
        _make_diffusion_state_adapter,
        _make_diffusion_token_normalizer,
        _predict_diffusion_tokens,
        _prepare_paper_diffusion_inputs,
    )
    from beyondmimic_repro.stage3.guidance.physical_velocity import (
        physical_root_velocity_xy,
        projection_pseudoinverse,
    )

    device = torch.device(robot.data.joint_pos.device)
    token_dim = int(diffusion_cfg["_token_dim"])
    latent_dim = int(vae.config.latent_dim)
    state_dim = token_dim - latent_dim
    seq_len = int(diffusion_cfg.get("sequence_length", 21))
    current_index = int(diffusion_cfg.get("past_steps", 4))
    diffusion_frequency_hz = float(diffusion_cfg.get("frequency_hz", config.frequency_hz))
    adapter = _make_diffusion_state_adapter(diffusion_metadata, state_dim=state_dim, device=device)
    normalizer = _make_diffusion_token_normalizer(diffusion_metadata, token_dim=token_dim, device=device)
    projection_inverse = projection_pseudoinverse(adapter.projection_matrix)
    tensors = _build_vae_tensors(command, robot, previous_action)
    actual_state = _build_actual_state(command, robot)
    raw_state = _build_paper_raw_state(command, robot)
    if adapter.representation == "actual_state":
        state = actual_state
        planning_history = []
        current_velocity_xy = torch.zeros(state.shape[0], 2, dtype=torch.float32, device=device)
    else:
        state, planning_history, current_velocity_xy = _prepare_paper_diffusion_inputs(
            adapter=adapter,
            raw_history=raw_state_history,
            latent_history=latent_history,
            current_raw=raw_state,
            latent_dim=latent_dim,
            current_index=current_index,
            seq_len=seq_len,
            device=device,
        )
    future_count = max(1, seq_len - current_index - 1)
    target_schedule = torch.tensor(target_velocity_xy, dtype=torch.float32, device=device).view(1, 2).repeat(
        future_count, 1
    )
    predicted, diagnostics = _predict_diffusion_tokens(
        diffusion=diffusion,
        history=planning_history,
        current_state=state,
        seq_len=seq_len,
        token_dim=token_dim,
        state_dim=state_dim,
        current_index=current_index,
        state_adapter=adapter,
        token_normalizer=normalizer,
        config=config,
        target_velocity_xy=target_velocity_xy,
        target_velocity_schedule_xy=target_schedule,
        current_velocity_xy=current_velocity_xy,
        projection_inverse=projection_inverse,
        waypoint_xy=torch.zeros(state.shape[0], 2, dtype=torch.float32, device=device),
        obstacle_xy=torch.zeros(state.shape[0], 2, dtype=torch.float32, device=device),
        inpaint_target_xy=torch.zeros(state.shape[0], 2, dtype=torch.float32, device=device),
        diffusion_frequency_hz=diffusion_frequency_hz,
        device=device,
    )
    velocity = physical_root_velocity_xy(
        predicted,
        state_dim=state_dim,
        velocity_slice=adapter.velocity_slice,
        velocity_is_relative=adapter.velocity_is_relative,
        current_velocity_xy=current_velocity_xy,
        projection_inverse=projection_inverse,
    )
    predicted_hybrid = _unproject_hybrid(predicted, state_dim=state_dim, projection_inverse=projection_inverse)
    latent_sequence = predicted[:, current_index:, state_dim:]
    first_action = vae.decode(latent_sequence[:, 0], tensors["decoder_proprio_input"])
    unguided = diagnostics.get("unguided_tokens")
    if isinstance(unguided, torch.Tensor):
        unguided_action = vae.decode(unguided[:, current_index, state_dim:], tensors["decoder_proprio_input"])
        action_diff_norm = torch.linalg.norm(first_action - unguided_action, dim=-1)
    else:
        action_diff_norm = torch.zeros(state.shape[0], dtype=torch.float32, device=device)
    return {
        "adapter": adapter,
        "normalizer_enabled": normalizer is not None,
        "projection_inverse": projection_inverse,
        "seq_len": seq_len,
        "current_index": current_index,
        "state_dim": state_dim,
        "token_dim": token_dim,
        "diffusion_frequency_hz": diffusion_frequency_hz,
        "predicted_tokens": predicted.detach(),
        "predicted_hybrid": predicted_hybrid.detach(),
        "predicted_velocity_xy": velocity.detach(),
        "latent_sequence": latent_sequence.detach(),
        "current_velocity_xy": current_velocity_xy.detach(),
        "current_root_height": raw_state["root_pos_w"][:, 2].detach(),
        "initial_root_pos_w": raw_state["root_pos_w"].detach(),
        "current_yaw_inv": __import__(
            "beyondmimic_repro.adapters.isaac.live_rollout",
            fromlist=["_root_yaw_inverse_from_quat"],
        )._root_yaw_inverse_from_quat(raw_state["root_quat_w"]).detach(),
        "first_action": first_action.detach(),
        "action_diff_norm": action_diff_norm.detach(),
        "diagnostics": {key: value.detach() for key, value in diagnostics.items() if isinstance(value, torch.Tensor)},
    }


def _run_causal_consistency(args: argparse.Namespace) -> dict[str, Any]:
    from beyondmimic_repro.adapters.isaac.live_rollout import (
        LiveRolloutConfig,
        _build_paper_raw_state,
        _build_vae_tensors,
        _current_root_velocity_xy,
        _decode_vae_action,
        _load_diffusion,
        _load_vae,
        _make_env,
        _physical_metrics,
        _root_yaw_inverse_from_quat,
        _step_env,
    )

    config = LiveRolloutConfig(
        task_name=args.task_name,
        num_envs=1,
        device=args.device,
        output_path=Path(args.output_dir) / "causal_consistency_placeholder.npz",
        steps=20 * int(args.frequency_hz),
        warmup_steps=0,
        frequency_hz=args.frequency_hz,
        disable_obs_noise=args.disable_obs_noise,
        disable_events=args.disable_events,
        motion_file=Path(args.motion_file),
        vae_checkpoint=Path(args.vae_checkpoint),
        diffusion_checkpoint=Path(args.unnormalized_checkpoint),
        diffusion_use_ema=args.diffusion_use_ema,
        diffusion_initial_noise_scale=args.diffusion_initial_noise_scale,
        guidance_mode="velocity",
        guidance_scale=args.guidance_scale,
        guidance_clip_norm=args.guidance_clip_norm,
        physical_velocity_guidance=True,
        velocity_schedule="walk_to_run",
        velocity_walk_seconds=4.0,
        velocity_ramp_seconds=7.0,
        walk_velocity_x=args.walk_velocity_x,
        walk_velocity_y=args.walk_velocity_y,
        run_velocity_x=args.run_velocity_x,
        run_velocity_y=args.run_velocity_y,
        physical_only_terminations=True,
        seed=args.seed,
    )
    env, _ = _make_env(config, Path(args.motion_file))
    started = time.time()
    try:
        vae = _load_vae(args.vae_checkpoint, env.unwrapped.device)
        diffusion_specs = [
            ("normalized_bridge", Path(args.normalized_checkpoint)),
            ("unnormalized_paper_aligned", Path(args.unnormalized_checkpoint)),
        ]
        loaded: list[tuple[str, torch.nn.Module, dict[str, Any], dict[str, Any]]] = []
        for name, checkpoint in diffusion_specs:
            model, cfg, token_dim, metadata = _load_diffusion(checkpoint, env.unwrapped.device, use_ema=args.diffusion_use_ema)
            cfg = dict(cfg)
            cfg["_token_dim"] = int(token_dim)
            loaded.append((name, model, cfg, metadata))

        _, _ = env.get_observations()
        command = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        device = torch.device(env.unwrapped.device)
        previous_action = torch.zeros(1, 29, dtype=torch.float32, device=device)
        raw_state_history: list[dict[str, torch.Tensor]] = []
        latent_history: list[torch.Tensor] = []

        for _ in range(args.history_steps):
            tensors = _build_vae_tensors(command, robot, previous_action)
            decoded = _decode_vae_action(vae, tensors, deterministic=True)
            raw_state_history.append(_build_paper_raw_state(command, robot))
            latent_history.append(decoded["student_latent"].detach())
            _, _, dones, _ = _step_env(env, decoded["student_action"])
            previous_action = decoded["student_action"].detach().clone()
            if dones is not None and bool(dones.to(device=device, dtype=torch.bool).any()):
                previous_action.zero_()

        snap = _snapshot(command, robot, previous_action)
        initial_raw = _build_paper_raw_state(command, robot)
        current_yaw_inv = _root_yaw_inverse_from_quat(initial_raw["root_quat_w"]).detach()
        _, left_foot_index, right_foot_index = _body_indices(command)
        physical_accepted = torch.ones(1, dtype=torch.bool, device=device)

        results: dict[str, Any] = {
            "status": "completed",
            "task_name": args.task_name,
            "motion_file": args.motion_file,
            "vae_checkpoint": args.vae_checkpoint,
            "normalized_checkpoint": args.normalized_checkpoint,
            "unnormalized_checkpoint": args.unnormalized_checkpoint,
            "frequency_hz": args.frequency_hz,
            "horizons": args.horizons,
            "target_velocity_xy": [args.run_velocity_x, args.run_velocity_y],
            "manifest": {
                "s_t": "Isaac state before executing action at the snapshot time.",
                "z_t": "Guided diffusion current latent at the same snapshot time; it is not mask-fixed.",
                "a_t": "Frozen VAE decoder output from guided z_t and latest Isaac proprioception at time t.",
                "s_t_plus_1": "Isaac state after executing decoded action for one real physics control step.",
                "multi_step_execution": (
                    "For horizon h, the predicted latent sequence z_t..z_{t+h-1} is executed open-loop; "
                    "each action is decoded with the current real Isaac proprioception before that step."
                ),
                "noisy_vs_clean": (
                    "This diagnostic executes clean guided decoder actions. OU noisy executed actions are a dataset "
                    "collection semantic and are not added here."
                ),
            },
            "models": {},
        }

        for name, diffusion, diffusion_cfg, metadata in loaded:
            previous_action_model = _restore(env, command, robot, snap)
            plan = _plan_once(
                diffusion=diffusion,
                diffusion_cfg=diffusion_cfg,
                diffusion_metadata=metadata,
                vae=vae,
                command=command,
                robot=robot,
                previous_action=previous_action_model,
                raw_state_history=raw_state_history,
                latent_history=latent_history,
                config=config,
                target_velocity_xy=(args.run_velocity_x, args.run_velocity_y),
            )
            predicted_contact, predicted_body_velocity_xy = _predicted_contact_and_body_velocity(
                plan["predicted_tokens"],
                state_dim=plan["state_dim"],
                projection_inverse=plan["projection_inverse"],
                current_root_height=plan["current_root_height"],
                current_velocity_xy=plan["current_velocity_xy"],
                left_foot_index=left_foot_index,
                right_foot_index=right_foot_index,
            )
            model_result: dict[str, Any] = {
                "checkpoint": str(dict(diffusion_specs)[name]),
                "state_representation": plan["adapter"].representation,
                "state_schema": plan["adapter"].state_schema,
                "state_dim": plan["state_dim"],
                "sequence_length": plan["seq_len"],
                "current_index": plan["current_index"],
                "diffusion_frequency_hz": plan["diffusion_frequency_hz"],
                "normalization_enabled": bool(plan["normalizer_enabled"]),
                "guided_current_latent_norm": float(torch.linalg.norm(plan["latent_sequence"][:, 0], dim=-1).mean().item()),
                "decoded_action_norm": float(torch.linalg.norm(plan["first_action"], dim=-1).mean().item()),
                "decoded_action_diff_norm": float(plan["action_diff_norm"].mean().item()),
                "guidance_diagnostics": {
                    key: float(value.reshape(1, -1).mean().item())
                    for key, value in plan["diagnostics"].items()
                    if key != "unguided_tokens"
                },
                "horizons": {},
            }
            for horizon in args.horizons:
                if horizon <= 0:
                    continue
                previous_action_h = _restore(env, command, robot, snap)
                actual_velocity_steps: list[torch.Tensor] = []
                actual_body_velocity_steps: list[torch.Tensor] = []
                actual_displacement_steps: list[torch.Tensor] = []
                actual_contact_steps: list[torch.Tensor] = []
                actual_root_height_steps: list[torch.Tensor] = []
                actual_physical_fall_steps: list[torch.Tensor] = []
                done_steps: list[torch.Tensor] = []
                for step in range(horizon):
                    tensors = _build_vae_tensors(command, robot, previous_action_h)
                    latent_index = min(step, plan["latent_sequence"].shape[1] - 1)
                    latent = plan["latent_sequence"][:, latent_index]
                    with torch.no_grad():
                        action = vae.decode(latent, tensors["decoder_proprio_input"])
                    _, _, dones, _ = _step_env(env, action)
                    previous_action_h = action.detach().clone()
                    raw = _build_paper_raw_state(command, robot)
                    physical, physical_accepted = _physical_metrics(env, command, robot, config, physical_accepted)
                    actual_velocity_steps.append(_current_root_velocity_xy(raw).detach())
                    actual_body_velocity_steps.append(_actual_body_velocity_xy(command, current_yaw_inv).detach())
                    actual_displacement_steps.append(
                        _root_displacement_xy_current_yaw(initial_raw["root_pos_w"], raw["root_pos_w"], current_yaw_inv).detach()
                    )
                    actual_contact_steps.append(_actual_foot_contact(command).detach())
                    actual_root_height_steps.append(physical["root_height"].detach())
                    actual_physical_fall_steps.append(physical["physical_fall"].detach())
                    if dones is None:
                        done_steps.append(torch.zeros(1, dtype=torch.bool, device=device))
                    else:
                        done_steps.append(dones.to(device=device, dtype=torch.bool).detach())
                token_index = min(plan["current_index"] + horizon, plan["predicted_tokens"].shape[1] - 1)
                actual_velocity = _stack_xy(actual_velocity_steps)[:, -1]
                actual_body_velocity = _stack_xy(actual_body_velocity_steps)[:, -1]
                actual_displacement = _stack_xy(actual_displacement_steps)[:, -1]
                actual_contact = torch.stack(actual_contact_steps, dim=1)[:, -1]
                predicted_velocity = plan["predicted_velocity_xy"][:, token_index]
                predicted_displacement = plan["predicted_hybrid"][:, token_index, 0:2]
                predicted_body_velocity = predicted_body_velocity_xy[:, token_index]
                predicted_contact_h = predicted_contact[:, token_index]
                actual_flight = ~actual_contact.any(dim=-1)
                predicted_flight = ~predicted_contact_h.any(dim=-1)
                model_result["horizons"][str(horizon)] = {
                    "predicted_vx": float(predicted_velocity[:, 0].mean().item()),
                    "predicted_vy": float(predicted_velocity[:, 1].mean().item()),
                    "actual_vx": float(actual_velocity[:, 0].mean().item()),
                    "actual_vy": float(actual_velocity[:, 1].mean().item()),
                    "velocity_xy_l2_error": float(torch.linalg.norm(predicted_velocity - actual_velocity, dim=-1).mean().item()),
                    "predicted_root_displacement_xy": predicted_displacement.mean(dim=0).detach().cpu().tolist(),
                    "actual_root_displacement_xy": actual_displacement.mean(dim=0).detach().cpu().tolist(),
                    "root_displacement_l2_error": float(
                        torch.linalg.norm(predicted_displacement - actual_displacement, dim=-1).mean().item()
                    ),
                    "predicted_body_velocity_xy": predicted_body_velocity.mean(dim=0).detach().cpu().tolist(),
                    "actual_body_velocity_xy": actual_body_velocity.mean(dim=0).detach().cpu().tolist(),
                    "body_velocity_l2_error": float(
                        torch.linalg.norm(predicted_body_velocity - actual_body_velocity, dim=-1).mean().item()
                    ),
                    "predicted_contact_left_right": predicted_contact_h.float().mean(dim=0).detach().cpu().tolist(),
                    "actual_contact_left_right": actual_contact.float().mean(dim=0).detach().cpu().tolist(),
                    "predicted_flight_phase": float(predicted_flight.float().mean().item()),
                    "actual_flight_phase": float(actual_flight.float().mean().item()),
                    "actual_root_height": float(torch.stack(actual_root_height_steps, dim=1)[:, -1].mean().item()),
                    "actual_physical_fall": bool(torch.stack(actual_physical_fall_steps, dim=1).any().item()),
                    "env_done": bool(torch.stack(done_steps, dim=1).any().item()),
                    "actual_velocity_increment_from_snapshot": float(
                        torch.linalg.norm(actual_velocity - plan["current_velocity_xy"], dim=-1).mean().item()
                    ),
                }
            results["models"][name] = model_result
        results["elapsed_s"] = time.time() - started
        return results
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="Tracking-Flat-G1-v0")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--normalized-checkpoint", required=True)
    parser.add_argument("--unnormalized-checkpoint", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--guidance-clip-norm", type=float, default=1.0)
    parser.add_argument("--diffusion-initial-noise-scale", type=float, default=1.0)
    parser.add_argument("--diffusion-use-ema", action="store_true", default=True)
    parser.add_argument("--disable-diffusion-ema", dest="diffusion_use_ema", action="store_false")
    parser.add_argument("--walk-velocity-x", type=float, default=0.468394304424024)
    parser.add_argument("--walk-velocity-y", type=float, default=0.0)
    parser.add_argument("--run-velocity-x", type=float, default=1.3162681813025874)
    parser.add_argument("--run-velocity-y", type=float, default=0.0)
    parser.add_argument("--disable-obs-noise", dest="disable_obs_noise", action="store_true", default=True)
    parser.add_argument("--enable-obs-noise", dest="disable_obs_noise", action="store_false")
    parser.add_argument("--disable-events", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260712)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    if not getattr(args, "experience", ""):
        args.experience = DEFAULT_LOOP_ISAAC_EXPERIENCE
    for label in ["vae_checkpoint", "normalized_checkpoint", "unnormalized_checkpoint", "motion_file"]:
        _ensure_file(getattr(args, label), f"--{label.replace('_', '-')}")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        summary = _run_causal_consistency(args)
        (output_dir / "state_latent_causal_consistency.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_jsonable(summary), sort_keys=True), flush=True)
        completed = True
        return 0
    except Exception as exc:  # noqa: BLE001
        error = {
            "status": "error",
            "entrypoint": "eval_state_latent_causal_consistency",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "state_latent_causal_consistency_error.json").write_text(
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
