"""Live Isaac rollouts for Stage-2 VAE and DAgger.

The module is imported only after Isaac's AppLauncher has started. It keeps
the runtime semantics explicit: the VAE student action is executed in Isaac,
and the teacher policy is queried only as a supervision label.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from beyondmimic_repro.contracts.dagger_dataset import DAggerDatasetMetadata, save_dagger_dataset
from beyondmimic_repro.contracts.teacher_assets import TeacherAssets, load_teacher_map
from beyondmimic_repro.contracts.vae_rollout import VAERolloutMetadata, save_vae_rollout
from beyondmimic_repro.stage2.models.conditional_action_vae import PaperConditionalActionVAE, PaperVAEConfig
from beyondmimic_repro.stage2.rollout.ou_noise import OrnsteinUhlenbeckNoise
from beyondmimic_repro.stage3.guidance.physical_velocity import (
    diagnostic_guided_sample,
    physical_root_velocity_xy,
    projection_pseudoinverse,
    smooth_walk_run_walk_velocity,
)


DEFAULT_G1_USD_CACHE = Path(
    "/dev/shm/BeyondMimic_Official_Stage1_runtime/asset_cache/"
    "g1_cylinder_85a384c67c45e5c44ac752e3a949b910"
)


@dataclass(frozen=True)
class LiveRolloutConfig:
    task_name: str
    num_envs: int
    device: str
    output_path: Path
    steps: int = 32
    warmup_steps: int = 1
    frequency_hz: float = 50.0
    disable_obs_noise: bool = True
    disable_events: bool = False
    deterministic: bool = True
    motion_name: str | None = None
    motion_file: Path | None = None
    teacher_map: Path | None = None
    teacher_checkpoint: Path | None = None
    agent_config: Path | None = None
    vae_checkpoint: Path | None = None
    diffusion_checkpoint: Path | None = None
    diffusion_use_ema: bool = True
    diffusion_initial_noise_scale: float = 1.0
    round_id: str = "D1"
    ou_sigma: float = 0.0
    ou_theta: float = 0.8
    ou_mu: float = 0.0
    ou_noise_seconds: float | None = None
    guidance_mode: str | None = None
    guidance_scale: float = 0.0
    guidance_clip_norm: float = 1.0
    physical_velocity_guidance: bool = False
    smooth_velocity_ramp: bool = False
    velocity_schedule: str = "walk_run_walk"
    velocity_walk_seconds: float = 4.0
    velocity_ramp_seconds: float = 7.0
    walk_velocity_x: float = 0.4
    walk_velocity_y: float = 0.0
    run_velocity_x: float = 1.2
    run_velocity_y: float = 0.0
    turn_rate_z: float = 0.8
    waypoint_x: float = 1.0
    waypoint_y: float = 0.0
    waypoint_relative: bool = True
    waypoint_weight: float = 1.0
    obstacle_x: float = 0.5
    obstacle_y: float = 0.0
    obstacle_relative: bool = True
    obstacle_radius: float = 0.25
    obstacle_delta: float = 0.1
    obstacle_weight: float = 1.0
    inpaint_x: float = 0.6
    inpaint_y: float = 0.0
    inpaint_relative: bool = True
    inpaint_token_index: int = -1
    physical_min_root_height: float = 0.40
    physical_max_abs_tilt: float = 1.35
    illegal_contact_force_threshold: float = 1.0
    physical_only_terminations: bool = False
    seed: int = 20260712


@dataclass(frozen=True)
class DiffusionStateAdapter:
    representation: str
    state_schema: str
    state_dim: int
    position_slice: tuple[int, int]
    velocity_slice: tuple[int, int]
    angular_velocity_z_index: int | None = None
    velocity_is_relative: bool = False
    projection_matrix: torch.Tensor | None = None


@dataclass(frozen=True)
class DiffusionTokenNormalizer:
    mean: torch.Tensor
    std: torch.Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_time(records: list[np.ndarray]) -> np.ndarray:
    return np.stack(records, axis=0)


def _flatten_time_env(records: list[np.ndarray]) -> np.ndarray:
    arr = _stack_time(records)
    if arr.ndim < 2:
        raise ValueError(f"expected time/env array, got {arr.shape}")
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _select_teacher(config: LiveRolloutConfig) -> TeacherAssets | None:
    if config.teacher_map is None:
        return None
    teachers = load_teacher_map(config.teacher_map)
    if not teachers:
        raise ValueError(f"teacher map is empty: {config.teacher_map}")
    if config.motion_name:
        if config.motion_name not in teachers:
            raise ValueError(f"motion {config.motion_name!r} is not in teacher map {config.teacher_map}")
        return teachers[config.motion_name]
    return next(iter(teachers.values()))


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(quat)
    w, x, y, z = q.unbind(dim=-1)
    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    return torch.stack(
        [
            torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1),
            torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)], dim=-1),
            torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    matrix = _quat_to_matrix(quat)
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def _rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    matrix = _quat_to_matrix(quat)
    return torch.einsum("...ji,...j->...i", matrix, vec)


def _root_tensor(data: Any, root_name: str, body_name: str) -> torch.Tensor:
    value = getattr(data, root_name, None)
    if value is not None:
        return value
    return getattr(data, body_name)[:, 0]


def _quat_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = _quat_normalize(quat)
    w, x, y, z = q.unbind(dim=-1)
    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    one = torch.as_tensor(1.0, dtype=q.dtype, device=q.device)
    roll = torch.atan2(two * (w * x + y * z), one - two * (x * x + y * y))
    sin_pitch = torch.clamp(two * (w * y - z * x), min=-1.0, max=1.0)
    pitch = torch.asin(sin_pitch)
    return roll, pitch


def _contact_force_metrics(
    env: Any,
    num_envs: int,
    device: torch.device,
    illegal_contact_force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    force_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
    illegal_force_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
    illegal_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    try:
        sensor = env.unwrapped.scene["contact_forces"]
    except Exception:  # noqa: BLE001 - Isaac scene/sensor availability varies across tasks.
        return force_max, illegal_force_max, illegal_contact
    data = getattr(sensor, "data", None)
    forces = getattr(data, "net_forces_w", None)
    if forces is None:
        forces = getattr(data, "net_forces_w_history", None)
    if not isinstance(forces, torch.Tensor) or forces.shape[0] != num_envs:
        return force_max, illegal_force_max, illegal_contact
    norms = torch.linalg.norm(forces.to(device=device, dtype=torch.float32), dim=-1)
    while norms.ndim > 1:
        norms = norms.max(dim=-1).values
    force_max = norms

    body_names = getattr(sensor, "body_names", None)
    if body_names is None:
        body_names = getattr(getattr(sensor, "cfg", None), "body_names", None)
    if body_names is None:
        return force_max, illegal_force_max, illegal_contact
    allowed_contacts = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    }
    names = [str(name) for name in body_names]
    body_axis = forces.ndim - 2
    if len(names) != forces.shape[body_axis]:
        return force_max, illegal_force_max, illegal_contact
    illegal_body_mask = torch.as_tensor([name not in allowed_contacts for name in names], dtype=torch.bool, device=device)
    if not bool(illegal_body_mask.any()):
        return force_max, illegal_force_max, illegal_contact
    illegal_forces = forces.to(device=device, dtype=torch.float32).index_select(body_axis, illegal_body_mask.nonzero().flatten())
    illegal_norms = torch.linalg.norm(illegal_forces, dim=-1)
    while illegal_norms.ndim > 1:
        illegal_norms = illegal_norms.max(dim=-1).values
    illegal_force_max = illegal_norms
    threshold = torch.as_tensor(float(illegal_contact_force_threshold), dtype=illegal_norms.dtype, device=device)
    illegal_contact = illegal_norms > threshold
    return force_max, illegal_force_max, illegal_contact


def _physical_metrics(
    env: Any,
    command: Any,
    robot: Any,
    config: LiveRolloutConfig,
    physical_accepted: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    data = robot.data
    root_pos = _root_tensor(data, "root_pos_w", "body_pos_w")
    root_height = root_pos[:, 2]
    torso_quat = getattr(command, "robot_anchor_quat_w", _root_tensor(data, "root_quat_w", "body_quat_w"))
    torso_roll, torso_pitch = _quat_to_roll_pitch(torso_quat)
    contact_force_max, illegal_contact_force_max, illegal_contact = _contact_force_metrics(
        env,
        root_height.shape[0],
        root_height.device,
        config.illegal_contact_force_threshold,
    )
    fall = torch.logical_or(
        root_height < float(config.physical_min_root_height),
        torch.logical_or(
            torch.abs(torso_roll) > float(config.physical_max_abs_tilt),
            torch.abs(torso_pitch) > float(config.physical_max_abs_tilt),
        ),
    )
    physical_failure = torch.logical_or(fall, illegal_contact)
    physical_accepted = torch.logical_and(physical_accepted, ~physical_failure)
    return (
        {
            "root_height": root_height,
            "torso_roll": torso_roll,
            "torso_pitch": torso_pitch,
            "physical_fall": fall,
            "contact_force_max": contact_force_max,
            "illegal_contact_force_max": illegal_contact_force_max,
            "physical_illegal_contact": illegal_contact,
            "physical_accepted": physical_accepted,
        },
        physical_accepted,
    )


def _termination_reason_numpy(done: torch.Tensor, physical: dict[str, torch.Tensor]) -> np.ndarray:
    done_np = _to_numpy(done).astype(bool)
    fall_np = _to_numpy(physical["physical_fall"]).astype(bool)
    illegal_np = _to_numpy(physical["physical_illegal_contact"]).astype(bool)
    physical_ok_np = _to_numpy(physical["physical_accepted"]).astype(bool)
    reason = np.full(done_np.shape, "none", dtype="<U64")
    reason[done_np] = "tracking_or_time_limit"
    reason[done_np & ~physical_ok_np] = "physical_violation"
    reason[done_np & fall_np] = "physical_fall_height_or_tilt"
    reason[done_np & illegal_np] = "illegal_contact"
    reason[(~done_np) & ~physical_ok_np] = "physical_violation_no_env_done"
    return reason


def _foot_contact_metrics(command: Any, *, height_threshold: float = 0.08) -> dict[str, torch.Tensor]:
    num_envs = command.time_steps.shape[0]
    device = command.time_steps.device
    body_names = [str(name) for name in getattr(command.cfg, "body_names", [])]
    contacts: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []
    for foot_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        if foot_name in body_names:
            foot_height = command.robot_body_pos_w[:, body_names.index(foot_name), 2]
        else:
            foot_height = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
        heights.append(foot_height)
        contacts.append(torch.isfinite(foot_height) & (foot_height <= float(height_threshold)))
    contact = torch.stack(contacts, dim=-1)
    height = torch.stack(heights, dim=-1)
    return {
        "foot_contact": contact,
        "foot_height": height,
        "flight_phase_proxy": ~contact.any(dim=-1),
    }


def _make_root_state(robot: Any) -> torch.Tensor:
    data = robot.data
    return torch.cat(
        [
            _root_tensor(data, "root_pos_w", "body_pos_w"),
            _root_tensor(data, "root_quat_w", "body_quat_w"),
            _root_tensor(data, "root_lin_vel_w", "body_lin_vel_w"),
            _root_tensor(data, "root_ang_vel_w", "body_ang_vel_w"),
        ],
        dim=-1,
    )


def _build_vae_tensors(command: Any, robot: Any, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
    data = robot.data
    anchor_pos_error = command.anchor_pos_w - command.robot_anchor_pos_w
    anchor_rot_error = _quat_mul(command.anchor_quat_w, _quat_conjugate(command.robot_anchor_quat_w))
    encoder = torch.cat(
        [
            command.joint_pos,
            command.joint_vel,
            anchor_pos_error,
            _rot6d_from_quat(anchor_rot_error),
        ],
        dim=-1,
    )

    root_quat = _root_tensor(data, "root_quat_w", "body_quat_w")
    gravity_w = torch.zeros(root_quat.shape[0], 3, dtype=root_quat.dtype, device=root_quat.device)
    gravity_w[:, 2] = -1.0
    projected_gravity = _rotate_inverse(root_quat, gravity_w)
    root_lin_vel_b = _rotate_inverse(root_quat, _root_tensor(data, "root_lin_vel_w", "body_lin_vel_w"))
    root_ang_vel_b = _rotate_inverse(root_quat, _root_tensor(data, "root_ang_vel_w", "body_ang_vel_w"))
    proprio = torch.cat(
        [
            projected_gravity,
            root_lin_vel_b,
            root_ang_vel_b,
            data.joint_pos,
            data.joint_vel,
            previous_action,
        ],
        dim=-1,
    )
    if encoder.shape[-1] != 67 or proprio.shape[-1] != 96:
        raise ValueError(f"unexpected VAE input shapes: encoder={tuple(encoder.shape)}, proprio={tuple(proprio.shape)}")
    return {"encoder_reference_input": encoder, "decoder_proprio_input": proprio}


def _build_actual_state(command: Any, robot: Any) -> torch.Tensor:
    data = robot.data
    root_quat = _root_tensor(data, "root_quat_w", "body_quat_w")
    return torch.cat(
        [
            command.robot_anchor_pos_w,
            _rot6d_from_quat(command.robot_anchor_quat_w),
            command.robot_anchor_lin_vel_w,
            command.robot_anchor_ang_vel_w,
            data.joint_pos,
            data.joint_vel,
            _rot6d_from_quat(root_quat),
        ],
        dim=-1,
    )


def _build_paper_raw_state(command: Any, robot: Any) -> dict[str, torch.Tensor]:
    data = robot.data
    return {
        "root_pos_w": _root_tensor(data, "root_pos_w", "body_pos_w"),
        "root_quat_w": _root_tensor(data, "root_quat_w", "body_quat_w"),
        "root_lin_vel_w": _root_tensor(data, "root_lin_vel_w", "body_lin_vel_w"),
        "root_ang_vel_w": _root_tensor(data, "root_ang_vel_w", "body_ang_vel_w"),
        "body_pos_w": command.robot_body_pos_w,
        "body_lin_vel_w": command.robot_body_lin_vel_w,
    }


def _state_representation_from_schema(state_schema: str) -> str:
    if state_schema == "paper_projected_163d_yaw_centric":
        return "paper_projected"
    if state_schema == "paper_hybrid_99d_yaw_centric":
        return "paper_hybrid"
    return "actual_state"


def _read_dataset_state_metadata(dataset_path: str | Path | None) -> dict[str, Any]:
    if dataset_path is None:
        return {}
    path = Path(dataset_path)
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {}
        if "state_representation" in data.files:
            out["state_representation"] = str(data["state_representation"])
        if "state_schema" in data.files:
            out["state_schema"] = str(data["state_schema"])
        if "state_dim" in data.files:
            out["state_dim"] = int(data["state_dim"])
        if "state_projection_matrix" in data.files:
            out["state_projection_matrix"] = np.asarray(data["state_projection_matrix"], dtype=np.float32)
        if "normalization_mean" in data.files:
            out["normalization_mean"] = np.asarray(data["normalization_mean"], dtype=np.float32)
        if "normalization_std" in data.files:
            out["normalization_std"] = np.asarray(data["normalization_std"], dtype=np.float32)
        return out


def _make_diffusion_state_adapter(
    checkpoint_metadata: dict[str, Any],
    *,
    state_dim: int,
    device: torch.device,
) -> DiffusionStateAdapter:
    dataset_meta = checkpoint_metadata.get("dataset_metadata", {}) if isinstance(checkpoint_metadata, dict) else {}
    state_schema = str(dataset_meta.get("state_schema", "character-yaw-centric"))
    representation = _state_representation_from_schema(state_schema)
    dataset_info = _read_dataset_state_metadata(checkpoint_metadata.get("dataset_path"))
    if dataset_info.get("state_schema"):
        state_schema = str(dataset_info["state_schema"])
        representation = _state_representation_from_schema(state_schema)
    if dataset_info.get("state_representation"):
        representation = str(dataset_info["state_representation"])
    if dataset_info.get("state_dim") is not None and int(dataset_info["state_dim"]) != state_dim:
        raise ValueError(
            f"diffusion checkpoint token state_dim={state_dim} but dataset state_dim={dataset_info['state_dim']}"
        )

    if representation == "actual_state":
        return DiffusionStateAdapter(
            representation=representation,
            state_schema=state_schema,
            state_dim=state_dim,
            position_slice=(0, 2),
            velocity_slice=(9, 11),
            angular_velocity_z_index=14,
            velocity_is_relative=False,
        )
    if representation == "paper_hybrid":
        if state_dim != 99:
            raise ValueError(f"paper_hybrid diffusion state_dim must be 99, got {state_dim}")
        return DiffusionStateAdapter(
            representation=representation,
            state_schema=state_schema,
            state_dim=state_dim,
            position_slice=(0, 2),
            velocity_slice=(9, 11),
            angular_velocity_z_index=14,
            velocity_is_relative=True,
        )
    if representation == "paper_projected":
        projection_matrix = dataset_info.get("state_projection_matrix")
        if projection_matrix is None:
            raise ValueError(
                "paper_projected diffusion rollout requires state_projection_matrix in the training dataset"
            )
        projection = torch.as_tensor(projection_matrix, dtype=torch.float32, device=device)
        if projection.ndim != 2 or projection.shape[0] != state_dim or projection.shape[1] != 99:
            raise ValueError(
                "paper_projected projection matrix must have shape "
                f"({state_dim}, 99), got {tuple(projection.shape)}"
            )
        hybrid_offset = int(projection.shape[0] - projection.shape[1])
        return DiffusionStateAdapter(
            representation=representation,
            state_schema=state_schema,
            state_dim=state_dim,
            position_slice=(hybrid_offset, hybrid_offset + 2),
            velocity_slice=(hybrid_offset + 9, hybrid_offset + 11),
            angular_velocity_z_index=hybrid_offset + 14,
            velocity_is_relative=True,
            projection_matrix=projection,
        )
    raise ValueError(f"unsupported diffusion state representation {representation!r}")


def _make_diffusion_token_normalizer(
    checkpoint_metadata: dict[str, Any],
    *,
    token_dim: int,
    device: torch.device,
) -> DiffusionTokenNormalizer | None:
    if not bool(checkpoint_metadata.get("normalization_enabled", False)):
        return None
    mean = checkpoint_metadata.get("normalization_mean")
    std = checkpoint_metadata.get("normalization_std")
    if mean is None or std is None:
        dataset_info = _read_dataset_state_metadata(checkpoint_metadata.get("dataset_path"))
        mean = dataset_info.get("normalization_mean")
        std = dataset_info.get("normalization_std")
    if mean is None or std is None:
        raise ValueError("normalized diffusion checkpoint is missing normalization_mean/std")
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device).flatten()
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device).flatten().clamp_min(1.0e-6)
    if mean_tensor.numel() != token_dim or std_tensor.numel() != token_dim:
        raise ValueError(
            "diffusion normalization mean/std must match token_dim, got "
            f"{mean_tensor.numel()}, {std_tensor.numel()}, token_dim={token_dim}"
        )
    return DiffusionTokenNormalizer(mean=mean_tensor, std=std_tensor)


def _yaw_matrix_torch(yaws: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaws)
    s = torch.sin(yaws)
    mats = torch.zeros(yaws.shape + (3, 3), dtype=yaws.dtype, device=yaws.device)
    mats[..., 0, 0] = c
    mats[..., 0, 1] = -s
    mats[..., 1, 0] = s
    mats[..., 1, 1] = c
    mats[..., 2, 2] = 1.0
    return mats


def _matrix_to_rot6d_torch(rotations: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotations[..., :, 0], rotations[..., :, 1]], dim=-1)


def _root_yaw_inverse_from_quat(root_quat_w: torch.Tensor) -> torch.Tensor:
    root_rot = _quat_to_matrix(root_quat_w)
    root_yaw = torch.atan2(root_rot[..., 1, 0], root_rot[..., 0, 0])
    return _yaw_matrix_torch(-root_yaw)


def _current_root_velocity_xy(raw_state: dict[str, torch.Tensor]) -> torch.Tensor:
    yaw_inv = _root_yaw_inverse_from_quat(raw_state["root_quat_w"])
    local_velocity = torch.einsum("bij,bj->bi", yaw_inv, raw_state["root_lin_vel_w"])
    return local_velocity[:, :2]


def _stack_raw_state_window(raw_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([state[key] for state in raw_states], dim=1) for key in raw_states[0]}


def _zero_raw_state_mask(raw_state: dict[str, torch.Tensor], done_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        cloned = value.detach().clone()
        cloned[done_mask] = 0.0
        out[key] = cloned
    return out


def _build_paper_hybrid_state_torch(raw_window: dict[str, torch.Tensor], *, current_index: int) -> torch.Tensor:
    root_pos = raw_window["root_pos_w"]
    root_quat = raw_window["root_quat_w"]
    root_lin_vel = raw_window["root_lin_vel_w"]
    root_ang_vel = raw_window["root_ang_vel_w"]
    body_pos = raw_window["body_pos_w"]
    body_lin_vel = raw_window["body_lin_vel_w"]
    root_rot = _quat_to_matrix(root_quat)
    root_yaw = torch.atan2(root_rot[..., 1, 0], root_rot[..., 0, 0])
    current_yaw_inv = _yaw_matrix_torch(-root_yaw[:, current_index])

    root_pos_rel_current = torch.einsum(
        "bij,btj->bti",
        current_yaw_inv,
        root_pos - root_pos[:, current_index : current_index + 1],
    )
    root_rot_rel_current = torch.matmul(current_yaw_inv[:, None], root_rot)
    root_rot6d_rel_current = _matrix_to_rot6d_torch(root_rot_rel_current)
    root_lin_vel_rel_current = torch.einsum(
        "bij,btj->bti",
        current_yaw_inv,
        root_lin_vel - root_lin_vel[:, current_index : current_index + 1],
    )
    root_ang_vel_rel_current = torch.einsum("bij,btj->bti", current_yaw_inv, root_ang_vel)

    per_step_yaw_inv = _yaw_matrix_torch(-root_yaw)
    body_pos_local = torch.einsum("btij,btnj->btni", per_step_yaw_inv, body_pos - root_pos[:, :, None, :])
    body_lin_vel_local = torch.einsum(
        "btij,btnj->btni",
        per_step_yaw_inv,
        body_lin_vel - root_lin_vel[:, :, None, :],
    )
    return torch.cat(
        [
            root_pos_rel_current,
            root_rot6d_rel_current,
            root_lin_vel_rel_current,
            root_ang_vel_rel_current,
            body_pos_local.reshape(body_pos_local.shape[0], body_pos_local.shape[1], -1),
            body_lin_vel_local.reshape(body_lin_vel_local.shape[0], body_lin_vel_local.shape[1], -1),
        ],
        dim=-1,
    )


def _project_paper_state(state: torch.Tensor, adapter: DiffusionStateAdapter) -> torch.Tensor:
    if adapter.representation == "paper_projected":
        if adapter.projection_matrix is None:
            raise RuntimeError("paper_projected adapter missing projection_matrix")
        return torch.matmul(state, adapter.projection_matrix.T)
    return state


def _prepare_paper_diffusion_inputs(
    *,
    adapter: DiffusionStateAdapter,
    raw_history: list[dict[str, torch.Tensor]],
    latent_history: list[torch.Tensor],
    current_raw: dict[str, torch.Tensor],
    latent_dim: int,
    current_index: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    past_capacity = min(current_index, seq_len - 1)
    available = min(past_capacity, len(raw_history), len(latent_history))
    selected_raw = raw_history[-available:] if available else []
    selected_latents = latent_history[-available:] if available else []
    raw_window = _stack_raw_state_window(selected_raw + [current_raw])
    hybrid_state = _build_paper_hybrid_state_torch(raw_window, current_index=available)
    state_window = _project_paper_state(hybrid_state, adapter)
    batch_size = current_raw["root_pos_w"].shape[0]
    zero_token = torch.zeros(batch_size, adapter.state_dim + latent_dim, dtype=torch.float32, device=device)
    history_tokens = [zero_token.clone() for _ in range(past_capacity - available)]
    for idx, latent in enumerate(selected_latents):
        history_tokens.append(torch.cat([state_window[:, idx], latent.detach()], dim=-1))
    current_state = state_window[:, available]
    current_velocity_xy = _current_root_velocity_xy(current_raw)
    return current_state, history_tokens, current_velocity_xy


def _load_vae(path: str | Path, device: str | torch.device) -> PaperConditionalActionVAE:
    checkpoint = torch.load(path, map_location=device)
    cfg_dict = checkpoint.get("config", {})
    config = PaperVAEConfig(
        encoder_input_dim=int(cfg_dict.get("encoder_input_dim", 67)),
        decoder_proprio_dim=int(cfg_dict.get("decoder_proprio_dim", 96)),
        action_dim=int(cfg_dict.get("action_dim", 29)),
        latent_dim=int(cfg_dict.get("latent_dim", 32)),
        encoder_hidden_dims=tuple(int(v) for v in cfg_dict.get("encoder_hidden_dims", [2048, 1024, 512])),
        decoder_hidden_dims=tuple(int(v) for v in cfg_dict.get("decoder_hidden_dims", [2048, 1024, 512])),
        activation=str(cfg_dict.get("activation", "ELU")),
        learning_rate=float(cfg_dict.get("learning_rate", 5e-4)),
        kl_coefficient=float(cfg_dict.get("kl_coefficient", 0.01)),
        gradient_accumulation_steps=int(cfg_dict.get("gradient_accumulation_steps", 15)),
        joint_position_semantics=str(cfg_dict.get("joint_position_semantics", "relative_to_default")),
    )
    model = PaperConditionalActionVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_diffusion(
    path: str | Path,
    device: str | torch.device,
    *,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], int, dict[str, Any]]:
    from beyondmimic_repro.stage3.models.state_latent_transformer import StateLatentTransformer

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"diffusion checkpoint missing model_state_dict: {path}")
    token_dim = int(state_dict["input_proj.weight"].shape[1])
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    model = StateLatentTransformer(
        token_dim=token_dim,
        sequence_length=int(cfg.get("sequence_length", 21)),
        denoising_steps=int(diffusion_cfg.get("denoising_steps", 20)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        transformer_layers=int(model_cfg.get("transformer_layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    ema_state = checkpoint.get("ema_state_dict", {})
    shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
    if use_ema and shadow:
        model.load_state_dict({key: value.to(device) for key, value in shadow.items()})
    else:
        model.load_state_dict(state_dict)
    model.eval()
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return model, cfg, token_dim, metadata


def _make_env(config: LiveRolloutConfig, motion_file: Path):
    import gymnasium as gym
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(config.task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = config.num_envs
    env_cfg.commands.motion.motion_file = str(motion_file)
    if hasattr(env_cfg, "episode_length_s") and config.steps > 0 and config.frequency_hz > 0.0:
        requested_episode_s = float(config.steps) / float(config.frequency_hz) + 1.0
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), requested_episode_s)
    _pin_cached_g1_usd(env_cfg)
    if config.disable_obs_noise and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if config.disable_events and hasattr(env_cfg, "events"):
        for event_name in dir(env_cfg.events):
            if event_name.startswith("_"):
                continue
            event_cfg = getattr(env_cfg.events, event_name)
            if hasattr(event_cfg, "mode"):
                setattr(env_cfg.events, event_name, None)
    if config.physical_only_terminations and hasattr(env_cfg, "terminations"):
        for term_name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)
    env = gym.make(config.task_name, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return RslRlVecEnvWrapper(env), env_cfg


def _pin_cached_g1_usd(env_cfg: Any) -> None:
    """Reuse the verified G1 USD cache instead of converting URDF every rollout."""
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "asset_path"):
        return
    if "unitree_description/urdf/g1/main.urdf" not in str(spawn_cfg.asset_path):
        return
    cache_dir = Path(os.environ.get("BEYONDMIMIC_G1_USD_CACHE", str(DEFAULT_G1_USD_CACHE)))
    required = [
        cache_dir / ".asset_hash",
        cache_dir / "main.usd",
        cache_dir / "configuration" / "main_base.usd",
    ]
    if not all(path.is_file() for path in required):
        print(f"[BeyondMimic] G1 USD cache missing at {cache_dir}; falling back to URDF conversion", flush=True)
        return
    spawn_cfg.usd_dir = str(cache_dir)
    spawn_cfg.usd_file_name = "main.usd"
    print(f"[BeyondMimic] using cached G1 USD {cache_dir}", flush=True)


def _load_teacher_policy(env: Any, config: LiveRolloutConfig, teacher: TeacherAssets | None):
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = config.teacher_checkpoint or (teacher.checkpoint_path if teacher is not None else None)
    agent_config = config.agent_config or _infer_agent_config(checkpoint)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise ValueError(f"teacher checkpoint is required and must exist: {checkpoint}")
    if agent_config is None or not Path(agent_config).is_file():
        raise ValueError(f"teacher agent config is required and must exist: {agent_config}")
    runner_cfg = _load_yaml(agent_config)
    runner_cfg["device"] = config.device
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=config.device)
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=env.unwrapped.device), str(checkpoint), str(agent_config)


def _infer_agent_config(checkpoint: str | Path | None) -> Path | None:
    if checkpoint is None:
        return None
    params = Path(checkpoint).parent / "params" / "agent.yaml"
    return params if params.is_file() else None


def _step_env(env: Any, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, Any]:
    step_out = env.step(actions)
    if len(step_out) == 4:
        obs, rewards, dones, infos = step_out
        return obs, rewards, dones, infos
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, infos = step_out
        return obs, rewards, torch.logical_or(terminated, truncated), infos
    raise RuntimeError(f"Unexpected env.step return length: {len(step_out)}")


def _safe_metric(command: Any, name: str, fallback_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    value = command.metrics.get(name)
    if value is None:
        return torch.zeros(fallback_shape, dtype=torch.float32, device=device)
    return value


def _decode_vae_action(
    model: PaperConditionalActionVAE,
    tensors: dict[str, torch.Tensor],
    *,
    deterministic: bool,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        mu, logvar = model.encode(tensors["encoder_reference_input"])
        latent = mu if deterministic else model.reparameterize(mu, logvar)
        action = model.decode(latent, tensors["decoder_proprio_input"])
    return {"student_mu": mu, "student_logvar": logvar, "student_latent": latent, "student_action": action}


def run_collect_dagger_round(config: LiveRolloutConfig) -> dict[str, object]:
    teacher = _select_teacher(config)
    motion_file = config.motion_file or (teacher.motion_file if teacher is not None else None)
    if motion_file is None:
        raise ValueError("--motion-file or --teacher-map is required")
    if config.vae_checkpoint is None:
        raise ValueError("--vae-checkpoint is required for DAgger collection")

    print(f"[BeyondMimic] creating Isaac env task={config.task_name} num_envs={config.num_envs}", flush=True)
    env, env_cfg = _make_env(config, Path(motion_file))
    try:
        print("[BeyondMimic] loading teacher policy", flush=True)
        teacher_policy, teacher_checkpoint, agent_config = _load_teacher_policy(env, config, teacher)
        print(f"[BeyondMimic] loading VAE checkpoint {config.vae_checkpoint}", flush=True)
        vae = _load_vae(config.vae_checkpoint, env.unwrapped.device)
        obs, _ = env.get_observations()
        command = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        device = torch.device(env.unwrapped.device)
        previous_action = torch.zeros(config.num_envs, 29, dtype=torch.float32, device=device)

        with torch.inference_mode():
            for _ in range(config.warmup_steps):
                teacher_action = teacher_policy(obs)
                obs, _, dones, _ = _step_env(env, teacher_action)
                previous_action = teacher_action.detach()
                if dones is not None:
                    previous_action = previous_action.clone()
                    previous_action[dones.to(device=device, dtype=torch.bool)] = 0.0

        print(f"[BeyondMimic] collecting DAgger steps={config.steps}", flush=True)
        episode_id = torch.zeros(config.num_envs, dtype=torch.int32, device=device)
        records: dict[str, list[np.ndarray]] = {
            "encoder_reference_input": [],
            "decoder_proprio_input": [],
            "student_mu": [],
            "student_logvar": [],
            "student_latent": [],
            "student_action": [],
            "teacher_action": [],
            "policy_observation": [],
            "root_state": [],
            "joint_position": [],
            "joint_velocity": [],
            "previous_action": [],
            "reward": [],
            "done": [],
            "body_position_error": [],
            "body_orientation_error": [],
            "joint_position_error": [],
            "joint_velocity_error": [],
            "motion_name": [],
            "environment_id": [],
            "episode_id": [],
            "step_index": [],
            "reference_frame_index": [],
            "frequency_hz": [],
        }
        started = time.time()
        for step in range(config.steps):
            tensors = _build_vae_tensors(command, robot, previous_action)
            decoded = _decode_vae_action(vae, tensors, deterministic=config.deterministic)
            with torch.inference_mode():
                teacher_action = teacher_policy(obs)

            current_episode = episode_id.clone()
            records["encoder_reference_input"].append(_to_numpy(tensors["encoder_reference_input"]))
            records["decoder_proprio_input"].append(_to_numpy(tensors["decoder_proprio_input"]))
            for key in ["student_mu", "student_logvar", "student_latent", "student_action"]:
                records[key].append(_to_numpy(decoded[key]))
            records["teacher_action"].append(_to_numpy(teacher_action))
            records["policy_observation"].append(_to_numpy(obs))
            records["root_state"].append(_to_numpy(_make_root_state(robot)))
            records["joint_position"].append(_to_numpy(robot.data.joint_pos))
            records["joint_velocity"].append(_to_numpy(robot.data.joint_vel))
            records["previous_action"].append(_to_numpy(previous_action))
            records["body_position_error"].append(_to_numpy(_safe_metric(command, "error_body_pos", (config.num_envs,), device)))
            records["body_orientation_error"].append(_to_numpy(_safe_metric(command, "error_body_rot", (config.num_envs,), device)))
            records["joint_position_error"].append(_to_numpy(_safe_metric(command, "error_joint_pos", (config.num_envs,), device)))
            records["joint_velocity_error"].append(_to_numpy(_safe_metric(command, "error_joint_vel", (config.num_envs,), device)))
            records["motion_name"].append(
                np.full((config.num_envs,), teacher.motion_name if teacher else config.motion_name or "motion", dtype="<U128")
            )
            records["environment_id"].append(np.arange(config.num_envs, dtype=np.int32))
            records["episode_id"].append(_to_numpy(current_episode))
            records["step_index"].append(np.full((config.num_envs,), step, dtype=np.int32))
            records["reference_frame_index"].append(_to_numpy(command.time_steps.to(torch.int32)))
            records["frequency_hz"].append(np.full((config.num_envs,), config.frequency_hz, dtype=np.float32))

            obs, rewards, dones, _ = _step_env(env, decoded["student_action"])
            if rewards is None:
                rewards = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
            if dones is None:
                dones = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
            done_mask = dones.to(device=device, dtype=torch.bool)
            records["reward"].append(_to_numpy(rewards))
            records["done"].append(_to_numpy(done_mask))
            previous_action = decoded["student_action"].detach().clone()
            previous_action[done_mask] = 0.0
            episode_id = episode_id + done_mask.to(torch.int32)

        payload: dict[str, np.ndarray] = {key: _flatten_time_env(value) for key, value in records.items()}
        metadata = DAggerDatasetMetadata(frequency_hz=config.frequency_hz, round_ids=(config.round_id,))
        summary = save_dagger_dataset(config.output_path, payload, metadata)
        summary.update(
            {
                "status": "collected",
                "task_name": config.task_name,
                "motion_file": str(motion_file),
                "teacher_checkpoint": teacher_checkpoint,
                "agent_config": agent_config,
                "vae_checkpoint": str(config.vae_checkpoint),
                "steps": config.steps,
                "num_envs": config.num_envs,
                "done_rate": float(payload["done"].astype(bool).mean()) if payload["done"].size else 0.0,
                "reward_mean": float(payload["reward"].mean()) if payload["reward"].size else None,
                "elapsed_s": time.time() - started,
                "obs_corruption_enabled": bool(getattr(env_cfg.observations.policy, "enable_corruption", False)),
                "events_disabled": bool(config.disable_events),
                "physical_only_terminations": bool(config.physical_only_terminations),
            }
        )
        _write_summary(config.output_path, summary)
        print(f"[BeyondMimic] saved DAgger dataset {config.output_path}", flush=True)
        return summary
    finally:
        env.close()


def run_collect_vae_rollout(config: LiveRolloutConfig) -> dict[str, object]:
    teacher = _select_teacher(config)
    motion_file = config.motion_file or (teacher.motion_file if teacher is not None else None)
    if motion_file is None:
        raise ValueError("--motion-file or --teacher-map is required")
    if config.vae_checkpoint is None:
        raise ValueError("--vae-checkpoint is required for VAE rollout")

    print(f"[BeyondMimic] creating Isaac env task={config.task_name} num_envs={config.num_envs}", flush=True)
    env, env_cfg = _make_env(config, Path(motion_file))
    try:
        teacher_policy = None
        teacher_checkpoint = None
        agent_config = None
        if config.warmup_steps > 0 and teacher is not None:
            print(f"[BeyondMimic] loading teacher policy for warmup_steps={config.warmup_steps}", flush=True)
            teacher_policy, teacher_checkpoint, agent_config = _load_teacher_policy(env, config, teacher)
        print(f"[BeyondMimic] loading VAE checkpoint {config.vae_checkpoint}", flush=True)
        vae = _load_vae(config.vae_checkpoint, env.unwrapped.device)
        obs, _ = env.get_observations()
        command = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        device = torch.device(env.unwrapped.device)
        previous_action = torch.zeros(config.num_envs, 29, dtype=torch.float32, device=device)
        if teacher_policy is not None:
            with torch.inference_mode():
                for _ in range(config.warmup_steps):
                    teacher_action = teacher_policy(obs)
                    obs, _, dones, _ = _step_env(env, teacher_action)
                    previous_action = teacher_action.detach().clone()
                    if dones is not None:
                        previous_action[dones.to(device=device, dtype=torch.bool)] = 0.0
        del obs
        ou = None
        if config.ou_sigma > 0.0:
            ou = OrnsteinUhlenbeckNoise(
                theta=config.ou_theta,
                mu=config.ou_mu,
                sigma=config.ou_sigma,
                dt=1.0 / config.frequency_hz,
                seed=config.seed,
            )
            ou.reset(config.num_envs, 29, device=device)
        ou_noise_steps = config.steps
        if config.ou_noise_seconds is not None:
            ou_noise_steps = max(0, min(config.steps, int(round(float(config.ou_noise_seconds) * config.frequency_hz))))

        records: dict[str, list[np.ndarray]] = {
            "actual_state": [],
            "next_actual_state": [],
            "root_pos_w": [],
            "root_quat_w": [],
            "root_lin_vel_w": [],
            "root_ang_vel_w": [],
            "body_pos_w": [],
            "body_lin_vel_w": [],
            "encoder_reference_input": [],
            "decoder_proprio_input": [],
            "latent": [],
            "clean_action": [],
            "executed_action": [],
            "accepted": [],
            "episode_id": [],
            "time_index": [],
            "motion_name": [],
            "environment_id": [],
            "reference_frame_index": [],
            "frequency_hz": [],
            "reward": [],
            "done": [],
            "termination_reason": [],
            "ou_noise": [],
            "root_height": [],
            "torso_roll": [],
            "torso_pitch": [],
            "physical_fall": [],
            "contact_force_max": [],
            "illegal_contact_force_max": [],
            "physical_illegal_contact": [],
            "physical_accepted": [],
            "post_step_physical_fall": [],
            "post_step_physical_illegal_contact": [],
            "post_step_physical_accepted": [],
            "foot_contact": [],
            "foot_height": [],
            "flight_phase_proxy": [],
        }
        accepted = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        physical_accepted = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        episode_id = torch.arange(config.num_envs, dtype=torch.int32, device=device)
        started = time.time()
        print(f"[BeyondMimic] collecting VAE rollout steps={config.steps} ou_sigma={config.ou_sigma}", flush=True)
        for step in range(config.steps):
            tensors = _build_vae_tensors(command, robot, previous_action)
            decoded = _decode_vae_action(vae, tensors, deterministic=config.deterministic)
            clean_action = decoded["student_action"]
            if ou is None or step >= ou_noise_steps:
                noise = torch.zeros_like(clean_action)
            else:
                noise = ou.step()
            executed_action = clean_action + noise

            records["actual_state"].append(_to_numpy(_build_actual_state(command, robot)))
            paper_raw_state = _build_paper_raw_state(command, robot)
            physical, physical_accepted = _physical_metrics(env, command, robot, config, physical_accepted)
            foot_contact = _foot_contact_metrics(command)
            for key in ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pos_w", "body_lin_vel_w"]:
                records[key].append(_to_numpy(paper_raw_state[key]))
            for key in [
                "root_height",
                "torso_roll",
                "torso_pitch",
                "physical_fall",
                "contact_force_max",
                "illegal_contact_force_max",
                "physical_illegal_contact",
                "physical_accepted",
            ]:
                records[key].append(_to_numpy(physical[key]))
            for key in ["foot_contact", "foot_height", "flight_phase_proxy"]:
                records[key].append(_to_numpy(foot_contact[key]))
            records["encoder_reference_input"].append(_to_numpy(tensors["encoder_reference_input"]))
            records["decoder_proprio_input"].append(_to_numpy(tensors["decoder_proprio_input"]))
            records["latent"].append(_to_numpy(decoded["student_latent"]))
            records["clean_action"].append(_to_numpy(clean_action))
            records["executed_action"].append(_to_numpy(executed_action))
            records["episode_id"].append(_to_numpy(episode_id))
            records["time_index"].append(np.full((config.num_envs,), step, dtype=np.int32))
            records["motion_name"].append(
                np.full((config.num_envs,), teacher.motion_name if teacher else config.motion_name or "motion", dtype="<U128")
            )
            records["environment_id"].append(np.arange(config.num_envs, dtype=np.int32))
            records["reference_frame_index"].append(_to_numpy(command.time_steps.to(torch.int32)))
            records["frequency_hz"].append(np.full((config.num_envs,), config.frequency_hz, dtype=np.float32))
            records["ou_noise"].append(_to_numpy(noise))

            obs, rewards, dones, _ = _step_env(env, executed_action)
            del obs
            if rewards is None:
                rewards = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
            if dones is None:
                dones = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
            done_mask = dones.to(device=device, dtype=torch.bool)
            accepted = torch.logical_and(accepted, ~done_mask)
            next_actual_state = _build_actual_state(command, robot)
            next_physical, _ = _physical_metrics(env, command, robot, config, physical_accepted.clone())
            records["next_actual_state"].append(_to_numpy(next_actual_state))
            records["post_step_physical_fall"].append(_to_numpy(next_physical["physical_fall"]))
            records["post_step_physical_illegal_contact"].append(_to_numpy(next_physical["physical_illegal_contact"]))
            records["post_step_physical_accepted"].append(_to_numpy(next_physical["physical_accepted"]))
            records["termination_reason"].append(_termination_reason_numpy(done_mask, next_physical))
            records["accepted"].append(_to_numpy(accepted))
            records["reward"].append(_to_numpy(rewards))
            records["done"].append(_to_numpy(done_mask))
            previous_action = executed_action.detach().clone()
            previous_action[done_mask] = 0.0
            episode_id = episode_id + done_mask.to(torch.int32)
            if ou is not None:
                ou.reset_mask(done_mask)

        payload = {
            key: _stack_time(value).transpose(1, 0, *range(2, _stack_time(value).ndim))
            for key, value in records.items()
        }
        payload["body_names"] = np.asarray(command.cfg.body_names)
        metadata = VAERolloutMetadata(
            frequency_hz=config.frequency_hz,
            isaac_validation_status="validated by live Isaac rollout script",
        )
        summary = save_vae_rollout(config.output_path, payload, metadata)
        physical_acceptance = payload.get("post_step_physical_accepted", payload["physical_accepted"])
        summary.update(
            {
                "status": "collected",
                "task_name": config.task_name,
                "motion_file": str(motion_file),
                "warmup_steps": config.warmup_steps,
                "warmup_teacher_checkpoint": teacher_checkpoint,
                "warmup_agent_config": agent_config,
                "vae_checkpoint": str(config.vae_checkpoint),
                "steps": config.steps,
                "num_envs": config.num_envs,
                "accepted_rate": float(payload["accepted"][:, -1].astype(bool).mean()) if payload["accepted"].size else 0.0,
                "physical_accepted_rate": float(physical_acceptance[:, -1].astype(bool).mean())
                if physical_acceptance.size
                else 0.0,
                "physical_fall_rate": float(payload["physical_fall"].astype(bool).mean())
                if payload["physical_fall"].size
                else 0.0,
                "physical_illegal_contact_rate": float(payload["physical_illegal_contact"].astype(bool).mean())
                if payload["physical_illegal_contact"].size
                else 0.0,
                "flight_phase_proxy_rate": float(payload["flight_phase_proxy"].astype(bool).mean())
                if "flight_phase_proxy" in payload and payload["flight_phase_proxy"].size
                else 0.0,
                "root_height_min": float(payload["root_height"].min()) if payload["root_height"].size else None,
                "torso_abs_roll_max": float(np.abs(payload["torso_roll"]).max()) if payload["torso_roll"].size else None,
                "torso_abs_pitch_max": float(np.abs(payload["torso_pitch"]).max()) if payload["torso_pitch"].size else None,
                "illegal_contact_force_max": float(payload["illegal_contact_force_max"].max())
                if payload["illegal_contact_force_max"].size
                else None,
                "physical_min_root_height": config.physical_min_root_height,
                "physical_max_abs_tilt": config.physical_max_abs_tilt,
                "illegal_contact_force_threshold": config.illegal_contact_force_threshold,
                "done_rate": float(payload["done"].astype(bool).mean()) if payload["done"].size else 0.0,
                "reward_mean": float(payload["reward"].mean()) if payload["reward"].size else None,
                "ou_sigma": config.ou_sigma,
                "ou_noise_seconds": config.ou_noise_seconds,
                "ou_noise_steps": int(ou_noise_steps),
                "elapsed_s": time.time() - started,
                "obs_corruption_enabled": bool(getattr(env_cfg.observations.policy, "enable_corruption", False)),
                "events_disabled": bool(config.disable_events),
                "physical_only_terminations": bool(config.physical_only_terminations),
            }
        )
        _write_summary(config.output_path, summary)
        print(f"[BeyondMimic] saved VAE rollout {config.output_path}", flush=True)
        return summary
    finally:
        env.close()


def run_collect_diffusion_rollout(config: LiveRolloutConfig) -> dict[str, object]:
    teacher = _select_teacher(config)
    motion_file = config.motion_file or (teacher.motion_file if teacher is not None else None)
    if motion_file is None:
        raise ValueError("--motion-file or --teacher-map is required")
    if config.vae_checkpoint is None:
        raise ValueError("--vae-checkpoint is required for diffusion rollout")
    if config.diffusion_checkpoint is None:
        raise ValueError("--diffusion-checkpoint is required for diffusion rollout")

    print(f"[BeyondMimic] creating Isaac env task={config.task_name} num_envs={config.num_envs}", flush=True)
    env, env_cfg = _make_env(config, Path(motion_file))
    try:
        teacher_policy = None
        teacher_checkpoint = None
        agent_config = None
        print(f"[BeyondMimic] loading VAE checkpoint {config.vae_checkpoint}", flush=True)
        vae = _load_vae(config.vae_checkpoint, env.unwrapped.device)
        print(f"[BeyondMimic] loading diffusion checkpoint {config.diffusion_checkpoint}", flush=True)
        diffusion, diffusion_cfg, token_dim, diffusion_metadata = _load_diffusion(
            config.diffusion_checkpoint,
            env.unwrapped.device,
            use_ema=config.diffusion_use_ema,
        )
        if config.warmup_steps > 0 and teacher is not None:
            print(f"[BeyondMimic] loading teacher policy for warmup_steps={config.warmup_steps}", flush=True)
            teacher_policy, teacher_checkpoint, agent_config = _load_teacher_policy(env, config, teacher)
        diffusion_frequency_hz = float(diffusion_cfg.get("frequency_hz", config.frequency_hz))
        if diffusion_frequency_hz <= 0.0:
            raise ValueError(f"invalid diffusion frequency_hz={diffusion_frequency_hz}")
        frequency_ratio = float(config.frequency_hz) / diffusion_frequency_hz
        diffusion_control_stride = int(round(frequency_ratio))
        if diffusion_control_stride < 1 or abs(frequency_ratio - diffusion_control_stride) > 1.0e-5:
            raise ValueError(
                "Isaac control frequency must be an integer multiple of diffusion frequency: "
                f"control={config.frequency_hz} diffusion={diffusion_frequency_hz}"
            )
        seq_len = int(diffusion_cfg.get("sequence_length", 21))
        current_index = int(diffusion_cfg.get("past_steps", 4))
        latent_dim = int(vae.config.latent_dim)
        state_dim = token_dim - latent_dim
        if state_dim <= 0:
            raise ValueError(f"invalid diffusion token/state dims: token_dim={token_dim} latent_dim={latent_dim}")
        state_adapter = _make_diffusion_state_adapter(
            diffusion_metadata,
            state_dim=state_dim,
            device=torch.device(env.unwrapped.device),
        )
        projection_inverse = projection_pseudoinverse(state_adapter.projection_matrix)
        token_normalizer = _make_diffusion_token_normalizer(
            diffusion_metadata,
            token_dim=token_dim,
            device=torch.device(env.unwrapped.device),
        )
        print(
            "[BeyondMimic] diffusion state "
            f"representation={state_adapter.representation} schema={state_adapter.state_schema} "
            f"state_dim={state_adapter.state_dim}",
            flush=True,
        )
        if token_normalizer is not None:
            print("[BeyondMimic] diffusion token normalization enabled", flush=True)

        obs, _ = env.get_observations()
        command = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        device = torch.device(env.unwrapped.device)
        previous_action = torch.zeros(config.num_envs, 29, dtype=torch.float32, device=device)
        if teacher_policy is not None:
            with torch.inference_mode():
                for _ in range(config.warmup_steps):
                    teacher_action = teacher_policy(obs)
                    obs, _, dones, _ = _step_env(env, teacher_action)
                    previous_action = teacher_action.detach().clone()
                    if dones is not None:
                        previous_action[dones.to(device=device, dtype=torch.bool)] = 0.0
        del obs
        token_history: list[torch.Tensor] = []
        raw_state_history: list[dict[str, torch.Tensor]] = []
        latent_history: list[torch.Tensor] = []

        records: dict[str, list[np.ndarray]] = {
            "actual_state": [],
            "diffusion_state": [],
            "latent": [],
            "clean_action": [],
            "executed_action": [],
            "accepted": [],
            "episode_id": [],
            "time_index": [],
            "reward": [],
            "done": [],
            "target_velocity_xy": [],
            "target_speed": [],
            "target_heading": [],
            "actual_velocity_xy_current_yaw": [],
            "actual_velocity_xy_world": [],
            "actual_speed_current_yaw": [],
            "actual_heading_current_yaw": [],
            "target_turn_rate_z": [],
            "target_waypoint_xy": [],
            "obstacle_xy": [],
            "inpaint_target_xy": [],
            "planned_next_velocity_xy": [],
            "planned_future_velocity_xy_mean": [],
            "planned_current_velocity_xy": [],
            "unguided_future_velocity_xy_mean": [],
            "guided_future_velocity_xy_mean": [],
            "unguided_future_speed_mean": [],
            "guided_future_speed_mean": [],
            "unguided_future_heading_mean": [],
            "guided_future_heading_mean": [],
            "current_latent_diff_norm": [],
            "current_latent_grad_norm": [],
            "decoded_action_diff_norm": [],
            "diffusion_planning_tick": [],
            "guidance_cost": [],
            "guidance_grad_norm": [],
            "root_height": [],
            "torso_roll": [],
            "torso_pitch": [],
            "physical_fall": [],
            "contact_force_max": [],
            "illegal_contact_force_max": [],
            "physical_illegal_contact": [],
            "physical_accepted": [],
            "foot_contact": [],
            "foot_height": [],
            "flight_phase_proxy": [],
        }
        accepted = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        physical_accepted = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        episode_id = torch.arange(config.num_envs, dtype=torch.int32, device=device)
        started = time.time()
        print(
            f"[BeyondMimic] collecting diffusion rollout steps={config.steps} guidance={config.guidance_mode or 'none'}",
            flush=True,
        )
        last_latent: torch.Tensor | None = None
        last_diagnostics: dict[str, torch.Tensor] = {}
        waypoint_xy: torch.Tensor | None = None
        obstacle_xy: torch.Tensor | None = None
        inpaint_target_xy: torch.Tensor | None = None
        zero_plan_xy = torch.zeros(config.num_envs, 2, dtype=torch.float32, device=device)
        last_plan_metrics: dict[str, torch.Tensor] = {
            "planned_next_velocity_xy": zero_plan_xy,
            "planned_future_velocity_xy_mean": zero_plan_xy,
            "planned_current_velocity_xy": zero_plan_xy,
        }
        for step in range(config.steps):
            tensors = _build_vae_tensors(command, robot, previous_action)
            actual_state = _build_actual_state(command, robot)
            paper_raw_state = _build_paper_raw_state(command, robot)
            physical, physical_accepted = _physical_metrics(env, command, robot, config, physical_accepted)
            foot_contact = _foot_contact_metrics(command)
            if state_adapter.representation == "actual_state":
                state = actual_state
                planning_history = token_history
                current_velocity_xy = torch.zeros(config.num_envs, 2, dtype=torch.float32, device=device)
                if state.shape[-1] != state_dim:
                    raise ValueError(f"actual state dim {state.shape[-1]} does not match diffusion state_dim {state_dim}")
            else:
                state, planning_history, current_velocity_xy = _prepare_paper_diffusion_inputs(
                    adapter=state_adapter,
                    raw_history=raw_state_history,
                    latent_history=latent_history,
                    current_raw=paper_raw_state,
                    latent_dim=latent_dim,
                    current_index=current_index,
                    seq_len=seq_len,
                    device=device,
                )
                if state.shape[-1] != state_dim:
                    raise ValueError(
                        f"{state_adapter.representation} state dim {state.shape[-1]} does not match diffusion state_dim {state_dim}"
                    )
            target_velocity = _target_velocity_for_step(config, step)
            actual_velocity_current_yaw = _current_root_velocity_xy(paper_raw_state)
            actual_velocity_world = paper_raw_state["root_lin_vel_w"][:, :2].detach()
            actual_speed_current_yaw = torch.linalg.norm(actual_velocity_current_yaw, dim=-1)
            actual_heading_current_yaw = torch.atan2(actual_velocity_current_yaw[:, 1], actual_velocity_current_yaw[:, 0])
            target_velocity_tensor = torch.tensor(target_velocity, dtype=torch.float32, device=device)
            target_speed = float(torch.linalg.norm(target_velocity_tensor).item())
            target_heading = float(torch.atan2(target_velocity_tensor[1], target_velocity_tensor[0]).item())
            current_xy = (
                state[:, state_adapter.position_slice[0] : state_adapter.position_slice[1]].detach()
                if state_adapter.representation != "actual_state"
                else actual_state[:, :2].detach()
            )
            if waypoint_xy is None:
                waypoint_xy = _make_xy_goal(
                    current_xy,
                    (config.waypoint_x, config.waypoint_y),
                    relative=config.waypoint_relative,
                )
            if obstacle_xy is None:
                obstacle_xy = _make_xy_goal(
                    current_xy,
                    (config.obstacle_x, config.obstacle_y),
                    relative=config.obstacle_relative,
                )
            if inpaint_target_xy is None:
                inpaint_target_xy = _make_xy_goal(
                    current_xy,
                    (config.inpaint_x, config.inpaint_y),
                    relative=config.inpaint_relative,
                )
            planning_tick = step % diffusion_control_stride == 0 or last_latent is None
            if planning_tick:
                target_velocity_schedule = torch.as_tensor(
                    [
                        _target_velocity_for_step(config, step + diffusion_control_stride * (offset + 1))
                        for offset in range(max(1, seq_len - current_index - 1))
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                predicted, diagnostics = _predict_diffusion_tokens(
                    diffusion=diffusion,
                    history=planning_history,
                    current_state=state,
                    seq_len=seq_len,
                    token_dim=token_dim,
                    state_dim=state_dim,
                    current_index=current_index,
                    state_adapter=state_adapter,
                    token_normalizer=token_normalizer,
                    config=config,
                    target_velocity_xy=target_velocity,
                    target_velocity_schedule_xy=target_velocity_schedule,
                    current_velocity_xy=current_velocity_xy,
                    projection_inverse=projection_inverse,
                    waypoint_xy=waypoint_xy,
                    obstacle_xy=obstacle_xy,
                    inpaint_target_xy=inpaint_target_xy,
                    diffusion_frequency_hz=diffusion_frequency_hz,
                    device=device,
                )
                latent = predicted[:, current_index, state_dim:]
                last_latent = latent.detach()
                last_diagnostics = diagnostics
                future_start = min(current_index + 1, predicted.shape[1] - 1)
                if config.physical_velocity_guidance and state_adapter.representation != "actual_state":
                    token_velocity = physical_root_velocity_xy(
                        predicted,
                        state_dim=state_dim,
                        velocity_slice=state_adapter.velocity_slice,
                        velocity_is_relative=state_adapter.velocity_is_relative,
                        current_velocity_xy=current_velocity_xy,
                        projection_inverse=projection_inverse,
                    )
                    future_velocity = token_velocity[:, future_start:]
                    current_planned_velocity = token_velocity[:, current_index]
                else:
                    velocity_start, velocity_end = state_adapter.velocity_slice
                    future_velocity = predicted[:, future_start:, velocity_start:velocity_end]
                    if state_adapter.velocity_is_relative:
                        future_velocity = future_velocity + current_velocity_xy[:, None, :]
                    current_planned_velocity = predicted[:, current_index, velocity_start:velocity_end]
                    if state_adapter.velocity_is_relative:
                        current_planned_velocity = current_planned_velocity + current_velocity_xy
                if future_velocity.shape[1] > 0:
                    next_velocity = future_velocity[:, 0]
                    mean_velocity = future_velocity.mean(dim=1)
                else:
                    next_velocity = current_planned_velocity
                    mean_velocity = next_velocity
                last_plan_metrics = {
                    "planned_next_velocity_xy": next_velocity.detach(),
                    "planned_future_velocity_xy_mean": mean_velocity.detach(),
                    "planned_current_velocity_xy": current_planned_velocity.detach(),
                }
            else:
                latent = last_latent
                diagnostics = last_diagnostics
            with torch.no_grad():
                action = vae.decode(latent, tensors["decoder_proprio_input"])
                unguided_tokens = diagnostics.get("unguided_tokens") if isinstance(diagnostics, dict) else None
                if isinstance(unguided_tokens, torch.Tensor):
                    unguided_latent = unguided_tokens[:, current_index, state_dim:]
                    unguided_action = vae.decode(unguided_latent, tensors["decoder_proprio_input"])
                    decoded_action_diff_norm = torch.linalg.norm(action - unguided_action, dim=-1)
                else:
                    decoded_action_diff_norm = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

            records["actual_state"].append(_to_numpy(actual_state))
            records["diffusion_state"].append(_to_numpy(state))
            for key in [
                "root_height",
                "torso_roll",
                "torso_pitch",
                "physical_fall",
                "contact_force_max",
                "illegal_contact_force_max",
                "physical_illegal_contact",
                "physical_accepted",
            ]:
                records[key].append(_to_numpy(physical[key]))
            for key in ["foot_contact", "foot_height", "flight_phase_proxy"]:
                records[key].append(_to_numpy(foot_contact[key]))
            records["latent"].append(_to_numpy(latent))
            records["clean_action"].append(_to_numpy(action))
            records["executed_action"].append(_to_numpy(action))
            records["accepted"].append(_to_numpy(accepted))
            records["episode_id"].append(_to_numpy(episode_id))
            records["time_index"].append(np.full((config.num_envs,), step, dtype=np.int32))
            records["target_velocity_xy"].append(
                np.repeat(np.asarray(target_velocity, dtype=np.float32).reshape(1, 2), config.num_envs, axis=0)
            )
            records["target_speed"].append(np.full((config.num_envs,), target_speed, dtype=np.float32))
            records["target_heading"].append(np.full((config.num_envs,), target_heading, dtype=np.float32))
            records["actual_velocity_xy_current_yaw"].append(_to_numpy(actual_velocity_current_yaw))
            records["actual_velocity_xy_world"].append(_to_numpy(actual_velocity_world))
            records["actual_speed_current_yaw"].append(_to_numpy(actual_speed_current_yaw))
            records["actual_heading_current_yaw"].append(_to_numpy(actual_heading_current_yaw))
            records["target_turn_rate_z"].append(
                np.full((config.num_envs,), float(config.turn_rate_z), dtype=np.float32)
            )
            records["target_waypoint_xy"].append(_to_numpy(waypoint_xy))
            records["obstacle_xy"].append(_to_numpy(obstacle_xy))
            records["inpaint_target_xy"].append(_to_numpy(inpaint_target_xy))
            records["planned_next_velocity_xy"].append(_to_numpy(last_plan_metrics["planned_next_velocity_xy"]))
            records["planned_future_velocity_xy_mean"].append(
                _to_numpy(last_plan_metrics["planned_future_velocity_xy_mean"])
            )
            records["planned_current_velocity_xy"].append(_to_numpy(last_plan_metrics["planned_current_velocity_xy"]))
            records["unguided_future_velocity_xy_mean"].append(
                _to_numpy(
                    diagnostics.get(
                        "future_velocity_before_xy_mean",
                        torch.zeros(config.num_envs, 2, dtype=torch.float32, device=device),
                    )
                )
            )
            records["guided_future_velocity_xy_mean"].append(
                _to_numpy(
                    diagnostics.get(
                        "future_velocity_after_xy_mean",
                        last_plan_metrics["planned_future_velocity_xy_mean"],
                    )
                )
            )
            records["unguided_future_speed_mean"].append(
                _diagnostic_scalar(diagnostics, "future_speed_before_mean", config.num_envs)
            )
            records["guided_future_speed_mean"].append(
                _diagnostic_scalar(diagnostics, "future_speed_after_mean", config.num_envs)
            )
            records["unguided_future_heading_mean"].append(
                _diagnostic_scalar(diagnostics, "future_heading_before_mean", config.num_envs)
            )
            records["guided_future_heading_mean"].append(
                _diagnostic_scalar(diagnostics, "future_heading_after_mean", config.num_envs)
            )
            records["current_latent_diff_norm"].append(
                _diagnostic_scalar(diagnostics, "current_latent_diff_norm", config.num_envs)
            )
            records["current_latent_grad_norm"].append(
                _diagnostic_scalar(diagnostics, "current_latent_grad_norm", config.num_envs)
            )
            records["decoded_action_diff_norm"].append(_to_numpy(decoded_action_diff_norm))
            records["diffusion_planning_tick"].append(
                np.full((config.num_envs,), bool(planning_tick), dtype=np.bool_)
            )
            records["guidance_cost"].append(_diagnostic_scalar(diagnostics, "guidance_cost", config.num_envs))
            records["guidance_grad_norm"].append(_diagnostic_scalar(diagnostics, "guidance_grad_norm", config.num_envs))

            _, rewards, dones, _ = _step_env(env, action)
            if rewards is None:
                rewards = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
            if dones is None:
                dones = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
            done_mask = dones.to(device=device, dtype=torch.bool)
            accepted = torch.logical_and(accepted, ~done_mask)
            records["reward"].append(_to_numpy(rewards))
            records["done"].append(_to_numpy(done_mask))
            previous_action = action.detach().clone()
            previous_action[done_mask] = 0.0
            last_latent = latent.detach().clone()
            last_latent[done_mask] = 0.0
            if planning_tick:
                token = torch.cat([state, latent.detach()], dim=-1)
                token = token.clone()
                token[done_mask] = 0.0
                token_history.append(token)
                raw_state_history.append(_zero_raw_state_mask(paper_raw_state, done_mask))
                latent_entry = latent.detach().clone()
                latent_entry[done_mask] = 0.0
                latent_history.append(latent_entry)
                if len(token_history) > seq_len:
                    token_history = token_history[-seq_len:]
                if len(raw_state_history) > seq_len:
                    raw_state_history = raw_state_history[-seq_len:]
                if len(latent_history) > seq_len:
                    latent_history = latent_history[-seq_len:]
            if bool(done_mask.any()):
                for token in token_history:
                    token[done_mask] = 0.0
                for raw_state in raw_state_history:
                    for raw_value in raw_state.values():
                        raw_value[done_mask] = 0.0
                for latent_value in latent_history:
                    latent_value[done_mask] = 0.0

        payload = {
            key: _stack_time(value).transpose(1, 0, *range(2, _stack_time(value).ndim))
            for key, value in records.items()
        }
        metadata = VAERolloutMetadata(
            frequency_hz=config.frequency_hz,
            source="diffusion VAE receding-horizon rollout",
            isaac_validation_status="validated by live Isaac diffusion rollout script",
        )
        summary = save_vae_rollout(config.output_path, payload, metadata)
        summary.update(
            {
                "status": "collected",
                "task_name": config.task_name,
                "motion_file": str(motion_file),
                "vae_checkpoint": str(config.vae_checkpoint),
                "diffusion_checkpoint": str(config.diffusion_checkpoint),
                "diffusion_use_ema": bool(config.diffusion_use_ema),
                "warmup_steps": config.warmup_steps,
                "warmup_teacher_checkpoint": teacher_checkpoint,
                "warmup_agent_config": agent_config,
                "diffusion_frequency_hz": diffusion_frequency_hz,
                "diffusion_control_stride": diffusion_control_stride,
                "diffusion_initial_noise_scale": config.diffusion_initial_noise_scale,
                "diffusion_state_representation": state_adapter.representation,
                "diffusion_state_schema": state_adapter.state_schema,
                "diffusion_state_dim": state_adapter.state_dim,
                "diffusion_normalization_enabled": token_normalizer is not None,
                "diffusion_position_slice": list(state_adapter.position_slice),
                "diffusion_velocity_slice": list(state_adapter.velocity_slice),
                "diffusion_angular_velocity_z_index": state_adapter.angular_velocity_z_index,
                "steps": config.steps,
                "num_envs": config.num_envs,
                "accepted_rate": float(payload["accepted"][:, -1].astype(bool).mean()) if payload["accepted"].size else 0.0,
                "physical_accepted_rate": float(payload["physical_accepted"][:, -1].astype(bool).mean())
                if payload["physical_accepted"].size
                else 0.0,
                "physical_fall_rate": float(payload["physical_fall"].astype(bool).mean())
                if payload["physical_fall"].size
                else 0.0,
                "physical_illegal_contact_rate": float(payload["physical_illegal_contact"].astype(bool).mean())
                if payload["physical_illegal_contact"].size
                else 0.0,
                "flight_phase_proxy_rate": float(payload["flight_phase_proxy"].astype(bool).mean())
                if "flight_phase_proxy" in payload and payload["flight_phase_proxy"].size
                else 0.0,
                "root_height_min": float(payload["root_height"].min()) if payload["root_height"].size else None,
                "torso_abs_roll_max": float(np.abs(payload["torso_roll"]).max()) if payload["torso_roll"].size else None,
                "torso_abs_pitch_max": float(np.abs(payload["torso_pitch"]).max()) if payload["torso_pitch"].size else None,
                "illegal_contact_force_max": float(payload["illegal_contact_force_max"].max())
                if payload["illegal_contact_force_max"].size
                else None,
                "physical_min_root_height": config.physical_min_root_height,
                "physical_max_abs_tilt": config.physical_max_abs_tilt,
                "illegal_contact_force_threshold": config.illegal_contact_force_threshold,
                "done_rate": float(payload["done"].astype(bool).mean()) if payload["done"].size else 0.0,
                "reward_mean": float(payload["reward"].mean()) if payload["reward"].size else None,
                "guidance_mode": config.guidance_mode or "none",
                "guidance_scale": config.guidance_scale,
                "guidance_clip_norm": config.guidance_clip_norm,
                "physical_velocity_guidance": bool(config.physical_velocity_guidance),
                "smooth_velocity_ramp": bool(config.smooth_velocity_ramp),
                "velocity_schedule": config.velocity_schedule,
                "velocity_walk_seconds": config.velocity_walk_seconds,
                "velocity_ramp_seconds": config.velocity_ramp_seconds,
                "walk_velocity_xy": [config.walk_velocity_x, config.walk_velocity_y],
                "run_velocity_xy": [config.run_velocity_x, config.run_velocity_y],
                "actual_speed_current_yaw_mean": float(payload["actual_speed_current_yaw"].mean())
                if "actual_speed_current_yaw" in payload and payload["actual_speed_current_yaw"].size
                else None,
                "actual_velocity_x_current_yaw_mean": float(payload["actual_velocity_xy_current_yaw"][..., 0].mean())
                if "actual_velocity_xy_current_yaw" in payload and payload["actual_velocity_xy_current_yaw"].size
                else None,
                "actual_velocity_y_abs_current_yaw_mean": float(np.abs(payload["actual_velocity_xy_current_yaw"][..., 1]).mean())
                if "actual_velocity_xy_current_yaw" in payload and payload["actual_velocity_xy_current_yaw"].size
                else None,
                "unguided_future_speed_mean": float(payload["unguided_future_speed_mean"].mean())
                if "unguided_future_speed_mean" in payload and payload["unguided_future_speed_mean"].size
                else None,
                "guided_future_speed_mean": float(payload["guided_future_speed_mean"].mean())
                if "guided_future_speed_mean" in payload and payload["guided_future_speed_mean"].size
                else None,
                "current_latent_diff_norm_mean": float(payload["current_latent_diff_norm"].mean())
                if "current_latent_diff_norm" in payload and payload["current_latent_diff_norm"].size
                else None,
                "current_latent_grad_norm_mean": float(payload["current_latent_grad_norm"].mean())
                if "current_latent_grad_norm" in payload and payload["current_latent_grad_norm"].size
                else None,
                "decoded_action_diff_norm_mean": float(payload["decoded_action_diff_norm"].mean())
                if "decoded_action_diff_norm" in payload and payload["decoded_action_diff_norm"].size
                else None,
                "turn_rate_z": config.turn_rate_z,
                "waypoint_xy": [config.waypoint_x, config.waypoint_y],
                "waypoint_relative": config.waypoint_relative,
                "obstacle_xy": [config.obstacle_x, config.obstacle_y],
                "obstacle_relative": config.obstacle_relative,
                "obstacle_radius": config.obstacle_radius,
                "obstacle_delta": config.obstacle_delta,
                "inpaint_xy": [config.inpaint_x, config.inpaint_y],
                "inpaint_relative": config.inpaint_relative,
                "inpaint_token_index": config.inpaint_token_index,
                "elapsed_s": time.time() - started,
                "obs_corruption_enabled": bool(getattr(env_cfg.observations.policy, "enable_corruption", False)),
                "events_disabled": bool(config.disable_events),
                "physical_only_terminations": bool(config.physical_only_terminations),
            }
        )
        _write_summary(config.output_path, summary)
        print(f"[BeyondMimic] saved diffusion rollout {config.output_path}", flush=True)
        return summary
    finally:
        env.close()


def _target_velocity_for_step(config: LiveRolloutConfig, step: int) -> tuple[float, float]:
    if config.guidance_mode not in {"velocity", "speed"}:
        return (0.0, 0.0)
    if config.velocity_schedule == "walk_to_run":
        elapsed = float(max(0, step)) / float(max(config.frequency_hz, 1.0e-6))
        walk_seconds = max(0.0, float(config.velocity_walk_seconds))
        ramp_seconds = max(1.0e-6, float(config.velocity_ramp_seconds))
        if elapsed < walk_seconds:
            blend = 0.0
        elif elapsed < walk_seconds + ramp_seconds:
            u = (elapsed - walk_seconds) / ramp_seconds
            blend = 0.5 - 0.5 * torch.cos(torch.tensor(u * torch.pi)).item()
        else:
            blend = 1.0
        walk = torch.tensor((config.walk_velocity_x, config.walk_velocity_y), dtype=torch.float64)
        run = torch.tensor((config.run_velocity_x, config.run_velocity_y), dtype=torch.float64)
        velocity = walk + float(blend) * (run - walk)
        return (float(velocity[0]), float(velocity[1]))
    if config.velocity_schedule not in {"walk_run_walk", "legacy_thirds"}:
        raise ValueError(f"unsupported velocity_schedule={config.velocity_schedule!r}")
    if config.smooth_velocity_ramp and config.velocity_schedule == "walk_run_walk":
        return smooth_walk_run_walk_velocity(
            step,
            config.steps,
            (config.walk_velocity_x, config.walk_velocity_y),
            (config.run_velocity_x, config.run_velocity_y),
        )
    third = max(1, config.steps // 3)
    vx = config.walk_velocity_x if step < third or step >= 2 * third else config.run_velocity_x
    vy = config.walk_velocity_y if step < third or step >= 2 * third else config.run_velocity_y
    return (float(vx), float(vy))


def _make_xy_goal(current_xy: torch.Tensor, xy: tuple[float, float], *, relative: bool) -> torch.Tensor:
    offset = torch.tensor(xy, dtype=current_xy.dtype, device=current_xy.device).view(1, 2)
    if relative:
        return current_xy + offset
    return offset.repeat(current_xy.shape[0], 1)


def _predict_diffusion_tokens(
    *,
    diffusion: torch.nn.Module,
    history: list[torch.Tensor],
    current_state: torch.Tensor,
    seq_len: int,
    token_dim: int,
    state_dim: int,
    current_index: int,
    state_adapter: DiffusionStateAdapter,
    token_normalizer: DiffusionTokenNormalizer | None,
    config: LiveRolloutConfig,
    target_velocity_xy: tuple[float, float],
    target_velocity_schedule_xy: torch.Tensor | None,
    current_velocity_xy: torch.Tensor,
    projection_inverse: torch.Tensor | None,
    waypoint_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    inpaint_target_xy: torch.Tensor,
    diffusion_frequency_hz: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from beyondmimic_repro.stage3.diffusion.sampler import guided_sample
    from beyondmimic_repro.stage3.diffusion.schedule import cosine_alpha_bar_schedule
    from beyondmimic_repro.stage3.guidance.inpainting import inpainting_guidance_cost
    from beyondmimic_repro.stage3.guidance.joystick import joystick_guidance_cost, turn_rate_guidance_cost
    from beyondmimic_repro.stage3.guidance.obstacle import obstacle_guidance_cost
    from beyondmimic_repro.stage3.guidance.waypoint import waypoint_guidance_cost

    latent_dim = token_dim - state_dim
    current_token = torch.cat(
        [current_state, torch.zeros(current_state.shape[0], latent_dim, dtype=current_state.dtype, device=device)],
        dim=-1,
    )
    past_capacity = min(current_index, seq_len - 1)
    future_count = max(0, seq_len - current_index - 1)
    past = history[-past_capacity:]
    if len(past) < past_capacity:
        past = [torch.zeros_like(current_token) for _ in range(past_capacity - len(past))] + past
    tokens = past + [current_token] + [torch.zeros_like(current_token) for _ in range(future_count)]
    noisy = torch.stack(tokens, dim=1)
    if noisy.shape[1] != seq_len:
        raise ValueError(f"diffusion token window must have seq_len={seq_len}, got {noisy.shape[1]}")
    observed_mask = torch.zeros_like(noisy, dtype=torch.bool)
    observed_values = noisy.detach().clone()
    observed_mask[:, :current_index, :] = True
    observed_mask[:, current_index, :state_dim] = True
    observed_step_mask = torch.zeros(noisy.shape[:2] + (2,), dtype=torch.bool, device=device)
    observed_step_mask[:, :current_index, :] = True
    observed_step_mask[:, current_index, 0] = True
    if token_normalizer is not None:
        mean = token_normalizer.mean.view(1, 1, -1)
        std = token_normalizer.std.view(1, 1, -1)
        sampler_input = torch.zeros_like(noisy)
        observed_values_for_sampler = (observed_values - mean) / std
        sampler_input = torch.where(observed_mask, observed_values_for_sampler, sampler_input)
        sampler_masks = {
            "observed_mask": observed_mask,
            "observed_values": observed_values_for_sampler,
            "observed_step_mask": observed_step_mask,
        }
        sampler_mean = token_normalizer.mean
        sampler_std = token_normalizer.std
    else:
        sampler_input = noisy
        sampler_masks = {
            "observed_mask": observed_mask,
            "observed_values": observed_values,
            "observed_step_mask": observed_step_mask,
        }
        sampler_mean = None
        sampler_std = None
    diffusion_steps = torch.zeros(noisy.shape[:2] + (2,), dtype=torch.long, device=device)
    if config.physical_velocity_guidance and config.guidance_mode in {"velocity", "speed"}:
        if target_velocity_schedule_xy is None:
            horizon = max(1, future_count)
            target_velocity_schedule_xy = torch.tensor(target_velocity_xy, dtype=noisy.dtype, device=device).view(1, 2).repeat(
                horizon, 1
            )
        physical_context: dict[str, Any] = {
            "cost_start_index": min(current_index + 1, seq_len - 1),
            "state_dim": state_dim,
            "velocity_slice": state_adapter.velocity_slice,
            "velocity_is_relative": state_adapter.velocity_is_relative,
            "current_velocity_xy": current_velocity_xy,
            "projection_inverse": projection_inverse,
            "target_velocity_xy_schedule": target_velocity_schedule_xy,
            "speed_only": config.guidance_mode == "speed",
            "dt": 1.0 / diffusion_frequency_hz,
        }
        return diagnostic_guided_sample(
            diffusion,
            sampler_input,
            diffusion_steps,
            physical_context,
            sampler_masks,
            guidance_scale=config.guidance_scale,
            gradient_clip_norm=config.guidance_clip_norm,
            alpha_bars=cosine_alpha_bar_schedule(int(getattr(diffusion, "denoising_steps", 20)), device=device),
            denoising_steps=int(getattr(diffusion, "denoising_steps", 20)),
            initial_noise_scale=config.diffusion_initial_noise_scale,
            token_mean=sampler_mean,
            token_std=sampler_std,
            return_denormalized=token_normalizer is not None,
            current_index=current_index,
            state_dim=state_dim,
        )

    guidance_costs: list[Any] = []
    guidance_weights: list[float] = []
    context: dict[str, Any] = {
        "cost_start_index": min(current_index + 1, seq_len - 1),
        "position_slice": state_adapter.position_slice,
        "velocity_slice": state_adapter.velocity_slice,
        "angular_velocity_z_index": state_adapter.angular_velocity_z_index,
        "velocity_offset_xy": current_velocity_xy if state_adapter.velocity_is_relative else None,
        "dt": 1.0 / diffusion_frequency_hz,
    }
    if config.guidance_mode == "velocity":
        guidance_costs.append(joystick_guidance_cost)
        guidance_weights.append(1.0)
        context["target_velocity_xy"] = torch.tensor(target_velocity_xy, dtype=noisy.dtype, device=device)
    elif config.guidance_mode == "speed":
        guidance_costs.append(joystick_guidance_cost)
        guidance_weights.append(1.0)
        target_velocity = torch.tensor(target_velocity_xy, dtype=noisy.dtype, device=device)
        context["target_velocity_xy"] = target_velocity
        context["target_speed"] = torch.linalg.norm(target_velocity)
        context["speed_only"] = True
    elif config.guidance_mode == "turn":
        guidance_costs.append(turn_rate_guidance_cost)
        guidance_weights.append(1.0)
        context["target_turn_rate_z"] = torch.tensor(float(config.turn_rate_z), dtype=noisy.dtype, device=device)
    elif config.guidance_mode == "waypoint":
        guidance_costs.append(waypoint_guidance_cost)
        guidance_weights.append(config.waypoint_weight)
        context["waypoint_xy"] = waypoint_xy
    elif config.guidance_mode == "obstacle":
        guidance_costs.extend([waypoint_guidance_cost, obstacle_guidance_cost])
        guidance_weights.extend([config.waypoint_weight, config.obstacle_weight])
        context["waypoint_xy"] = waypoint_xy
        context["obstacle_xy"] = obstacle_xy
        context["radius"] = config.obstacle_radius
        context["delta"] = config.obstacle_delta
    elif config.guidance_mode == "inpainting":
        target = torch.zeros_like(noisy)
        mask = torch.zeros_like(noisy)
        token_index = config.inpaint_token_index if config.inpaint_token_index >= 0 else seq_len + config.inpaint_token_index
        token_index = int(max(current_index + 1, min(seq_len - 1, token_index)))
        position_start, position_end = state_adapter.position_slice
        target[:, token_index, position_start:position_end] = inpaint_target_xy
        mask[:, token_index, position_start:position_end] = 1.0
        guidance_costs.append(inpainting_guidance_cost)
        guidance_weights.append(1.0)
        context["target"] = target
        context["mask"] = mask
    predicted, diagnostics = guided_sample(
        diffusion,
        sampler_input,
        diffusion_steps,
        context,
        guidance_costs,
        guidance_weights,
        sampler_masks,
        guidance_scale=config.guidance_scale,
        gradient_clip_norm=config.guidance_clip_norm,
        alpha_bars=cosine_alpha_bar_schedule(int(getattr(diffusion, "denoising_steps", 20)), device=device),
        denoising_steps=int(getattr(diffusion, "denoising_steps", 20)),
        prediction_type="x0",
        initial_noise_scale=config.diffusion_initial_noise_scale,
        token_mean=sampler_mean,
        token_std=sampler_std,
        return_denormalized=token_normalizer is not None,
    )
    return predicted, diagnostics


def _diagnostic_scalar(diagnostics: dict[str, torch.Tensor], key: str, num_envs: int) -> np.ndarray:
    value = diagnostics.get(key)
    if value is None:
        return np.zeros((num_envs,), dtype=np.float32)
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 0:
        return np.full((num_envs,), float(arr), dtype=np.float32)
    return arr.reshape(num_envs, -1).mean(axis=1).astype(np.float32)


def _write_summary(output_path: Path, summary: dict[str, object]) -> None:
    summary_path = output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
