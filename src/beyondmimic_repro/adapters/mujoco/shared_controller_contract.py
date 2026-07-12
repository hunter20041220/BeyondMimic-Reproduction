"""Controller contract extracted from Stage-1 MuJoCo bridges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beyondmimic_repro.contracts.action import validate_normalized_action


@dataclass(frozen=True)
class JointControllerMetadata:
    """Joint and PD metadata required by a backend."""

    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    torque_limit: np.ndarray
    control_dt: float
    simulation_dt: float
    quaternion_format: str = "wxyz"


def map_named_vector(values: dict[str, float], joint_names: tuple[str, ...]) -> np.ndarray:
    """Map a name-value dict into controller joint order."""
    missing = [name for name in joint_names if name not in values]
    if missing:
        raise ValueError(f"missing joint values for {missing}")
    return np.asarray([values[name] for name in joint_names], dtype=np.float32)


def normalized_action_to_pd_target(action: np.ndarray, metadata: JointControllerMetadata) -> np.ndarray:
    """Convert normalized 29-D action to target joint position."""
    act = validate_normalized_action(action, action_dim=len(metadata.joint_names))
    return metadata.default_joint_pos + metadata.action_scale * act


def pd_torque(
    target_joint_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    metadata: JointControllerMetadata,
) -> np.ndarray:
    """Compute clipped PD torque."""
    torque = metadata.stiffness * (target_joint_pos - joint_pos) - metadata.damping * joint_vel
    return np.clip(torque, -metadata.torque_limit, metadata.torque_limit)
