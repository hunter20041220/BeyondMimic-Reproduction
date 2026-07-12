"""Protocols for paper-correct DAgger collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Minimal robot state exchanged by backends and policy frontends."""

    policy_observation: np.ndarray
    root_state: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    previous_action: np.ndarray


class StudentPolicy(Protocol):
    def act(self, reference_input: np.ndarray, robot_state: RobotState, previous_action: np.ndarray) -> np.ndarray: ...


class TeacherPolicy(Protocol):
    def label(self, robot_state: RobotState, reference_frame_index: int) -> np.ndarray: ...


class RobotBackend(Protocol):
    def get_robot_state(self) -> RobotState: ...

    def apply_normalized_action(self, action: np.ndarray) -> None: ...

    def step(self) -> None: ...

    def reset(self, episode_id: int | None = None) -> None: ...
