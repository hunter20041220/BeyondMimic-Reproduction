"""Shared robot backend protocol for MuJoCo and Isaac bridges."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from beyondmimic_repro.stage2.dagger.interfaces import RobotState


class RobotBackend(Protocol):
    def get_robot_state(self) -> RobotState: ...

    def apply_normalized_action(self, action: np.ndarray) -> None: ...

    def step(self) -> None: ...

    def reset(self, episode_id: int | None = None) -> None: ...
