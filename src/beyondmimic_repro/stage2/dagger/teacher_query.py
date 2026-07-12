"""Teacher query helpers for DAgger labels."""

from __future__ import annotations

import numpy as np

from beyondmimic_repro.contracts.action import validate_normalized_action
from beyondmimic_repro.stage2.dagger.interfaces import RobotState, TeacherPolicy


def query_teacher_action(teacher: TeacherPolicy, robot_state: RobotState, reference_frame_index: int) -> np.ndarray:
    """Query a teacher label for the current student state."""
    return validate_normalized_action(teacher.label(robot_state, reference_frame_index))
