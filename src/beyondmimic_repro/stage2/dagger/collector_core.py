"""Core DAgger collection loop.

The student action is executed. The teacher action is only a supervision label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beyondmimic_repro.contracts.action import validate_normalized_action
from beyondmimic_repro.stage2.dagger.interfaces import RobotBackend, StudentPolicy, TeacherPolicy
from beyondmimic_repro.stage2.dagger.teacher_query import query_teacher_action


@dataclass(frozen=True)
class DAggerCollectionStep:
    """One student-state teacher-label row."""

    step_index: int
    reference_frame_index: int
    student_action: np.ndarray
    teacher_action: np.ndarray
    policy_observation: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


def collect_dagger_steps(
    backend: RobotBackend,
    student: StudentPolicy,
    teacher: TeacherPolicy,
    reference_inputs: np.ndarray,
    *,
    start_reference_frame: int = 0,
) -> list[DAggerCollectionStep]:
    """Collect a synthetic or Isaac DAgger round with correct control semantics."""
    refs = np.asarray(reference_inputs, dtype=np.float32)
    if refs.ndim != 2:
        raise ValueError(f"reference_inputs must be [T,D], got {refs.shape}")
    rows: list[DAggerCollectionStep] = []
    for local_step, reference_input in enumerate(refs):
        robot_state = backend.get_robot_state()
        ref_idx = start_reference_frame + local_step
        student_action = validate_normalized_action(student.act(reference_input, robot_state, robot_state.previous_action))
        teacher_action = query_teacher_action(teacher, robot_state, ref_idx)
        backend.apply_normalized_action(student_action)
        backend.step()
        rows.append(
            DAggerCollectionStep(
                step_index=local_step,
                reference_frame_index=ref_idx,
                student_action=student_action,
                teacher_action=teacher_action,
                policy_observation=np.asarray(robot_state.policy_observation, dtype=np.float32),
                joint_position=np.asarray(robot_state.joint_position, dtype=np.float32),
                joint_velocity=np.asarray(robot_state.joint_velocity, dtype=np.float32),
            )
        )
    return rows
