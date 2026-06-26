"""Typed DAgger sample schema for teacher-query audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from beyondmimic_repro.validation import ensure_finite


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DAggerSample:
    """One finite teacher-query sample from a trajectory rollout.

    Shapes are ``state[state_dim]``, ``student_action[action_dim]``, and
    ``teacher_action[action_dim]``. The ``teacher_queried`` flag distinguishes
    true DAgger-style supervision from offline behavior-cloning rows.
    """

    sample_id: str
    rollout_id: str
    timestep: int
    state: np.ndarray
    student_action: np.ndarray
    teacher_action: np.ndarray
    teacher_queried: bool
    accepted: bool
    split: SplitName

    def to_metadata(self: "DAggerSample") -> dict[str, Any]:
        """Return JSON-safe trajectory provenance with array shape metadata."""
        return {
            **asdict(self),
            "state": {"shape": list(self.state.shape)},
            "student_action": {"shape": list(self.student_action.shape)},
            "teacher_action": {"shape": list(self.teacher_action.shape)},
        }


def build_dagger_sample(
    sample_id: str,
    rollout_id: str,
    timestep: int,
    state: np.ndarray,
    student_action: np.ndarray,
    teacher_action: np.ndarray,
    teacher_queried: bool,
    accepted: bool,
    split: SplitName,
) -> DAggerSample:
    """Validate and build a finite DAgger sample with action shape ``[A]``."""
    state_arr = ensure_finite("state", state)
    student_arr = ensure_finite("student_action", student_action)
    teacher_arr = ensure_finite("teacher_action", teacher_action)
    if not sample_id or not rollout_id:
        raise ValueError("sample_id and rollout_id must be nonempty")
    if timestep < 0:
        raise ValueError("timestep must be nonnegative")
    if state_arr.ndim != 1 or student_arr.ndim != 1 or teacher_arr.ndim != 1:
        raise ValueError("state and action arrays must be one-dimensional")
    if student_arr.shape != teacher_arr.shape:
        raise ValueError(f"student action shape {student_arr.shape} != teacher action shape {teacher_arr.shape}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid split {split!r}")
    return DAggerSample(
        sample_id=sample_id,
        rollout_id=rollout_id,
        timestep=timestep,
        state=state_arr,
        student_action=student_arr,
        teacher_action=teacher_arr,
        teacher_queried=teacher_queried,
        accepted=accepted,
        split=split,
    )


def teacher_student_discrepancy(samples: list[DAggerSample]) -> dict[str, float]:
    """Compute DAgger teacher/student action metrics for samples ``[N,A]``."""
    if not samples:
        raise ValueError("samples must be nonempty")
    diffs = []
    queried = 0
    accepted = 0
    for sample in samples:
        if not sample.teacher_queried:
            raise ValueError("all samples must record teacher_queried=True for DAgger discrepancy")
        diffs.append(sample.student_action - sample.teacher_action)
        queried += int(sample.teacher_queried)
        accepted += int(sample.accepted)
    stacked = ensure_finite("action_differences", np.stack(diffs, axis=0))
    per_sample_mse = np.mean(stacked**2, axis=1)
    return {
        "sample_count": float(len(samples)),
        "teacher_query_count": float(queried),
        "accepted_count": float(accepted),
        "action_mse": float(np.mean(per_sample_mse)),
        "action_rmse": float(np.sqrt(np.mean(per_sample_mse))),
        "max_abs_action_error": float(np.max(np.abs(stacked))),
    }
