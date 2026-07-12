"""D0 warm-start helpers for teacher closed-loop rollouts."""

from __future__ import annotations

from pathlib import Path

from beyondmimic_repro.contracts.teacher_rollout import load_teacher_rollout_contract


BC_WARMSTART_NOTICE = "This is offline BC warm start, not a completed DAgger distillation."


def load_d0_teacher_rollout(path: str | Path) -> tuple[dict[str, object], object]:
    """Load D0 teacher rollout data and return the required warning message."""
    payload, metadata = load_teacher_rollout_contract(path)
    return {"notice": BC_WARMSTART_NOTICE, "payload": payload}, metadata
