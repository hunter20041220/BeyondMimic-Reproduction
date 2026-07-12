"""VAE rollout acceptance policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VAERolloutAcceptance:
    """Paper-style perturb then closed-loop survival gate."""

    perturbation_duration_s: float = 2.5
    validation_duration_s: float = 5.0


def evaluate_rollout_acceptance(
    *,
    survival_time_s: float,
    failed: bool,
    rejection_reason: str = "",
    rule: VAERolloutAcceptance | None = None,
) -> dict[str, object]:
    """Accept episodes that survive the full 5.0 s validation window."""
    rule = rule or VAERolloutAcceptance()
    accepted = (not failed) and survival_time_s >= rule.validation_duration_s
    return {
        "accepted": accepted,
        "rejection_reason": "" if accepted else (rejection_reason or "failed_before_5s"),
        "survival_time": float(survival_time_s),
        "perturbation_duration_s": rule.perturbation_duration_s,
        "validation_duration_s": rule.validation_duration_s,
    }
