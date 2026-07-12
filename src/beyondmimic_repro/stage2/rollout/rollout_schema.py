"""Rows recorded by VAE student rollout collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beyondmimic_repro.contracts.action import validate_normalized_action


@dataclass(frozen=True)
class VAERolloutStep:
    """Action/noise row needed for Stage-3 state-latent construction."""

    actual_state: np.ndarray
    latent: np.ndarray
    clean_action: np.ndarray
    ou_noise: np.ndarray
    executed_action: np.ndarray
    accepted: bool
    rejection_reason: str
    survival_time: float

    def validate(self) -> "VAERolloutStep":
        validate_normalized_action(self.clean_action)
        validate_normalized_action(self.executed_action)
        if self.clean_action.shape != self.ou_noise.shape or self.clean_action.shape != self.executed_action.shape:
            raise ValueError("clean_action, ou_noise, and executed_action shapes must match")
        return self
