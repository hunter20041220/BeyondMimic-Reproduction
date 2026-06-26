from __future__ import annotations

import numpy as np

from beyondmimic_repro.rollout.schema import validate_teacher_rollout
from beyondmimic_repro.trajectory.dataset import stack_state_latent_tokens


def test_teacher_rollout_schema() -> None:
    states = np.zeros((2, 4, 8), dtype=np.float32)
    actions = np.zeros((2, 4, 29), dtype=np.float32)
    state_arr, action_arr = validate_teacher_rollout(states, actions)
    assert state_arr.shape == (2, 4, 8)
    assert action_arr.shape == (2, 4, 29)


def test_state_latent_stack_shape() -> None:
    states = np.zeros((5, 12), dtype=np.float32)
    latents = np.zeros((5, 3), dtype=np.float32)
    tokens = stack_state_latent_tokens(states, latents)
    assert tokens.shape == (5, 15)
