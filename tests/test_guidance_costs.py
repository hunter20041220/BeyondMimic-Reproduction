from __future__ import annotations

import numpy as np

from beyondmimic_repro.guidance.tasks import joystick_cost, waypoint_cost


def test_waypoint_cost_zero_at_target() -> None:
    tokens = np.zeros((1, 4, 5), dtype=np.float32)
    tokens[0, -1, :2] = np.array([1.0, 0.5])
    assert waypoint_cost(tokens, np.array([1.0, 0.5])) == 0.0


def test_joystick_cost_zero_for_constant_velocity() -> None:
    tokens = np.zeros((1, 4, 5), dtype=np.float32)
    tokens[0, :, 0] = np.array([0.0, 0.1, 0.2, 0.3])
    cost = joystick_cost(tokens, np.array([0.1, 0.0]))
    assert cost < 1e-12
