from __future__ import annotations

import numpy as np

from beyondmimic_repro.diffusion.schedules import apply_observation_mask, q_sample


def test_q_sample_shape() -> None:
    x0 = np.ones((4, 6))
    eps = np.zeros((4, 6))
    noisy = q_sample(x0, eps, alpha_bar=0.25)
    assert noisy.shape == x0.shape
    assert np.allclose(noisy, 0.5)


def test_apply_observation_mask() -> None:
    noisy = np.zeros((2, 3))
    clean = np.ones((2, 3))
    mask = np.array([[True, False, False], [False, True, False]])
    out = apply_observation_mask(noisy, clean, mask)
    assert out[0, 0] == 1.0
    assert out[1, 1] == 1.0
    assert out[0, 1] == 0.0
