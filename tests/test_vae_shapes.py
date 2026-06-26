from __future__ import annotations

import numpy as np

from beyondmimic_repro.vae.latent import kl_standard_normal, reparameterize


def test_reparameterize_shape() -> None:
    mu = np.zeros(4)
    logvar = np.zeros(4)
    eps = np.ones(4)
    latent = reparameterize(mu, logvar, eps)
    assert latent.shape == (4,)
    assert np.allclose(latent, 1.0)


def test_kl_standard_normal_zero() -> None:
    assert kl_standard_normal(np.zeros(3), np.zeros(3)) == 0.0
