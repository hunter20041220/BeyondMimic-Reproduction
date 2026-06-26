"""Latent-space formulas for VAE smoke tests."""

from __future__ import annotations

import numpy as np

from beyondmimic_repro.validation import ensure_finite


def reparameterize(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Apply VAE reparameterization for finite latent vectors ``[latent_dim]``."""
    mu = ensure_finite("mu", mu)
    logvar = ensure_finite("logvar", logvar)
    eps = ensure_finite("eps", eps)
    if mu.shape != logvar.shape or mu.shape != eps.shape:
        raise ValueError(f"latent shapes must match, got {mu.shape}, {logvar.shape}, {eps.shape}")
    return mu + eps * np.exp(0.5 * logvar)


def kl_standard_normal(mu: np.ndarray, logvar: np.ndarray) -> float:
    """KL(N(mu,var)||N(0,I)) summed over finite latent vector ``[latent_dim]``."""
    mu = ensure_finite("mu", mu)
    logvar = ensure_finite("logvar", logvar)
    if mu.shape != logvar.shape:
        raise ValueError(f"latent shapes must match, got {mu.shape}, {logvar.shape}")
    return float(-0.5 * np.sum(1.0 + logvar - mu**2 - np.exp(logvar)))
