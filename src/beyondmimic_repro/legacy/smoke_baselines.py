"""Aliases for the original simplified public baselines."""

from __future__ import annotations

from beyondmimic_repro.data.state_latent import build_state_latent_dataset as LegacyStateLatentTeacherEncoder
from beyondmimic_repro.diffusion.torch_denoiser import StateLatentTransformerDenoiser as LegacyEpsilonDenoiser
from beyondmimic_repro.vae.torch_model import ConditionalActionVAE as LegacyConditionalActionVAE
