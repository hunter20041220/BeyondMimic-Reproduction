"""Legacy/smoke baselines retained for comparison, not paper-faithful claims."""

from beyondmimic_repro.legacy.smoke_baselines import (
    LegacyConditionalActionVAE,
    LegacyEpsilonDenoiser,
    LegacyStateLatentTeacherEncoder,
)

__all__ = ["LegacyConditionalActionVAE", "LegacyEpsilonDenoiser", "LegacyStateLatentTeacherEncoder"]
