"""Observation and VAE input shape contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ENCODER_REFERENCE_DIM = 67
DECODER_PROPRIO_DIM = 96
POLICY_OBSERVATION_DIM = 160


@dataclass(frozen=True)
class ArraySpec:
    """JSON-safe description for one tensor field."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str
    frame: str
    units: str
    normalization: str
    semantic_meaning: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "frame": self.frame,
            "units": self.units,
            "normalization": self.normalization,
            "semantic_meaning": self.semantic_meaning,
        }


PAPER_VAE_ENCODER_REFERENCE_SPEC = ArraySpec(
    name="encoder_reference_input",
    shape=("batch", ENCODER_REFERENCE_DIM),
    dtype="float32",
    frame="reference/current-anchor contract",
    units="mixed: rad, rad/s, m, Rot6D",
    normalization="dataset mean/std",
    semantic_meaning="reference joint pos/vel, anchor position error, anchor orientation Rot6D",
)

PAPER_VAE_DECODER_PROPRIO_SPEC = ArraySpec(
    name="decoder_proprio_input",
    shape=("batch", DECODER_PROPRIO_DIM),
    dtype="float32",
    frame="robot proprioception",
    units="mixed: gravity, twist, rad, rad/s, normalized previous action",
    normalization="dataset mean/std",
    semantic_meaning="projected gravity, root twist, current joint pos/vel, previous normalized action",
)


def validate_last_dim(name: str, value: np.ndarray, expected_dim: int) -> np.ndarray:
    """Return a finite float32 array with a checked trailing dimension."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr
