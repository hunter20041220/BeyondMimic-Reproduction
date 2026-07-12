"""Explicit Stage-3 trajectory window semantics."""

from __future__ import annotations

from beyondmimic_repro.contracts.state_latent import TimeWindowConfig


def paper_sequence_length(past_steps: int = 4, include_current: bool = True, future_steps: int = 16) -> int:
    """Return past + current + future length without using ambiguous horizon names."""
    return past_steps + int(include_current) + future_steps


def paper_time_window_50hz() -> TimeWindowConfig:
    """Engineering gate: 16 / 50 = 0.32 s future horizon."""
    return TimeWindowConfig(frequency_hz=50.0)


def paper_time_window_25hz() -> TimeWindowConfig:
    """Paper-faithful timing: 16 / 25 = 0.64 s future horizon."""
    return TimeWindowConfig(frequency_hz=25.0)
