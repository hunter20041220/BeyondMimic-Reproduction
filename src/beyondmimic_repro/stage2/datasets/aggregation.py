"""DAgger round aggregation helpers."""

from __future__ import annotations

from pathlib import Path

from beyondmimic_repro.contracts.dagger_dataset import merge_dagger_rounds


def aggregate_dagger_rounds(round_paths: list[str | Path], output_path: str | Path) -> dict[str, object]:
    """Merge D0, D1, ... without changing sample semantics."""
    return merge_dagger_rounds(round_paths, output_path)
