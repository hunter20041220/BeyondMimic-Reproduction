"""Teacher asset loading for Isaac entrypoints."""

from __future__ import annotations

from pathlib import Path

from beyondmimic_repro.contracts.teacher_assets import TeacherAssets, load_teacher_map, validate_teacher_assets


def load_isaac_teacher_assets(
    teacher_map: str | Path,
    *,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    require_files: bool = True,
) -> dict[str, TeacherAssets]:
    """Load and validate assets after path relocation on the 4090 host."""
    assets = load_teacher_map(teacher_map, data_root=data_root, checkpoint_root=checkpoint_root)
    errors = validate_teacher_assets(assets, require_files=require_files)
    if errors:
        raise ValueError("; ".join(errors))
    return assets
