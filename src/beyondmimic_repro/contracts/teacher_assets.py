"""Teacher checkpoint, motion, ONNX, and rollout asset contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherAssets:
    """Relocatable metadata for one Stage-1 tracking teacher."""

    motion_name: str
    checkpoint_path: Path
    onnx_path: Path
    teacher_rollout_path: Path
    motion_file: Path
    task_name: str
    frequency_hz: float
    checkpoint_iteration: int
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    anchor_body_name: str
    checkpoint_sha256: str = ""
    onnx_sha256: str = ""
    motion_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ["checkpoint_path", "onnx_path", "teacher_rollout_path", "motion_file"]:
            data[key] = str(data[key])
        if data["onnx_path"] == ".":
            data["onnx_path"] = ""
        data["joint_names"] = list(self.joint_names)
        data["body_names"] = list(self.body_names)
        return data


def sha256_file(path: str | Path) -> str:
    """Return SHA256 for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _relocate(path_value: Any, *, root: Path | None) -> Path:
    path = Path(str(path_value or ""))
    if not path_value:
        return Path("")
    if root is not None and path.is_absolute():
        return root / path.name
    if root is not None and not path.is_absolute():
        return root / path
    return path


def teacher_assets_from_record(
    record: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> TeacherAssets:
    """Build a relocatable asset record from flexible teacher_map JSON keys."""
    data_root_path = Path(data_root).expanduser().resolve() if data_root else None
    checkpoint_root_path = Path(checkpoint_root).expanduser().resolve() if checkpoint_root else data_root_path
    motion_name = str(_first(record, "motion_name", "name", "motion", default="unknown_motion"))
    return TeacherAssets(
        motion_name=motion_name,
        checkpoint_path=_relocate(_first(record, "checkpoint_path", "checkpoint", "pt_path"), root=checkpoint_root_path),
        onnx_path=_relocate(_first(record, "onnx_path", "onnx", "policy_onnx"), root=checkpoint_root_path),
        teacher_rollout_path=_relocate(
            _first(record, "teacher_rollout_path", "rollout_path", "teacher_rollout"),
            root=data_root_path,
        ),
        motion_file=_relocate(_first(record, "motion_file", "motion_path", "motion_npz"), root=data_root_path),
        task_name=str(_first(record, "task_name", "task", default="Tracking-Flat-G1-v0")),
        frequency_hz=float(_first(record, "frequency_hz", "fps", "control_frequency", default=50.0)),
        checkpoint_iteration=int(_first(record, "checkpoint_iteration", "iteration", "iter", default=-1)),
        joint_names=tuple(str(v) for v in _first(record, "joint_names", default=[])),
        body_names=tuple(str(v) for v in _first(record, "body_names", default=[])),
        anchor_body_name=str(_first(record, "anchor_body_name", "anchor", default="pelvis")),
        checkpoint_sha256=str(_first(record, "checkpoint_sha256", "checkpoint_hash", default="")),
        onnx_sha256=str(_first(record, "onnx_sha256", "onnx_hash", default="")),
        motion_sha256=str(_first(record, "motion_sha256", "motion_hash", default="")),
    )


def load_teacher_map(
    teacher_map: str | Path,
    *,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> dict[str, TeacherAssets]:
    """Load a teacher map without requiring source-machine absolute paths."""
    payload = json.loads(Path(teacher_map).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "teachers" in payload:
        raw_records = payload["teachers"]
    elif isinstance(payload, dict):
        raw_records = [
            {"motion_name": key, **value} if isinstance(value, dict) else {"motion_name": key, "checkpoint_path": value}
            for key, value in payload.items()
        ]
    elif isinstance(payload, list):
        raw_records = payload
    else:
        raise ValueError("teacher_map must be a JSON dict/list or contain a 'teachers' list")
    assets = [
        teacher_assets_from_record(dict(record), data_root=data_root, checkpoint_root=checkpoint_root)
        for record in raw_records
    ]
    return {asset.motion_name: asset for asset in assets}


def validate_frequency(asset: TeacherAssets, *, expected_hz: float | None = None) -> None:
    """Reject nonpositive or mismatched control frequencies."""
    if asset.frequency_hz <= 0.0:
        raise ValueError(f"{asset.motion_name}: frequency_hz must be positive")
    if expected_hz is not None and abs(asset.frequency_hz - expected_hz) > 1e-6:
        raise ValueError(f"{asset.motion_name}: frequency_hz {asset.frequency_hz} != expected {expected_hz}")


def validate_motion_mapping(asset: TeacherAssets, *, require_joints: bool = True) -> None:
    """Validate joint/body metadata needed by controller bridges."""
    if require_joints and not asset.joint_names:
        raise ValueError(f"{asset.motion_name}: joint_names are required")
    if not asset.anchor_body_name:
        raise ValueError(f"{asset.motion_name}: anchor_body_name is required")
    if asset.body_names and asset.anchor_body_name not in asset.body_names:
        raise ValueError(f"{asset.motion_name}: anchor body {asset.anchor_body_name!r} is not in body_names")


def validate_teacher_assets(
    assets: dict[str, TeacherAssets] | list[TeacherAssets],
    *,
    require_files: bool = False,
    expected_hz: float | None = None,
) -> list[str]:
    """Validate asset metadata and optionally verify local files and hashes."""
    records = list(assets.values()) if isinstance(assets, dict) else list(assets)
    errors: list[str] = []
    for asset in records:
        try:
            validate_frequency(asset, expected_hz=expected_hz)
            validate_motion_mapping(asset, require_joints=False)
            if require_files:
                for attr in ["checkpoint_path", "teacher_rollout_path", "motion_file"]:
                    path = getattr(asset, attr)
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"{asset.motion_name}: missing {attr}: {path}")
                onnx_path = Path(asset.onnx_path)
                if str(onnx_path) not in ("", ".") and not onnx_path.is_file():
                    raise FileNotFoundError(f"{asset.motion_name}: missing onnx_path: {onnx_path}")
                if asset.checkpoint_sha256 and sha256_file(asset.checkpoint_path) != asset.checkpoint_sha256:
                    raise ValueError(f"{asset.motion_name}: checkpoint SHA256 mismatch")
                if asset.onnx_sha256 and str(onnx_path) not in ("", ".") and sha256_file(onnx_path) != asset.onnx_sha256:
                    raise ValueError(f"{asset.motion_name}: ONNX SHA256 mismatch")
                if asset.motion_sha256 and asset.motion_file and sha256_file(asset.motion_file) != asset.motion_sha256:
                    raise ValueError(f"{asset.motion_name}: motion SHA256 mismatch")
        except Exception as exc:  # noqa: BLE001 - return all validation failures
            errors.append(str(exc))
    return errors


def with_relocated_roots(
    asset: TeacherAssets,
    *,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> TeacherAssets:
    """Return a copy with path fields relocated by filename under new roots."""
    data_root_path = Path(data_root).expanduser().resolve() if data_root else None
    checkpoint_root_path = Path(checkpoint_root).expanduser().resolve() if checkpoint_root else data_root_path
    return replace(
        asset,
        checkpoint_path=_relocate(asset.checkpoint_path, root=checkpoint_root_path),
        onnx_path=_relocate(asset.onnx_path, root=checkpoint_root_path),
        teacher_rollout_path=_relocate(asset.teacher_rollout_path, root=data_root_path),
        motion_file=_relocate(asset.motion_file, root=data_root_path),
    )
