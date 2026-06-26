"""LAFAN1 retargeted CSV preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
G1_ACTION_DIM = 29
LAFAN1_G1_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM + G1_ACTION_DIM


@dataclass(frozen=True)
class MotionRecord:
    """One prepared motion sequence."""

    name: str
    path: str
    frames: int
    feature_dim: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_lafan1_csv(path: str | Path) -> np.ndarray:
    """Load one no-header retargeted LAFAN1 CSV with 36 columns."""
    csv_path = Path(path)
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != LAFAN1_G1_DIM:
        raise ValueError(f"expected [T,{LAFAN1_G1_DIM}] retargeted G1 CSV, got {data.shape} from {csv_path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{csv_path} contains NaN or Inf")
    return data


def split_lafan1_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Split raw 36-D rows into root and 29-D action channels."""
    if data.ndim != 2 or data.shape[1] != LAFAN1_G1_DIM:
        raise ValueError(f"expected [T,{LAFAN1_G1_DIM}] data, got {data.shape}")
    return {
        "root_pos": data[:, :3],
        "root_quat_xyzw": data[:, 3:7],
        "actions": data[:, 7:],
    }


def collect_csv_files(input_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Collect CSV files below `input_dir` in deterministic order."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.csv"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no CSV files found under {root}")
    return files


def prepare_lafan1_npz(input_dir: str | Path, output_path: str | Path, limit: int | None = None) -> dict[str, object]:
    """Pack retargeted LAFAN1 CSV files into a compact NPZ dataset."""
    files = collect_csv_files(input_dir, limit=limit)
    motions: list[np.ndarray] = []
    records: list[MotionRecord] = []
    max_frames = 0
    for csv_path in files:
        data = load_lafan1_csv(csv_path)
        motions.append(data)
        max_frames = max(max_frames, data.shape[0])
        records.append(MotionRecord(name=csv_path.stem, path=str(csv_path), frames=data.shape[0], feature_dim=data.shape[1]))

    padded = np.zeros((len(motions), max_frames, LAFAN1_G1_DIM), dtype=np.float32)
    lengths = np.zeros(len(motions), dtype=np.int64)
    for idx, data in enumerate(motions):
        lengths[idx] = data.shape[0]
        padded[idx, : data.shape[0]] = data.astype(np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motions=padded,
        lengths=lengths,
        names=np.array([record.name for record in records]),
        schema=np.array(["root_pos_xyz:0:3", "root_quat_xyzw:3:7", "g1_action_29d:7:36"]),
    )
    return {
        "status": "ok",
        "output": str(output),
        "motion_count": len(records),
        "total_frames": int(lengths.sum()),
        "feature_dim": LAFAN1_G1_DIM,
        "records": [record.to_dict() for record in records],
    }
