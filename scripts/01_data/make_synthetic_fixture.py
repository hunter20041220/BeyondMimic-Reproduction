#!/usr/bin/env python3
"""Create a tiny synthetic motion fixture for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_fixture(output: Path, motions: int, frames: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    data = np.zeros((motions, frames, 36), dtype=np.float32)
    for idx in range(motions):
        phase = idx * 0.37
        data[idx, :, 0] = 0.8 * t
        data[idx, :, 1] = 0.1 * np.sin(2.0 * np.pi * (t + phase))
        data[idx, :, 2] = 0.78 + 0.02 * np.sin(4.0 * np.pi * (t + phase))
        data[idx, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        base = np.sin(2.0 * np.pi * (t[:, None] + phase) + np.linspace(0.0, 2.0, 29)[None, :])
        data[idx, :, 7:] = 0.35 * base + 0.01 * rng.normal(size=(frames, 29))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motions=data,
        lengths=np.full(motions, frames, dtype=np.int64),
        names=np.array([f"synthetic_{idx:03d}" for idx in range(motions)]),
        schema=np.array(["root_pos_xyz:0:3", "root_quat_xyzw:3:7", "g1_action_29d:7:36"]),
    )
    return {"status": "ok", "output": str(output), "motion_count": motions, "frames": frames, "feature_dim": 36}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/synthetic_lafan1_g1.npz")
    parser.add_argument("--motions", type=int, default=4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(make_fixture(Path(args.output), args.motions, args.frames, args.seed))


if __name__ == "__main__":
    main()
