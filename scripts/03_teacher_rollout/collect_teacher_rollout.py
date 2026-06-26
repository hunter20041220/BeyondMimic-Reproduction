#!/usr/bin/env python3
"""Build a teacher rollout dataset from prepared motion data."""

from __future__ import annotations

import argparse

from beyondmimic_repro.rollout.from_motion import motion_npz_to_teacher_rollouts
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/synthetic_lafan1_g1.npz")
    parser.add_argument("--output", default="data/teacher_rollouts/teacher_rollout_train.npz")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--summary", default="outputs/metrics/teacher_rollout_summary.json")
    args = parser.parse_args()
    summary = motion_npz_to_teacher_rollouts(
        args.input,
        args.output,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
