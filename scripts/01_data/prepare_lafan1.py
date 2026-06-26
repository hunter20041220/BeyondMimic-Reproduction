#!/usr/bin/env python3
"""Pack retargeted LAFAN1 G1 CSV files into a compact NPZ dataset."""

from __future__ import annotations

import argparse

from beyondmimic_repro.data.lafan1 import prepare_lafan1_npz
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing retargeted no-header LAFAN1 CSV files.")
    parser.add_argument("--output", default="data/processed/lafan1_g1.npz")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of CSV files for quick smoke runs.")
    parser.add_argument("--summary", default="outputs/metrics/prepare_lafan1_summary.json")
    args = parser.parse_args()
    summary = prepare_lafan1_npz(args.input, args.output, limit=args.limit)
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
