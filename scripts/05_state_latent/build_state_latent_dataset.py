#!/usr/bin/env python3
"""Build state-latent trajectory windows from teacher rollouts."""

from __future__ import annotations

import argparse

from beyondmimic_repro.data.state_latent import build_state_latent_dataset
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", default="data/teacher_rollouts/teacher_rollout_train.npz")
    parser.add_argument("--output", default="data/state_latent/train_windows.npz")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--summary", default="outputs/metrics/state_latent_summary.json")
    args = parser.parse_args()
    summary = build_state_latent_dataset(args.rollout, args.output, latent_dim=args.latent_dim)
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
