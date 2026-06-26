#!/usr/bin/env python3
"""Evaluate a tracking-policy rollout dataset schema.

This clean repository does not bundle official BeyondMimic tracking
checkpoints. If a trained IsaacLab/RSL-RL policy is available, use the same
rollout schema written by `scripts/03_teacher_rollout/collect_teacher_rollout.py`.
"""

from __future__ import annotations

import argparse

from beyondmimic_repro.rollout.schema import load_teacher_rollout
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", default="data/teacher_rollouts/teacher_rollout_train.npz")
    parser.add_argument("--summary", default="outputs/metrics/tracking_eval_summary.json")
    args = parser.parse_args()
    states, actions, names = load_teacher_rollout(args.rollout)
    summary = {
        "status": "ok_schema_eval",
        "rollout": args.rollout,
        "rollout_count": int(states.shape[0]),
        "horizon": int(states.shape[1]),
        "state_dim": int(states.shape[2]),
        "action_dim": int(actions.shape[2]),
        "first_rollout": str(names[0]),
        "official_tracking_checkpoint_included": False,
    }
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
