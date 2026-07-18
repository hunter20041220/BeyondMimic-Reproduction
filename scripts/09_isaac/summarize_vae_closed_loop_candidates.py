#!/usr/bin/env python3
"""Summarize VAE closed-loop candidate rollouts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


FILENAME_RE = re.compile(
    r"vae_(?P<candidate>best_total|best_action|latest)_(?P<motion>.+)_"
    r"(?P<envs>\d+)env_(?P<steps>\d+)steps_physical\.npz$"
)


def _first_true_mean(mask: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError(f"expected [env, step] mask, got shape={mask.shape}")
    steps = mask.shape[1]
    first = np.full(mask.shape[0], steps, dtype=np.int32)
    any_hit = mask.any(axis=1)
    first[any_hit] = np.argmax(mask[any_hit], axis=1)
    return float(first.mean())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def summarize_rollout(path: Path) -> dict[str, Any] | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None

    sidecar = Path(str(path)[:-4] + ".json")
    if not sidecar.is_file():
        return None
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))

    with np.load(path, allow_pickle=False) as data:
        done = data["done"].astype(bool)
        accepted = data["accepted"].astype(bool)
        physical_accepted = (
            data["physical_accepted"].astype(bool) if "physical_accepted" in data.files else accepted
        )
        physical_fail = ~physical_accepted
        item = {
            "candidate": match.group("candidate"),
            "motion": match.group("motion"),
            "accepted_final_rate": float(accepted[:, -1].mean()),
            "accepted_mean_rate": float(accepted.mean()),
            "physical_accepted_final_rate": float(physical_accepted[:, -1].mean()),
            "physical_accepted_mean_rate": float(physical_accepted.mean()),
            "done_mean_rate": float(done.mean()),
            "mean_survival_steps": _first_true_mean(done),
            "mean_physical_survival_steps": _first_true_mean(physical_fail),
            "root_height_min": float(data["root_height"].min()) if "root_height" in data.files else None,
            "torso_abs_roll_max": float(np.abs(data["torso_roll"]).max())
            if "torso_roll" in data.files
            else None,
            "torso_abs_pitch_max": float(np.abs(data["torso_pitch"]).max())
            if "torso_pitch" in data.files
            else None,
            "physical_illegal_contact_rate": float(data["physical_illegal_contact"].mean())
            if "physical_illegal_contact" in data.files
            else None,
            "illegal_contact_force_max": float(data["illegal_contact_force_max"].max())
            if "illegal_contact_force_max" in data.files
            else None,
            "reward_mean": _safe_float(sidecar_data.get("reward_mean")),
            "sidecar_accepted_rate": _safe_float(sidecar_data.get("accepted_rate")),
            "sidecar_physical_accepted_rate": _safe_float(sidecar_data.get("physical_accepted_rate")),
            "sidecar_physical_fall_rate": _safe_float(sidecar_data.get("physical_fall_rate")),
            "sidecar_done_rate": _safe_float(sidecar_data.get("done_rate")),
            "file": str(path),
        }
    return item


def build_summary(root: Path) -> dict[str, Any]:
    items = [
        item
        for path in sorted((root / "rollouts").glob("vae_*_*_*env_*steps_physical.npz"))
        if (item := summarize_rollout(path)) is not None
    ]

    candidate_summaries = []
    for candidate in sorted({item["candidate"] for item in items}):
        rows = [item for item in items if item["candidate"] == candidate]
        worst_reference = min(rows, key=lambda item: item["accepted_final_rate"])
        worst_physical = min(rows, key=lambda item: item["physical_accepted_final_rate"])
        candidate_summaries.append(
            {
                "candidate": candidate,
                "motion_count": len(rows),
                "mean_accepted_final_rate": float(
                    np.mean([row["accepted_final_rate"] for row in rows])
                ),
                "mean_physical_accepted_final_rate": float(
                    np.mean([row["physical_accepted_final_rate"] for row in rows])
                ),
                "mean_survival_steps": float(np.mean([row["mean_survival_steps"] for row in rows])),
                "mean_physical_survival_steps": float(
                    np.mean([row["mean_physical_survival_steps"] for row in rows])
                ),
                "worst_reference_motion": worst_reference["motion"],
                "worst_reference_accepted_final_rate": worst_reference["accepted_final_rate"],
                "worst_physical_motion": worst_physical["motion"],
                "worst_physical_accepted_final_rate": worst_physical[
                    "physical_accepted_final_rate"
                ],
                "items": sorted(rows, key=lambda item: item["motion"]),
            }
        )

    return {
        "status": "ok" if candidate_summaries else "empty",
        "rollout_count": len(items),
        "candidate_count": len(candidate_summaries),
        "best_by_mean_reference_accepted": max(
            candidate_summaries,
            key=lambda row: row["mean_accepted_final_rate"],
            default=None,
        ),
        "best_by_mean_physical_accepted": max(
            candidate_summaries,
            key=lambda row: row["mean_physical_accepted_final_rate"],
            default=None,
        ),
        "candidates": candidate_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
