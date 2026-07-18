#!/usr/bin/env python3
"""Build and merge H8/F32 or H4/F32 state-latent branches from OU rollout summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.contracts.state_latent import StateLatentMetadata
from beyondmimic_repro.stage3.datasets.state_latent_builder import build_from_vae_rollout, merge_state_latent_datasets


def _load_rollout_rows(path: Path) -> list[dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in summary.get("rows", []) if row.get("status") == "collected"]
    missing = [row for row in rows if not Path(str(row.get("output"))).is_file()]
    if missing:
        raise SystemExit(f"summary references missing rollout files: {[row.get('motion') for row in missing[:8]]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--branch", choices=["h8_f32", "h4_f32"], required=True)
    parser.add_argument("--past-steps", type=int, required=True)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--projection-seed", type=int, default=7)
    parser.add_argument("--acceptance-key", default="post_step_physical_accepted")
    parser.add_argument("--episode-acceptance-seconds", type=float, default=5.0)
    parser.add_argument("--phase-bins", type=int, default=10)
    parser.add_argument("--mirror-device", default="cpu")
    parser.add_argument("--chunk-windows", type=int, default=8192)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    rows = _load_rollout_rows(args.rollout_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    shard_rows: list[dict[str, Any]] = []
    metadata = StateLatentMetadata(
        frequency_hz=float(args.frequency_hz),
        past_steps=int(args.past_steps),
        include_current=True,
        future_steps=int(args.future_steps),
    )
    for idx, row in enumerate(rows):
        motion = str(row["motion"])
        motion_id = int(row.get("motion_id", idx))
        shard = shards_dir / f"{motion}_{args.branch}_paper_projected.npz"
        print(f"[{idx + 1:02d}/{len(rows):02d}] build {args.branch} {motion}", flush=True)
        shard_summary = build_from_vae_rollout(
            row["output"],
            shard,
            metadata=metadata,
            motion_id=motion_id,
            target_frequency_hz=float(args.frequency_hz),
            compressed=False,
            state_representation="paper_projected",
            projection_seed=int(args.projection_seed),
            acceptance_key=args.acceptance_key,
            require_full_episode_accepted=True,
            episode_acceptance_seconds=float(args.episode_acceptance_seconds),
        )
        shard_paths.append(shard)
        shard_rows.append({"motion": motion, "motion_id": motion_id, **shard_summary})

    merged_raw = args.output_root / f"state_latent_{args.branch}_paper_projected_merged_raw.npz"
    print(f"[merge] {len(shard_paths)} shards -> {merged_raw}", flush=True)
    merge_summary = merge_state_latent_datasets(shard_paths, merged_raw, compressed=False)

    final_dataset = args.output_root / f"state_latent_{args.branch}_paper_projected_symmetric_weighted.npz"
    final_summary_path = args.output_root / f"state_latent_{args.branch}_paper_projected_symmetric_weighted_summary.json"
    sym_cmd = [
        "python3",
        "scripts/05_state_latent/build_motion_phase_symmetric_state_latent_dataset.py",
        "--input-dataset",
        str(merged_raw),
        "--output",
        str(final_dataset),
        "--summary",
        str(final_summary_path),
        "--phase-bins",
        str(args.phase_bins),
        "--mirror-device",
        args.mirror_device,
        "--chunk-windows",
        str(args.chunk_windows),
    ]
    if args.compressed:
        sym_cmd.append("--compressed")
    print(f"[symmetry] {merged_raw} -> {final_dataset}", flush=True)
    proc = subprocess.run(sym_cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"state-latent symmetry build failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    final_summary = json.loads(proc.stdout.strip().splitlines()[-1])

    branch_summary = {
        "status": "ok",
        "branch": args.branch,
        "past_steps": args.past_steps,
        "future_steps": args.future_steps,
        "sequence_length": args.past_steps + 1 + args.future_steps,
        "frequency_hz": args.frequency_hz,
        "projection_seed": args.projection_seed,
        "acceptance_key": args.acceptance_key,
        "episode_acceptance_seconds": args.episode_acceptance_seconds,
        "rollout_summary": str(args.rollout_summary),
        "shard_count": len(shard_paths),
        "shards": shard_rows,
        "merged_raw": merge_summary,
        "final_dataset": str(final_dataset),
        "final_summary": final_summary,
    }
    output_summary = args.output_root / f"{args.branch}_state_latent_branch_summary.json"
    output_summary.write_text(json.dumps(branch_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in branch_summary.items() if k != "shards"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
