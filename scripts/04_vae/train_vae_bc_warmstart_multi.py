#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.stage2.datasets.teacher_d0 import BC_WARMSTART_NOTICE
from beyondmimic_repro.stage2.training_runtime import build_vae_arrays_from_teacher_rollout, train_vae_runtime


def _ensure_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train paper Stage-2 VAE D0 warm start from multiple teacher rollout shards.")
    parser.add_argument("--teacher-rollout", action="append", required=True)
    parser.add_argument("--config", default="configs/stage2/vae_paper.yaml")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples", type=int, help="Optional cap for smoke tests or memory-limited runs.")
    parser.add_argument("--output-dir", default="outputs/stage23/vae_multi_D0")
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(argv)

    rollout_paths = [_ensure_file(path, "--teacher-rollout") for path in args.teacher_rollout]
    _ensure_file(args.config, "--config")
    if args.resume_checkpoint:
        _ensure_file(args.resume_checkpoint, "--resume-checkpoint")

    print(BC_WARMSTART_NOTICE)
    arrays_by_key: dict[str, list[np.ndarray]] = {
        "encoder_reference_input": [],
        "decoder_proprio_input": [],
        "teacher_action": [],
    }
    shard_metadata = []
    for rollout in rollout_paths:
        arrays, metadata = build_vae_arrays_from_teacher_rollout(rollout)
        shard_metadata.append(metadata)
        for key in arrays_by_key:
            arrays_by_key[key].append(arrays[key])

    merged = {key: np.concatenate(values, axis=0) for key, values in arrays_by_key.items()}
    metadata = {
        "training_source": "multi-motion D0 teacher rollout BC warm start",
        "teacher_rollouts": [str(path) for path in rollout_paths],
        "teacher_rollout_count": len(rollout_paths),
        "shards": shard_metadata,
    }
    summary = train_vae_runtime(
        arrays=merged,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_samples=args.max_samples,
        metadata=metadata,
    )
    payload = {
        "status": summary["status"],
        "teacher_rollout_count": len(rollout_paths),
        "sample_count": summary["metadata"]["sample_count"],
        "config": args.config,
        "device": args.device,
        "latest_checkpoint": summary["latest_checkpoint"],
        "best_checkpoint": summary["best_checkpoint"],
        "best_validation_loss": summary["best_validation_loss"],
        "summary_path": str(Path(args.output_dir) / "summary.json"),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "vae_bc_warmstart_multi_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
