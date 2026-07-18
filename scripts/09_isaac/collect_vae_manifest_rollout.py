#!/usr/bin/env python3
"""Collect VAE rollout shards for every motion in a teacher manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PYTHON_SH = Path("/dev/shm/BeyondMimic_Official_Stage1_runtime/envs/isaacsim-4.5.0/python.sh")


def _load_manifest(path: Path, selected: set[str] | None) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = list(manifest.get("teachers", []))
    if selected is not None:
        rows = [row for row in rows if str(row.get("motion_name")) in selected]
    rows = sorted(rows, key=lambda row: str(row.get("motion_name")))
    missing = [
        str(row.get("motion_name"))
        for row in rows
        if not row.get("motion_file") or not Path(str(row["motion_file"])).is_file()
    ]
    if missing:
        raise SystemExit(f"manifest has missing motion_file entries: {missing[:8]}")
    return rows


def _load_summary(output_path: Path, motion_dir: Path) -> dict[str, Any]:
    sidecar = Path(str(output_path)[:-4] + ".json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    fallback = motion_dir / "collect_vae_rollout_summary.json"
    if fallback.is_file():
        return json.loads(fallback.read_text(encoding="utf-8"))
    return {"status": "missing_summary"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_from_summary(motion_id: int, motion_name: str, output_path: Path, summary: dict[str, Any], returncode: int) -> dict[str, Any]:
    return {
        "motion_id": int(motion_id),
        "motion": motion_name,
        "status": summary.get("status", "error" if returncode else "unknown"),
        "returncode": int(returncode),
        "output": str(output_path),
        "accepted_rate": _safe_float(summary.get("accepted_rate")),
        "physical_accepted_rate": _safe_float(summary.get("physical_accepted_rate")),
        "done_rate": _safe_float(summary.get("done_rate")),
        "reward_mean": _safe_float(summary.get("reward_mean")),
        "physical_fall_rate": _safe_float(summary.get("physical_fall_rate")),
        "physical_illegal_contact_rate": _safe_float(summary.get("physical_illegal_contact_rate")),
        "flight_phase_proxy_rate": _safe_float(summary.get("flight_phase_proxy_rate")),
        "root_height_min": _safe_float(summary.get("root_height_min")),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--python-sh", type=Path, default=DEFAULT_PYTHON_SH)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--ou-sigma", type=float, default=0.1)
    parser.add_argument("--ou-theta", type=float, default=0.8)
    parser.add_argument("--ou-mu", type=float, default=0.0)
    parser.add_argument("--ou-noise-seconds", type=float, default=2.5)
    parser.add_argument("--motions", nargs="*")
    parser.add_argument("--physical-only-terminations", action="store_true")
    parser.add_argument("--disable-events", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.python_sh.is_file():
        raise SystemExit(f"python.sh not found: {args.python_sh}")
    if not args.vae_checkpoint.is_file():
        raise SystemExit(f"VAE checkpoint not found: {args.vae_checkpoint}")

    selected = set(args.motions) if args.motions else None
    rows = _load_manifest(args.manifest, selected)
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    results: list[dict[str, Any]] = []
    started = time.time()
    for motion_id, teacher in enumerate(rows):
        motion_name = str(teacher["motion_name"])
        motion_dir = args.output_root / motion_name
        motion_dir.mkdir(parents=True, exist_ok=True)
        output_path = motion_dir / (
            f"{motion_name}_{args.label}_vae_ou_{args.num_envs}env_{args.steps}steps.npz"
        )
        log_path = logs_dir / f"{args.label}_{motion_name}.log"
        if output_path.is_file() and not args.force:
            summary = _load_summary(output_path, motion_dir)
            row = _row_from_summary(motion_id, motion_name, output_path, summary, 0)
            row["skipped_existing"] = True
            results.append(row)
            print(f"[{motion_id + 1:02d}/{len(rows):02d}] skip existing {motion_name}", flush=True)
            continue

        cmd = [
            str(args.python_sh),
            "scripts/09_isaac/collect_vae_rollout.py",
            "--motion-name",
            motion_name,
            "--motion-file",
            str(teacher["motion_file"]),
            "--vae-checkpoint",
            str(args.vae_checkpoint),
            "--output",
            str(output_path),
            "--output-dir",
            str(motion_dir),
            "--num-envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--frequency-hz",
            str(args.frequency_hz),
            "--ou-sigma",
            str(args.ou_sigma),
            "--ou-theta",
            str(args.ou_theta),
            "--ou-mu",
            str(args.ou_mu),
            "--ou-noise-seconds",
            str(args.ou_noise_seconds),
            "--device",
            args.device,
        ]
        if args.physical_only_terminations:
            cmd.append("--physical-only-terminations")
        if args.disable_events:
            cmd.append("--disable-events")

        print(f"[{motion_id + 1:02d}/{len(rows):02d}] collect OU {motion_name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(" ".join(cmd) + "\n")
            log_file.flush()
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], env=env, stdout=log_file, stderr=log_file)
        summary = _load_summary(output_path, motion_dir)
        if proc.returncode != 0:
            summary = {
                "status": "error",
                "returncode": proc.returncode,
                "log": str(log_path),
                **summary,
            }
        row = _row_from_summary(motion_id, motion_name, output_path, summary, proc.returncode)
        row["log"] = str(log_path)
        results.append(row)
        print(
            f"[{motion_id + 1:02d}/{len(rows):02d}] {motion_name} "
            f"status={row['status']} acc={row['accepted_rate']} phys={row['physical_accepted_rate']}",
            flush=True,
        )

    collected = [row for row in results if row.get("status") == "collected"]
    failed = [row for row in results if row.get("status") != "collected"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed and results else "partial" if collected else "failed",
        "label": args.label,
        "manifest": str(args.manifest),
        "vae_checkpoint": str(args.vae_checkpoint),
        "output_root": str(args.output_root),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "frequency_hz": args.frequency_hz,
        "ou_sigma": args.ou_sigma,
        "ou_theta": args.ou_theta,
        "ou_mu": args.ou_mu,
        "ou_noise_seconds": args.ou_noise_seconds,
        "gpu": args.gpu,
        "motion_count": len(results),
        "collected_count": len(collected),
        "failed_count": len(failed),
        "accepted_rate_mean": _mean(collected, "accepted_rate"),
        "physical_accepted_rate_mean": _mean(collected, "physical_accepted_rate"),
        "done_rate_mean": _mean(collected, "done_rate"),
        "reward_mean": _mean(collected, "reward_mean"),
        "elapsed_s": time.time() - started,
        "rows": results,
    }
    if failed:
        summary["failed_motions"] = [row["motion"] for row in failed]
    summary_path = args.output_root / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
