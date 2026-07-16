"""Shared Stage-2/3 CLI implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from beyondmimic_repro.adapters.isaac.contracts import ISAAC_UNAVAILABLE_MESSAGE, IsaacRunConfig, require_isaac_app_launcher
from beyondmimic_repro.contracts.dagger_dataset import load_dagger_dataset, merge_dagger_rounds
from beyondmimic_repro.contracts.state_latent import load_state_latent_dataset
from beyondmimic_repro.contracts.teacher_assets import load_teacher_map, validate_teacher_assets
from beyondmimic_repro.stage2.datasets.teacher_d0 import BC_WARMSTART_NOTICE
from beyondmimic_repro.stage3.datasets.state_latent_builder import build_from_vae_rollout


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise SystemExit(f"{name} does not exist: {resolved}")
    return resolved


def add_output_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="outputs/stage23", help="Directory for JSON summaries/checkpoints.")
    parser.add_argument("--seed", type=int, default=20260712)


def main_audit_teacher_assets(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit relocatable teacher asset metadata.")
    parser.add_argument("--teacher-map", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--expected-hz", type=float)
    parser.add_argument("--require-files", action="store_true")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    teacher_map = ensure_file(args.teacher_map, "--teacher-map")
    assets = load_teacher_map(teacher_map, data_root=args.data_root, checkpoint_root=args.checkpoint_root)
    errors = validate_teacher_assets(assets, require_files=args.require_files, expected_hz=args.expected_hz)
    payload = {
        "status": "ok" if not errors else "error",
        "teacher_count": len(assets),
        "teachers": {name: asset.to_dict() for name, asset in assets.items()},
        "errors": errors,
        "seed": args.seed,
    }
    write_json(Path(args.output_dir) / "teacher_asset_audit.json", payload)
    print(json.dumps({"status": payload["status"], "teacher_count": len(assets), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


def main_train_vae_bc_warmstart(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Stage-2 VAE D0 BC warm start.")
    parser.add_argument("--teacher-rollout", required=True)
    parser.add_argument("--config", default="configs/stage2/vae_paper.yaml")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.teacher_rollout, "--teacher-rollout")
    print(BC_WARMSTART_NOTICE)
    payload = {
        "status": "dry_run_ok" if args.dry_run else "ready_requires_training_runtime",
        "notice": BC_WARMSTART_NOTICE,
        "teacher_rollout": args.teacher_rollout,
        "config": args.config,
        "device": args.device,
        "seed": args.seed,
    }
    write_json(Path(args.output_dir) / "vae_bc_warmstart_summary.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def main_train_vae_dagger(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update Stage-2 VAE from aggregated DAgger data.")
    parser.add_argument("--dagger-dataset", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--config", default="configs/stage2/dagger_walk1_50hz.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.dagger_dataset, "--dagger-dataset")
    ensure_file(args.vae_checkpoint, "--vae-checkpoint")
    payload, metadata = load_dagger_dataset(args.dagger_dataset)
    summary = {
        "status": "dry_run_ok" if args.dry_run else "ready_requires_training_runtime",
        "schema_version": metadata.schema_version,
        "sample_count": int(next(iter(payload.values())).shape[0]),
        "vae_checkpoint": args.vae_checkpoint,
        "seed": args.seed,
    }
    write_json(Path(args.output_dir) / "vae_dagger_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_merge_dagger_rounds(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge D0/D1/... DAgger rounds.")
    parser.add_argument("--round", dest="rounds", action="append", required=True)
    parser.add_argument("--output", required=True)
    add_output_seed(parser)
    args = parser.parse_args(argv)
    for path in args.rounds:
        ensure_file(path, "--round")
    summary = merge_dagger_rounds(args.rounds, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_eval_vae_offline(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline VAE checkpoint/data contract evaluation.")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.vae_checkpoint, "--vae-checkpoint")
    ensure_file(args.dataset, "--dataset")
    summary = {"status": "contract_ready", "vae_checkpoint": args.vae_checkpoint, "dataset": args.dataset, "device": args.device}
    write_json(Path(args.output_dir) / "vae_offline_eval_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_build_from_vae_rollout(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage-3 state-latent windows from VAE rollout.")
    parser.add_argument("--vae-rollout", required=True)
    parser.add_argument("--output", required=True)
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.vae_rollout, "--vae-rollout")
    summary = build_from_vae_rollout(args.vae_rollout, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_audit_state_latent(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a state-latent dataset schema.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--allow-legacy-teacher-source", action="store_true")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.dataset, "--dataset")
    payload, metadata = load_state_latent_dataset(args.dataset, allow_legacy_teacher_source=args.allow_legacy_teacher_source)
    summary = {"status": "ok", "schema_version": metadata.schema_version, "token_shape": list(payload["tokens"].shape)}
    write_json(Path(args.output_dir) / "state_latent_audit.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_diffusion_train(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Stage-3 state-latent diffusion.")
    parser.add_argument("--state-latent-dataset", required=True)
    parser.add_argument("--config", default="configs/stage3/diffusion_engineering_50hz.yaml")
    parser.add_argument("--prediction-type", choices=["x0", "epsilon"], default="x0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.state_latent_dataset, "--state-latent-dataset")
    load_state_latent_dataset(args.state_latent_dataset)
    summary = {"status": "dry_run_ok" if args.dry_run else "ready_requires_training_runtime", "prediction_type": args.prediction_type}
    write_json(Path(args.output_dir) / "diffusion_train_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_diffusion_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline denoising evaluation.")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--state-latent-dataset", required=True)
    parser.add_argument("--device", default="cpu")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.diffusion_checkpoint, "--diffusion-checkpoint")
    ensure_file(args.state_latent_dataset, "--state-latent-dataset")
    summary = {"status": "contract_ready", "diffusion_checkpoint": args.diffusion_checkpoint}
    write_json(Path(args.output_dir) / "diffusion_eval_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_diffusion_sample(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample unguided Stage-3 trajectories.")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.diffusion_checkpoint, "--diffusion-checkpoint")
    summary = {"status": "contract_ready", "num_samples": args.num_samples, "note": "offline sampler; not Isaac closed-loop validation"}
    write_json(Path(args.output_dir) / "diffusion_sample_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_guidance_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate differentiable guidance costs offline.")
    parser.add_argument("--state-latent-dataset")
    parser.add_argument("--guidance-config", default="configs/stage3/guidance_velocity.yaml")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    if args.state_latent_dataset:
        ensure_file(args.state_latent_dataset, "--state-latent-dataset")
    summary = {"status": "offline_guidance_contract_ready", "guidance_config": args.guidance_config}
    write_json(Path(args.output_dir) / "guidance_eval_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_guidance_sample(argv: list[str] | None = None, *, mode: str) -> int:
    parser = argparse.ArgumentParser(description=f"Sample {mode} guided trajectories offline.")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--guidance-config", required=True)
    parser.add_argument("--device", default="cpu")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    ensure_file(args.diffusion_checkpoint, "--diffusion-checkpoint")
    summary = {"status": "contract_ready", "mode": mode, "guidance_config": args.guidance_config}
    write_json(Path(args.output_dir) / f"guidance_{mode}_sample_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main_collect_dagger_isaac(argv: list[str] | None = None) -> int:
    return main_isaac_entry(argv, entrypoint="collect_dagger_isaac")


def main_isaac_entry(argv: list[str] | None = None, *, entrypoint: str) -> int:
    parser = argparse.ArgumentParser(description=f"Isaac adapter entrypoint: {entrypoint}.")
    parser.add_argument("--task-name", default="Tracking-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-map")
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--diffusion-checkpoint")
    parser.add_argument("--motion-file")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Parse args/config only; do not launch Isaac.")
    add_output_seed(parser)
    args = parser.parse_args(argv)
    for label in ["teacher_map", "vae_checkpoint", "diffusion_checkpoint", "motion_file"]:
        value = getattr(args, label)
        if value:
            ensure_file(value, f"--{label.replace('_', '-')}")
    config = IsaacRunConfig(
        task_name=args.task_name,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        teacher_map=Path(args.teacher_map) if args.teacher_map else None,
        vae_checkpoint=Path(args.vae_checkpoint) if args.vae_checkpoint else None,
        diffusion_checkpoint=Path(args.diffusion_checkpoint) if args.diffusion_checkpoint else None,
        output_dir=Path(args.output_dir),
    )
    if args.dry_run:
        summary = {"status": "dry_run_ok", "entrypoint": entrypoint, "config": str(config)}
        write_json(Path(args.output_dir) / f"{entrypoint}_summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return 0
    try:
        require_isaac_app_launcher()
    except RuntimeError as exc:
        raise SystemExit(str(exc) or ISAAC_UNAVAILABLE_MESSAGE) from exc
    summary = {
        "status": "isaac_runtime_available",
        "entrypoint": entrypoint,
        "validation": "requires Isaac Sim / Isaac Lab runtime validation",
    }
    write_json(Path(args.output_dir) / f"{entrypoint}_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0
