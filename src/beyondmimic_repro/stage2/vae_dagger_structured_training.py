"""Structured long-run VAE updates from live DAgger datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-2 DAgger VAE training") from exc

from beyondmimic_repro.stage2.sagittal_symmetry import mirror_index_and_sign, mirror_vae_arrays
from beyondmimic_repro.stage2.models.conditional_action_vae import PaperConditionalActionVAE
from beyondmimic_repro.stage2.training_runtime import build_vae_arrays_from_teacher_rollout
from beyondmimic_repro.stage2.vae_overnight_training import (
    StructuredVAEDataset,
    _balanced_sample_weights,
    _best_state_from_rows,
    _flatten_phase_bins,
    _flatten_selection,
    _load_history_rows,
    _load_manifest_reference_frames,
    _load_motion_time_step,
    _load_yaml,
    _maybe_writer,
    _motion_name_from_path,
    _relative_improvement,
    _run_epoch,
    _save_checkpoint,
    _vae_config_from_yaml,
)


class NumpyWeightedRandomSampler(Sampler[int]):
    """Replacement for torch.multinomial when the category count exceeds 2^24."""

    def __init__(self, weights: np.ndarray, *, num_samples: int, seed: int, chunk_size: int = 1_000_000) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"weights must be a non-empty 1D array, got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("weights must be finite and non-negative")
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        self.cdf = np.cumsum(values)
        self.total = float(self.cdf[-1])
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self._iteration = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        remaining = self.num_samples
        while remaining > 0:
            count = min(self.chunk_size, remaining)
            draws = rng.random(count, dtype=np.float64) * self.total
            for index in np.searchsorted(self.cdf, draws, side="right"):
                yield int(index)
            remaining -= count

    def __len__(self) -> int:
        return self.num_samples


def _stable_unique(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_training_fields(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only fields needed for VAE training from a DAgger NPZ.

    The full DAgger schema stores diagnostics and rollout bookkeeping that are
    useful for audits but expensive to decompress before every VAE update.
    """
    required = ["encoder_reference_input", "decoder_proprio_input", "teacher_action"]
    optional = ["motion_name", "environment_id", "reference_frame_index"]
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"DAgger training dataset missing required fields: {missing}")
        payload = {key: np.asarray(data[key]) for key in required}
        for key in optional:
            if key in data.files:
                payload[key] = np.asarray(data[key])
        meta_raw = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
    metadata = json.loads(meta_raw) if meta_raw else {}
    n = int(payload["teacher_action"].shape[0])
    expected_dims = {
        "encoder_reference_input": 67,
        "decoder_proprio_input": 96,
        "teacher_action": 29,
    }
    for key, dim in expected_dims.items():
        value = np.asarray(payload[key])
        if value.ndim != 2 or value.shape != (n, dim):
            raise ValueError(f"{key} must be [{n},{dim}], got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf")
        payload[key] = value.astype(np.float32, copy=False)
    for key in optional:
        if key in payload and payload[key].shape != (n,):
            raise ValueError(f"{key} must be [{n}], got {payload[key].shape}")
    return payload, metadata


def _load_manifest_joint_names(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for row in manifest.get("teachers", []):
        names = row.get("joint_names")
        if names:
            return [str(value) for value in names]
    return []


def _phase_bins_from_reference_index(
    motion_values: np.ndarray,
    reference_index: np.ndarray | None,
    motion_names: list[str],
    reference_frames_by_motion: dict[str, int],
    *,
    phase_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if phase_bins <= 0:
        raise ValueError(f"phase_bins must be positive, got {phase_bins}")
    phase_bin = np.zeros((motion_values.shape[0],), dtype=np.int64)
    phase_sources: dict[str, str] = {}
    if reference_index is None:
        return phase_bin, {"phase_source": "missing_reference_frame_index", "phase_bins": int(phase_bins)}
    reference_index = np.asarray(reference_index, dtype=np.int64)
    for motion in motion_names:
        mask = motion_values == motion
        if not np.any(mask):
            continue
        frames = reference_frames_by_motion.get(motion)
        if frames is None or frames <= 0:
            frames = int(np.max(reference_index[mask])) + 1 if np.any(mask) else 1
            phase_sources[motion] = "inferred_from_reference_frame_index"
        else:
            phase_sources[motion] = "asset_manifest_reference_frames"
        phase = (reference_index[mask] % int(frames)).astype(np.float64) / float(frames)
        bins = np.floor(phase * phase_bins).astype(np.int64)
        phase_bin[mask] = np.clip(bins, 0, phase_bins - 1)
    return phase_bin, {
        "phase_source_by_motion": phase_sources,
        "phase_bins": int(phase_bins),
    }


def _maybe_mirror_split(
    encoder: np.ndarray,
    proprio: np.ndarray,
    action: np.ndarray,
    motion_index: np.ndarray,
    phase_bin: np.ndarray | None,
    *,
    sagittal_symmetry: bool,
    mirror_source: np.ndarray | None,
    mirror_signs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not sagittal_symmetry:
        return encoder, proprio, action, motion_index, phase_bin
    if mirror_source is None or mirror_signs is None:
        raise ValueError("sagittal_symmetry requested but mirror joint mapping is unavailable")
    mirrored_encoder, mirrored_proprio, mirrored_action = mirror_vae_arrays(
        encoder,
        proprio,
        action,
        source=mirror_source,
        signs=mirror_signs,
    )
    encoder = np.concatenate([encoder, mirrored_encoder], axis=0)
    proprio = np.concatenate([proprio, mirrored_proprio], axis=0)
    action = np.concatenate([action, mirrored_action], axis=0)
    motion_index = np.concatenate([motion_index, motion_index], axis=0)
    if phase_bin is not None:
        phase_bin = np.concatenate([phase_bin, phase_bin], axis=0)
    return encoder, proprio, action, motion_index, phase_bin


def _build_motion_env_split(
    dagger_dataset: str | Path,
    *,
    validation_fraction: float,
    reference_frames_by_motion: dict[str, int] | None = None,
    motion_phase_balanced: bool = False,
    phase_bins: int = 10,
    sagittal_symmetry: bool = False,
    mirror_source: np.ndarray | None = None,
    mirror_signs: np.ndarray | None = None,
) -> tuple[StructuredVAEDataset, StructuredVAEDataset, list[str], dict[str, Any], np.ndarray | None, np.ndarray]:
    payload, dagger_meta = _load_training_fields(dagger_dataset)
    n = int(payload["teacher_action"].shape[0])
    motion_values = np.asarray(payload.get("motion_name", np.full((n,), "motion", dtype="<U64"))).astype(str)
    env_values = np.asarray(payload.get("environment_id", np.arange(n, dtype=np.int32)), dtype=np.int64)
    motion_names = _stable_unique(motion_values)
    motion_to_index = {name: idx for idx, name in enumerate(motion_names)}
    val_mask = np.zeros((n,), dtype=bool)
    shards: list[dict[str, Any]] = []
    for motion in motion_names:
        motion_mask = motion_values == motion
        envs = np.unique(env_values[motion_mask])
        if envs.size <= 1:
            indices = np.flatnonzero(motion_mask)
            val_count = max(1, int(round(indices.size * validation_fraction)))
            val_mask[indices[-val_count:]] = True
            shards.append(
                {
                    "motion_name": motion,
                    "sample_count": int(indices.size),
                    "environment_count": int(envs.size),
                    "train_environment_count": int(envs.size),
                    "validation_environment_count": 0,
                    "fallback_split": "contiguous_tail_samples",
                }
            )
            continue
        val_env_count = max(1, int(round(envs.size * validation_fraction)))
        val_env_count = min(val_env_count, int(envs.size) - 1)
        val_envs = envs[-val_env_count:]
        motion_val = motion_mask & np.isin(env_values, val_envs)
        val_mask |= motion_val
        shards.append(
            {
                "motion_name": motion,
                "sample_count": int(motion_mask.sum()),
                "environment_count": int(envs.size),
                "train_environment_count": int(envs.size - val_env_count),
                "validation_environment_count": int(val_env_count),
                "validation_environment_ids": [int(v) for v in val_envs[:10]],
                "validation_environment_id_tail_count": int(max(0, val_envs.size - 10)),
            }
        )
    train_mask = ~val_mask
    motion_index = np.asarray([motion_to_index[name] for name in motion_values], dtype=np.int64)
    phase_bin, phase_metadata = _phase_bins_from_reference_index(
        motion_values,
        np.asarray(payload["reference_frame_index"], dtype=np.int64) if "reference_frame_index" in payload else None,
        motion_names,
        reference_frames_by_motion or {},
        phase_bins=phase_bins,
    )

    def make(mask: np.ndarray) -> tuple[StructuredVAEDataset, np.ndarray]:
        encoder, proprio, action, sample_motion_index, sample_phase_bin = _maybe_mirror_split(
            payload["encoder_reference_input"][mask],
            payload["decoder_proprio_input"][mask],
            payload["teacher_action"][mask],
            motion_index[mask],
            phase_bin[mask],
            sagittal_symmetry=sagittal_symmetry,
            mirror_source=mirror_source,
            mirror_signs=mirror_signs,
        )
        return StructuredVAEDataset(encoder, proprio, action, sample_motion_index), np.asarray(sample_phase_bin, dtype=np.int64)

    train, train_phase_bin = make(train_mask)
    val, _ = make(val_mask)
    train_weights: np.ndarray | None = None
    sampling_metadata: dict[str, Any] = {
        "motion_phase_balanced": bool(motion_phase_balanced),
        "phase_bins": int(phase_bins),
        **phase_metadata,
    }
    if motion_phase_balanced:
        train_weights, weight_stats = _balanced_sample_weights(train.motion_index.numpy(), train_phase_bin)
        sampling_metadata.update(weight_stats)

    metadata = {
        "dagger_dataset": str(dagger_dataset),
        "dagger_metadata": dagger_meta,
        "validation_split": "motion_environment_rollout",
        "validation_fraction": validation_fraction,
        "motion_names": motion_names,
        "sample_count": n,
        "train_count": len(train),
        "validation_count": len(val),
        "sagittal_symmetry": bool(sagittal_symmetry),
        "sampling": sampling_metadata,
        "shards": shards,
    }
    return train, val, motion_names, metadata, train_weights, train_phase_bin


def _dataset_arrays(dataset: StructuredVAEDataset) -> dict[str, np.ndarray]:
    return {
        "encoder": dataset.encoder.numpy(),
        "proprio": dataset.proprio.numpy(),
        "action": dataset.action.numpy(),
        "motion_index": dataset.motion_index.numpy(),
    }


def _build_mixed_d0_dagger_split(
    *,
    teacher_rollouts: list[str | Path],
    dagger_dataset: str | Path,
    validation_fraction: float,
    reference_frames_by_motion: dict[str, int] | None = None,
    motion_phase_balanced: bool = False,
    phase_bins: int = 10,
    sagittal_symmetry: bool = False,
    mirror_source: np.ndarray | None = None,
    mirror_signs: np.ndarray | None = None,
) -> tuple[StructuredVAEDataset, StructuredVAEDataset, list[str], dict[str, Any], np.ndarray | None]:
    d1_train, d1_val, motion_names, d1_metadata, _, d1_train_phase_bin = _build_motion_env_split(
        dagger_dataset,
        validation_fraction=validation_fraction,
        reference_frames_by_motion=reference_frames_by_motion,
        motion_phase_balanced=motion_phase_balanced,
        phase_bins=phase_bins,
        sagittal_symmetry=sagittal_symmetry,
        mirror_source=mirror_source,
        mirror_signs=mirror_signs,
    )
    motion_to_index = {name: idx for idx, name in enumerate(motion_names)}
    train_parts: dict[str, list[np.ndarray]] = {key: [value] for key, value in _dataset_arrays(d1_train).items()}
    train_parts["phase_bin"] = [d1_train_phase_bin]
    val_parts: dict[str, list[np.ndarray]] = {key: [value] for key, value in _dataset_arrays(d1_val).items()}
    d0_shards: list[dict[str, Any]] = []
    reference_frames_by_motion = reference_frames_by_motion or {}

    def motion_index_for(name: str) -> int:
        if name not in motion_to_index:
            motion_to_index[name] = len(motion_names)
            motion_names.append(name)
        return motion_to_index[name]

    for rollout_path in [Path(path).expanduser() for path in teacher_rollouts]:
        arrays, source_metadata = build_vae_arrays_from_teacher_rollout(rollout_path, flatten=False)
        encoder = arrays["encoder_reference_input"]
        proprio = arrays["decoder_proprio_input"]
        action = arrays["teacher_action"]
        if encoder.ndim != 3:
            raise ValueError(f"expected D0 rollout arrays [T,E,D] for {rollout_path}, got {encoder.shape}")
        if encoder.shape[:2] != proprio.shape[:2] or encoder.shape[:2] != action.shape[:2]:
            raise ValueError(f"D0 leading [T,E] mismatch for {rollout_path}: {encoder.shape}, {proprio.shape}, {action.shape}")
        time_steps, env_count = encoder.shape[:2]
        val_env_count = max(1, int(round(env_count * validation_fraction)))
        val_env_count = min(val_env_count, env_count - 1)
        train_envs = np.arange(0, env_count - val_env_count, dtype=np.int64)
        val_envs = np.arange(env_count - val_env_count, env_count, dtype=np.int64)
        motion_name = _motion_name_from_path(rollout_path)
        motion_idx = motion_index_for(motion_name)
        motion_time_step = _load_motion_time_step(rollout_path) if motion_phase_balanced else None
        phase_info: dict[str, Any] = {}
        for target, env_ids in [(train_parts, train_envs), (val_parts, val_envs)]:
            flat_encoder = _flatten_selection(encoder, env_ids)
            flat_proprio = _flatten_selection(proprio, env_ids)
            flat_action = _flatten_selection(action, env_ids)
            count = int(flat_action.shape[0])
            sample_motion_index = np.full((count,), motion_idx, dtype=np.int64)
            sample_phase_bin: np.ndarray | None = None
            if target is train_parts:
                sample_phase_bin, phase_info = _flatten_phase_bins(
                    motion_time_step,
                    env_ids,
                    reference_frames=reference_frames_by_motion.get(motion_name),
                    phase_bins=phase_bins,
                    sample_count=count,
                )
            flat_encoder, flat_proprio, flat_action, sample_motion_index, sample_phase_bin = _maybe_mirror_split(
                flat_encoder,
                flat_proprio,
                flat_action,
                sample_motion_index,
                sample_phase_bin,
                sagittal_symmetry=sagittal_symmetry,
                mirror_source=mirror_source,
                mirror_signs=mirror_signs,
            )
            target["encoder"].append(flat_encoder)
            target["proprio"].append(flat_proprio)
            target["action"].append(flat_action)
            target["motion_index"].append(sample_motion_index)
            if target is train_parts and sample_phase_bin is not None:
                target["phase_bin"].append(sample_phase_bin)
        d0_shards.append(
            {
                "rollout": str(rollout_path),
                "motion_name": motion_name,
                "time_steps": int(time_steps),
                "environment_count": int(env_count),
                "train_environment_count": int(len(train_envs)),
                "validation_environment_count": int(len(val_envs)),
                "phase_sampling": phase_info,
                "source_metadata": source_metadata,
            }
        )

    def make(parts: dict[str, list[np.ndarray]]) -> StructuredVAEDataset:
        return StructuredVAEDataset(
            np.concatenate(parts["encoder"], axis=0),
            np.concatenate(parts["proprio"], axis=0),
            np.concatenate(parts["action"], axis=0),
            np.concatenate(parts["motion_index"], axis=0),
        )

    train = make(train_parts)
    val = make(val_parts)
    train_weights: np.ndarray | None = None
    sampling_metadata: dict[str, Any] = {
        "motion_phase_balanced": bool(motion_phase_balanced),
        "phase_bins": int(phase_bins),
    }
    if motion_phase_balanced:
        train_phase_bin = np.concatenate(train_parts["phase_bin"], axis=0)
        train_weights, weight_stats = _balanced_sample_weights(train.motion_index.numpy(), train_phase_bin)
        sampling_metadata.update(weight_stats)
    metadata = {
        "training_source": "D0 teacher rollouts union D1 student-state DAgger",
        "validation_split": "motion_environment_rollout",
        "validation_fraction": validation_fraction,
        "motion_names": motion_names,
        "teacher_rollouts": [str(path) for path in teacher_rollouts],
        "dagger_dataset": str(dagger_dataset),
        "train_count": len(train),
        "validation_count": len(val),
        "sagittal_symmetry": bool(sagittal_symmetry),
        "sampling": sampling_metadata,
        "d0_shards": d0_shards,
        "d1_metadata": d1_metadata,
    }
    return train, val, motion_names, metadata, train_weights


def run_vae_dagger_structured_training(
    *,
    dagger_dataset: str | Path,
    vae_checkpoint: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    teacher_rollouts: list[str | Path] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _load_yaml(config_path)
    vae_config = _vae_config_from_yaml(config_path)
    training_cfg = cfg.get("training", {})
    validation_cfg = cfg.get("validation", {})
    early_cfg = cfg.get("early_stopping", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    safety_cfg = cfg.get("safety", {})
    augmentation_cfg = cfg.get("augmentation", {})
    sampling_cfg = cfg.get("sampling", {})
    data_cfg = cfg.get("data", {})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 1000)))
    batch_size = int(training_cfg.get("batch_size", 512))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", vae_config.gradient_accumulation_steps)))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    validation_fraction = float(validation_cfg.get("fraction", 0.1))
    sagittal_symmetry = bool(augmentation_cfg.get("sagittal_symmetry", False))
    motion_phase_balanced = bool(sampling_cfg.get("motion_phase_balanced", False))
    phase_bins = int(sampling_cfg.get("phase_bins", 10))
    samples_per_epoch_cfg = sampling_cfg.get("samples_per_epoch")
    early_metric = str(validation_cfg.get("early_stopping_metric", "val_action_mse"))
    patience = int(early_cfg.get("patience", 50))
    min_delta_relative = float(early_cfg.get("min_delta_relative", 0.001))
    early_enabled = bool(early_cfg.get("enabled", True))
    milestone_epochs = {int(v) for v in checkpoint_cfg.get("milestone_epochs", [])}
    save_latest_every = max(1, int(checkpoint_cfg.get("save_latest_every_epochs", 1)))
    torch_device = torch.device(device)
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()
    autocast_dtype: torch.dtype | None = None
    if torch_device.type == "cuda" and mixed_precision in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif torch_device.type == "cuda" and mixed_precision in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda" and autocast_dtype == torch.float16)
    asset_manifest = data_cfg.get("asset_manifest")
    reference_frames_by_motion = _load_manifest_reference_frames(asset_manifest)
    mirror_source: np.ndarray | None = None
    mirror_signs: np.ndarray | None = None
    if sagittal_symmetry:
        joint_names = _load_manifest_joint_names(asset_manifest)
        if not joint_names and teacher_rollouts:
            with np.load(Path(teacher_rollouts[0]).expanduser(), allow_pickle=False) as data:
                if "joint_names" in data.files:
                    joint_names = [str(value) for value in data["joint_names"].tolist()]
        if not joint_names:
            raise ValueError("sagittal_symmetry requested but no joint_names were found in manifest or D0 rollout")
        mirror_source, mirror_signs = mirror_index_and_sign(joint_names)
    if teacher_rollouts:
        train_ds, val_ds, motion_names, split_metadata, train_weights = _build_mixed_d0_dagger_split(
            teacher_rollouts=teacher_rollouts,
            dagger_dataset=dagger_dataset,
            validation_fraction=validation_fraction,
            reference_frames_by_motion=reference_frames_by_motion,
            motion_phase_balanced=motion_phase_balanced,
            phase_bins=phase_bins,
            sagittal_symmetry=sagittal_symmetry,
            mirror_source=mirror_source,
            mirror_signs=mirror_signs,
        )
    else:
        train_ds, val_ds, motion_names, split_metadata, train_weights, _ = _build_motion_env_split(
            dagger_dataset,
            validation_fraction=validation_fraction,
            reference_frames_by_motion=reference_frames_by_motion,
            motion_phase_balanced=motion_phase_balanced,
            phase_bins=phase_bins,
            sagittal_symmetry=sagittal_symmetry,
            mirror_source=mirror_source,
            mirror_signs=mirror_signs,
        )
    train_sampler: Sampler[int] | None = None
    if train_weights is not None:
        samples_per_epoch = int(samples_per_epoch_cfg) if samples_per_epoch_cfg is not None else len(train_ds)
        if len(train_weights) > 2**24:
            train_sampler = NumpyWeightedRandomSampler(train_weights, num_samples=samples_per_epoch, seed=seed)
            sampler_name = "NumpyWeightedRandomSampler(replacement=True)"
        else:
            train_sampler = WeightedRandomSampler(
                torch.as_tensor(train_weights, dtype=torch.double),
                num_samples=samples_per_epoch,
                replacement=True,
            )
            sampler_name = "WeightedRandomSampler(replacement=True)"
        split_metadata["sampling"]["samples_per_epoch"] = int(samples_per_epoch)
        split_metadata["sampling"]["sampler"] = sampler_name
    else:
        split_metadata.setdefault("sampling", {})
        split_metadata["sampling"]["samples_per_epoch"] = int(len(train_ds))
        split_metadata["sampling"]["sampler"] = "shuffle"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch_device.type == "cuda")
    model = PaperConditionalActionVAE(vae_config).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=vae_config.learning_rate)
    resume_path = Path(vae_checkpoint).expanduser().resolve()
    latest_path = (output / "checkpoints" / "latest.pt").resolve()
    continue_history = resume_path == latest_path
    checkpoint = torch.load(vae_checkpoint, map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if continue_history and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) if continue_history else 0
    global_step = int(checkpoint.get("global_step", 0)) if continue_history else 0
    checkpoint_metrics = checkpoint.get("metrics", {})
    start_local_epoch = (
        int(checkpoint_metrics.get("local_epoch", 0))
        if continue_history and isinstance(checkpoint_metrics, dict)
        else 0
    )
    metadata = {
        "config_path": str(config_path),
        "resume_checkpoint": str(vae_checkpoint),
        "model_config": vae_config.to_dict(),
        "train_count": len(train_ds),
        "validation_count": len(val_ds),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "mixed_precision": mixed_precision,
        "asset_manifest": str(asset_manifest) if asset_manifest else None,
        **split_metadata,
    }
    writer = _maybe_writer(output / "tensorboard")
    history_path = output / "training_history.jsonl"
    history = _load_history_rows(history_path) if start_local_epoch > 0 and continue_history else []
    if history:
        start_local_epoch = max(int(row.get("local_epoch", 0)) for row in history)
        start_epoch = max(int(row.get("epoch", start_epoch)) for row in history)
    elif history_path.exists():
        history_path.unlink()
    best_total_loss, best_action_mse, best_early_metric, no_improve_epochs = _best_state_from_rows(
        history,
        early_metric=early_metric,
        min_delta_relative=min_delta_relative,
    )
    started = time.time()

    def write_row(row: dict[str, Any]) -> None:
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if start_local_epoch > 0 and not history:
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=torch_device,
            optimizer=None,
            grad_accum=1,
            kl_coefficient=vae_config.kl_coefficient,
            autocast_dtype=autocast_dtype,
            scaler=None,
            max_grad_norm=max_grad_norm,
            motion_names=motion_names,
            safety=safety_cfg,
        )
        row = {
            "epoch": start_epoch,
            "local_epoch": start_local_epoch,
            "phase": "resume_validation",
            "val_total_loss": val_metrics["total_loss"],
            "val_action_mse": val_metrics["action_mse"],
            "val_kl": val_metrics["kl"],
            "val_latent_mean": val_metrics["latent_mean"],
            "val_latent_std": val_metrics["latent_std"],
            "per_motion_val_action_mse": val_metrics["per_motion_action_mse"],
            "worst_motion_val_action_mse": val_metrics["worst_motion_action_mse"],
            "worst_motion": val_metrics["worst_motion"],
            "global_step": global_step,
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        write_row(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        best_total_loss = float(row["val_total_loss"])
        best_action_mse = float(row["val_action_mse"])
        best_early_metric = float(row[early_metric])
        for name in ["latest.pt", "best_by_total_loss.pt", "best_by_action_mse.pt"]:
            _save_checkpoint(
                output / "checkpoints" / name,
                model=model,
                optimizer=optimizer,
                epoch=start_epoch,
                global_step=global_step,
                config=vae_config,
                metadata=metadata,
                best_total_loss=best_total_loss,
                best_action_mse=best_action_mse,
                row=row,
            )

    stopped_reason = "max_epochs"
    if early_enabled and no_improve_epochs >= patience:
        stopped_reason = f"early_stopping_{early_metric}_patience_{patience}"
    for local_epoch in range(start_local_epoch + 1, max_epochs + 1):
        if early_enabled and no_improve_epochs >= patience:
            break
        epoch = start_epoch + (local_epoch - start_local_epoch)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=torch_device,
            optimizer=optimizer,
            grad_accum=grad_accum,
            kl_coefficient=vae_config.kl_coefficient,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            motion_names=motion_names,
            safety=safety_cfg,
        )
        global_step += len(train_loader)
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=torch_device,
            optimizer=None,
            grad_accum=1,
            kl_coefficient=vae_config.kl_coefficient,
            autocast_dtype=autocast_dtype,
            scaler=None,
            max_grad_norm=max_grad_norm,
            motion_names=motion_names,
            safety=safety_cfg,
        )
        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "phase": "dagger_train_validation",
            "train_total_loss": train_metrics["total_loss"],
            "train_action_mse": train_metrics["action_mse"],
            "train_kl": train_metrics["kl"],
            "train_latent_mean": train_metrics["latent_mean"],
            "train_latent_std": train_metrics["latent_std"],
            "val_total_loss": val_metrics["total_loss"],
            "val_action_mse": val_metrics["action_mse"],
            "val_kl": val_metrics["kl"],
            "val_latent_mean": val_metrics["latent_mean"],
            "val_latent_std": val_metrics["latent_std"],
            "per_motion_val_action_mse": val_metrics["per_motion_action_mse"],
            "worst_motion_val_action_mse": val_metrics["worst_motion_action_mse"],
            "worst_motion": val_metrics["worst_motion"],
            "global_step": global_step,
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        write_row(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if writer is not None:
            for key in [
                "train_total_loss",
                "train_action_mse",
                "train_kl",
                "val_total_loss",
                "val_action_mse",
                "val_kl",
                "worst_motion_val_action_mse",
                "train_latent_mean",
                "train_latent_std",
                "val_latent_mean",
                "val_latent_std",
            ]:
                writer.add_scalar(key, float(row[key]), local_epoch)
            for motion, value in row["per_motion_val_action_mse"].items():
                writer.add_scalar(f"per_motion_val_action_mse/{motion}", float(value), local_epoch)
        if checkpoint_cfg.get("save_best_total_loss", True) and float(row["val_total_loss"]) < best_total_loss:
            best_total_loss = float(row["val_total_loss"])
            _save_checkpoint(
                output / "checkpoints" / "best_by_total_loss.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=vae_config,
                metadata=metadata,
                best_total_loss=best_total_loss,
                best_action_mse=best_action_mse,
                row=row,
            )
        if checkpoint_cfg.get("save_best_action_mse", True) and float(row["val_action_mse"]) < best_action_mse:
            best_action_mse = float(row["val_action_mse"])
            _save_checkpoint(
                output / "checkpoints" / "best_by_action_mse.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=vae_config,
                metadata=metadata,
                best_total_loss=best_total_loss,
                best_action_mse=best_action_mse,
                row=row,
            )
        current_early_metric = float(row[early_metric])
        if _relative_improvement(current_early_metric, best_early_metric, min_delta_relative):
            best_early_metric = current_early_metric
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if local_epoch % save_latest_every == 0:
            _save_checkpoint(
                output / "checkpoints" / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=vae_config,
                metadata=metadata,
                best_total_loss=best_total_loss,
                best_action_mse=best_action_mse,
                row=row,
            )
        if local_epoch in milestone_epochs:
            _save_checkpoint(
                output / "checkpoints" / f"epoch_{local_epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=vae_config,
                metadata=metadata,
                best_total_loss=best_total_loss,
                best_action_mse=best_action_mse,
                row=row,
            )
        if early_enabled and no_improve_epochs >= patience:
            stopped_reason = f"early_stopping_{early_metric}_patience_{patience}"
            break
    if writer is not None:
        writer.close()
    summary = {
        "status": "trained",
        "stopped_reason": stopped_reason,
        "max_epochs": max_epochs,
        "last_epoch": history[-1]["epoch"] if history else start_epoch,
        "last_local_epoch": history[-1]["local_epoch"] if history else 0,
        "best_total_loss": best_total_loss,
        "best_action_mse": best_action_mse,
        "early_stopping_metric": early_metric,
        "best_early_stopping_metric": best_early_metric,
        "no_improve_epochs": no_improve_epochs,
        "latest_checkpoint": str(output / "checkpoints" / "latest.pt"),
        "best_by_total_loss_checkpoint": str(output / "checkpoints" / "best_by_total_loss.pt"),
        "best_by_action_mse_checkpoint": str(output / "checkpoints" / "best_by_action_mse.pt"),
        "history_tail": history[-5:],
        "metadata": metadata,
        "elapsed_s": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the paper VAE on structured multi-motion DAgger data.")
    parser.add_argument("--dagger-dataset", required=True)
    parser.add_argument("--teacher-rollout", action="append", default=[])
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--config", default="configs/stage2/vae_dagger_locomotion_overnight.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/stage23/vae_dagger_structured")
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(argv)
    for label, path in [("--dagger-dataset", args.dagger_dataset), ("--vae-checkpoint", args.vae_checkpoint), ("--config", args.config)]:
        if not Path(path).is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    for rollout in args.teacher_rollout:
        if not Path(rollout).is_file():
            raise SystemExit(f"--teacher-rollout does not exist: {rollout}")
    run_vae_dagger_structured_training(
        dagger_dataset=args.dagger_dataset,
        teacher_rollouts=args.teacher_rollout,
        vae_checkpoint=args.vae_checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
