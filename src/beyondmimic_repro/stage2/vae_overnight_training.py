"""Structured long-run training for the paper Stage-2 VAE."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-2 VAE long-run training") from exc

from beyondmimic_repro.stage2.models.conditional_action_vae import (
    PaperConditionalActionVAE,
    PaperVAEConfig,
    paper_vae_loss,
)
from beyondmimic_repro.stage2.sagittal_symmetry import mirror_index_and_sign, mirror_vae_arrays
from beyondmimic_repro.stage2.training_runtime import build_vae_arrays_from_teacher_rollout


class StructuredVAEDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, encoder: np.ndarray, proprio: np.ndarray, action: np.ndarray, motion_index: np.ndarray) -> None:
        self.encoder = torch.from_numpy(np.asarray(encoder, dtype=np.float32))
        self.proprio = torch.from_numpy(np.asarray(proprio, dtype=np.float32))
        self.action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        self.motion_index = torch.from_numpy(np.asarray(motion_index, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder[idx], self.proprio[idx], self.action[idx], self.motion_index[idx]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _vae_config_from_yaml(path: str | Path) -> PaperVAEConfig:
    cfg = _load_yaml(path)
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return PaperVAEConfig(
        encoder_input_dim=int(model.get("encoder_input_dim", 67)),
        decoder_proprio_dim=int(model.get("decoder_proprio_dim", 96)),
        action_dim=int(model.get("action_dim", 29)),
        latent_dim=int(model.get("latent_dim", 32)),
        encoder_hidden_dims=tuple(int(v) for v in model.get("encoder_hidden_dims", [2048, 1024, 512])),
        decoder_hidden_dims=tuple(int(v) for v in model.get("decoder_hidden_dims", [2048, 1024, 512])),
        activation=str(model.get("activation", "ELU")),
        learning_rate=float(training.get("learning_rate", 5e-4)),
        kl_coefficient=float(training.get("kl_coefficient", 0.01)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 15)),
        joint_position_semantics=str(data.get("joint_position_semantics", "relative_to_default")),
    )


def _motion_name_from_path(path: Path) -> str:
    stem = path.stem
    for marker in ["_model", "_teacher_rollout"]:
        if marker in stem:
            return stem.split(marker)[0]
    return path.parent.name.split("_model")[0]


def _flatten_selection(arr: np.ndarray, env_ids: np.ndarray) -> np.ndarray:
    selected = np.asarray(arr[:, env_ids], dtype=np.float32)
    return selected.reshape(selected.shape[0] * selected.shape[1], selected.shape[-1])


def _load_joint_names(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        if "joint_names" not in data.files:
            raise ValueError(f"teacher rollout lacks joint_names needed for sagittal symmetry: {path}")
        return [str(value) for value in data["joint_names"].tolist()]


def _load_motion_time_step(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if "motion_time_step" not in data.files:
            return None
        return np.asarray(data["motion_time_step"], dtype=np.int64)


def _load_manifest_reference_frames(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"asset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: dict[str, int] = {}
    for row in manifest.get("teachers", []):
        name = row.get("motion_name")
        reference_frames = row.get("reference_frames")
        if name is not None and reference_frames is not None:
            frames[str(name)] = int(reference_frames)
    return frames


def _flatten_phase_bins(
    motion_time_step: np.ndarray | None,
    env_ids: np.ndarray,
    *,
    reference_frames: int | None,
    phase_bins: int,
    sample_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if phase_bins <= 0:
        raise ValueError(f"phase_bins must be positive, got {phase_bins}")
    if motion_time_step is None:
        return np.zeros((sample_count,), dtype=np.int64), {
            "phase_source": "missing_motion_time_step",
            "reference_frames": reference_frames,
        }
    selected = np.asarray(motion_time_step[:, env_ids], dtype=np.int64)
    flat_steps = selected.reshape(selected.shape[0] * selected.shape[1])
    if flat_steps.shape[0] != sample_count:
        raise ValueError(f"motion_time_step sample count mismatch: {flat_steps.shape[0]} vs {sample_count}")
    if reference_frames is None or reference_frames <= 0:
        reference_frames = int(np.max(flat_steps)) + 1 if flat_steps.size else 1
        phase_source = "inferred_from_motion_time_step"
    else:
        phase_source = "asset_manifest_reference_frames"
    phase = (flat_steps % int(reference_frames)).astype(np.float64) / float(reference_frames)
    bins = np.floor(phase * phase_bins).astype(np.int64)
    return np.clip(bins, 0, phase_bins - 1), {
        "phase_source": phase_source,
        "reference_frames": int(reference_frames),
        "phase_bins": int(phase_bins),
    }


def _balanced_sample_weights(motion_index: np.ndarray, phase_bin: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    motion_index = np.asarray(motion_index, dtype=np.int64)
    phase_bin = np.asarray(phase_bin, dtype=np.int64)
    if motion_index.shape != phase_bin.shape:
        raise ValueError(f"motion_index/phase_bin shape mismatch: {motion_index.shape} vs {phase_bin.shape}")
    pair = np.stack([motion_index, phase_bin], axis=1)
    unique, inverse, counts = np.unique(pair, axis=0, return_inverse=True, return_counts=True)
    weights = (1.0 / counts[inverse].astype(np.float64)).astype(np.float64)
    weights *= float(len(weights)) / float(weights.sum())
    stats = {
        "balanced_unit": "motion_name + reference_phase_bin",
        "nonempty_motion_phase_bins": int(unique.shape[0]),
        "min_bin_count": int(counts.min()) if counts.size else 0,
        "max_bin_count": int(counts.max()) if counts.size else 0,
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
    }
    return weights, stats


def _build_env_rollout_split(
    rollout_paths: list[Path],
    *,
    validation_fraction: float,
    sagittal_symmetry: bool,
    reference_frames_by_motion: dict[str, int] | None = None,
    motion_phase_balanced: bool = False,
    phase_bins: int = 10,
) -> tuple[StructuredVAEDataset, StructuredVAEDataset, list[str], dict[str, Any], np.ndarray | None]:
    train_parts: dict[str, list[np.ndarray]] = {
        "encoder": [],
        "proprio": [],
        "action": [],
        "motion_index": [],
        "phase_bin": [],
    }
    val_parts: dict[str, list[np.ndarray]] = {"encoder": [], "proprio": [], "action": [], "motion_index": []}
    motion_names: list[str] = []
    shards: list[dict[str, Any]] = []
    reference_frames_by_motion = reference_frames_by_motion or {}
    for motion_idx, rollout in enumerate(rollout_paths):
        arrays, metadata = build_vae_arrays_from_teacher_rollout(rollout, flatten=False)
        encoder = arrays["encoder_reference_input"]
        proprio = arrays["decoder_proprio_input"]
        action = arrays["teacher_action"]
        if encoder.ndim != 3:
            raise ValueError(f"expected rollout arrays [T,E,D] for {rollout}, got {encoder.shape}")
        if encoder.shape[:2] != proprio.shape[:2] or encoder.shape[:2] != action.shape[:2]:
            raise ValueError(f"leading [T,E] mismatch for {rollout}: {encoder.shape}, {proprio.shape}, {action.shape}")
        time_steps, env_count = encoder.shape[:2]
        val_env_count = max(1, int(round(env_count * validation_fraction)))
        val_env_count = min(val_env_count, env_count - 1)
        train_envs = np.arange(0, env_count - val_env_count, dtype=np.int64)
        val_envs = np.arange(env_count - val_env_count, env_count, dtype=np.int64)
        motion_name = _motion_name_from_path(rollout)
        motion_names.append(motion_name)
        motion_time_step = _load_motion_time_step(rollout) if motion_phase_balanced else None
        reference_frames = reference_frames_by_motion.get(motion_name)
        train_phase_info: dict[str, Any] = {}
        mirror_source: np.ndarray | None = None
        mirror_signs: np.ndarray | None = None
        if sagittal_symmetry:
            mirror_source, mirror_signs = mirror_index_and_sign(_load_joint_names(rollout))
        for target, env_ids in [(train_parts, train_envs), (val_parts, val_envs)]:
            flat_encoder = _flatten_selection(encoder, env_ids)
            flat_proprio = _flatten_selection(proprio, env_ids)
            flat_action = _flatten_selection(action, env_ids)
            if sagittal_symmetry:
                assert mirror_source is not None and mirror_signs is not None
                mirrored_encoder, mirrored_proprio, mirrored_action = mirror_vae_arrays(
                    flat_encoder,
                    flat_proprio,
                    flat_action,
                    source=mirror_source,
                    signs=mirror_signs,
                )
                flat_encoder = np.concatenate([flat_encoder, mirrored_encoder], axis=0)
                flat_proprio = np.concatenate([flat_proprio, mirrored_proprio], axis=0)
                flat_action = np.concatenate([flat_action, mirrored_action], axis=0)
            count = int(flat_action.shape[0])
            target["encoder"].append(flat_encoder)
            target["proprio"].append(flat_proprio)
            target["action"].append(flat_action)
            target["motion_index"].append(np.full((count,), motion_idx, dtype=np.int64))
            if target is train_parts:
                phase_bin, train_phase_info = _flatten_phase_bins(
                    motion_time_step,
                    env_ids,
                    reference_frames=reference_frames,
                    phase_bins=phase_bins,
                    sample_count=count // (2 if sagittal_symmetry else 1),
                )
                if sagittal_symmetry:
                    phase_bin = np.concatenate([phase_bin, phase_bin], axis=0)
                target["phase_bin"].append(phase_bin)
        shards.append(
            {
                "rollout": str(rollout),
                "motion_name": motion_name,
                "time_steps": int(time_steps),
                "environment_count": int(env_count),
                "train_environment_count": int(len(train_envs)),
                "validation_environment_count": int(len(val_envs)),
                "sagittal_symmetry": bool(sagittal_symmetry),
                "samples_per_split_multiplier": 2 if sagittal_symmetry else 1,
                "phase_sampling": train_phase_info,
                "source_metadata": metadata,
            }
        )
    train_motion_index = np.concatenate(train_parts["motion_index"], axis=0)
    train_phase_bin = np.concatenate(train_parts["phase_bin"], axis=0) if train_parts["phase_bin"] else np.zeros_like(train_motion_index)
    train_weights: np.ndarray | None = None
    sampling_metadata: dict[str, Any] = {
        "motion_phase_balanced": bool(motion_phase_balanced),
        "phase_bins": int(phase_bins),
    }
    if motion_phase_balanced:
        train_weights, weight_stats = _balanced_sample_weights(train_motion_index, train_phase_bin)
        sampling_metadata.update(weight_stats)
    train = StructuredVAEDataset(
        np.concatenate(train_parts["encoder"], axis=0),
        np.concatenate(train_parts["proprio"], axis=0),
        np.concatenate(train_parts["action"], axis=0),
        train_motion_index,
    )
    val = StructuredVAEDataset(
        np.concatenate(val_parts["encoder"], axis=0),
        np.concatenate(val_parts["proprio"], axis=0),
        np.concatenate(val_parts["action"], axis=0),
        np.concatenate(val_parts["motion_index"], axis=0),
    )
    metadata = {
        "validation_split": "environment_rollout",
        "validation_fraction": validation_fraction,
        "teacher_rollout_count": len(rollout_paths),
        "teacher_rollouts": [str(path) for path in rollout_paths],
        "sagittal_symmetry": bool(sagittal_symmetry),
        "symmetry_split_policy": "mirror augmentation applied separately after environment-rollout train/validation split",
        "sampling": sampling_metadata,
        "shards": shards,
    }
    return train, val, motion_names, metadata, train_weights


def _maybe_writer(log_dir: Path) -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _finite_or_raise(name: str, value: float, *, stop_on_nan: bool, stop_on_inf: bool) -> None:
    if stop_on_nan and math.isnan(value):
        raise FloatingPointError(f"{name} became NaN")
    if stop_on_inf and math.isinf(value):
        raise FloatingPointError(f"{name} became Inf")


def _run_epoch(
    model: PaperConditionalActionVAE,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_accum: int,
    kl_coefficient: float,
    autocast_dtype: torch.dtype | None,
    scaler: Any | None,
    max_grad_norm: float,
    motion_names: list[str],
    safety: dict[str, bool],
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {"total_loss": 0.0, "action_mse": 0.0, "kl": 0.0}
    total_count = 0
    latent_sum = 0.0
    latent_sq_sum = 0.0
    latent_count = 0
    motion_sums = np.zeros((len(motion_names),), dtype=np.float64)
    motion_counts = np.zeros((len(motion_names),), dtype=np.int64)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (encoder, proprio, action, motion_index) in enumerate(loader, start=1):
        encoder = encoder.to(device=device, non_blocking=True)
        proprio = proprio.to(device=device, non_blocking=True)
        action = action.to(device=device, non_blocking=True)
        motion_index = motion_index.to(device=device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred, mu, logvar, _ = model(encoder, proprio)
                loss, parts = paper_vae_loss(pred, action, mu, logvar, kl_coefficient=kl_coefficient)
                scaled_loss = loss / max(1, grad_accum)
            _finite_or_raise(
                "vae_total_loss",
                float(loss.detach().cpu()),
                stop_on_nan=safety.get("stop_on_nan", True),
                stop_on_inf=safety.get("stop_on_inf", True),
            )
            if training:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if step % grad_accum == 0 or step == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        batch = int(action.shape[0])
        total_count += batch
        totals["total_loss"] += float(loss.detach().cpu()) * batch
        totals["action_mse"] += float(parts["reconstruction_loss"].detach().cpu()) * batch
        totals["kl"] += float(parts["kl_loss"].detach().cpu()) * batch
        latent = mu.detach()
        latent_sum += float(latent.sum().cpu())
        latent_sq_sum += float(latent.square().sum().cpu())
        latent_count += int(latent.numel())
        per_sample_mse = (pred.detach() - action).square().mean(dim=-1)
        for idx in torch.unique(motion_index).detach().cpu().numpy().astype(np.int64):
            mask = motion_index == int(idx)
            motion_sums[idx] += float(per_sample_mse[mask].sum().detach().cpu())
            motion_counts[idx] += int(mask.sum().detach().cpu())
    metrics: dict[str, Any] = {key: value / max(1, total_count) for key, value in totals.items()}
    latent_mean = latent_sum / max(1, latent_count)
    latent_var = max(0.0, latent_sq_sum / max(1, latent_count) - latent_mean * latent_mean)
    metrics["latent_mean"] = latent_mean
    metrics["latent_std"] = math.sqrt(latent_var)
    per_motion = {
        motion_names[i]: float(motion_sums[i] / motion_counts[i])
        for i in range(len(motion_names))
        if motion_counts[i] > 0
    }
    metrics["per_motion_action_mse"] = per_motion
    metrics["worst_motion_action_mse"] = max(per_motion.values()) if per_motion else float("nan")
    metrics["worst_motion"] = max(per_motion, key=per_motion.get) if per_motion else None
    return metrics


def _save_checkpoint(
    path: Path,
    *,
    model: PaperConditionalActionVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: PaperVAEConfig,
    metadata: dict[str, Any],
    best_total_loss: float,
    best_action_mse: float,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config.to_dict(),
            "metadata": metadata,
            "best_total_loss": best_total_loss,
            "best_action_mse": best_action_mse,
            "metrics": row,
        },
        tmp,
    )
    tmp.replace(path)


def _relative_improvement(value: float, best: float, min_delta_relative: float) -> bool:
    if not math.isfinite(best):
        return True
    return value <= best * (1.0 - min_delta_relative)


def _load_history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _best_state_from_rows(
    rows: list[dict[str, Any]],
    *,
    early_metric: str,
    min_delta_relative: float,
) -> tuple[float, float, float, int]:
    best_total_loss = math.inf
    best_action_mse = math.inf
    best_early_metric = math.inf
    no_improve_epochs = 0
    for row in rows:
        if "val_total_loss" in row:
            best_total_loss = min(best_total_loss, float(row["val_total_loss"]))
        if "val_action_mse" in row:
            best_action_mse = min(best_action_mse, float(row["val_action_mse"]))
        if early_metric in row:
            current = float(row[early_metric])
            if _relative_improvement(current, best_early_metric, min_delta_relative):
                best_early_metric = current
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
    return best_total_loss, best_action_mse, best_early_metric, no_improve_epochs


def run_vae_overnight_training(
    *,
    teacher_rollouts: list[str | Path],
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    resume_checkpoint: str | Path | None = None,
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
    rollouts = [Path(path).expanduser() for path in teacher_rollouts]
    reference_frames_by_motion = _load_manifest_reference_frames(data_cfg.get("asset_manifest"))
    train_ds, val_ds, motion_names, split_metadata, train_weights = _build_env_rollout_split(
        rollouts,
        validation_fraction=validation_fraction,
        sagittal_symmetry=sagittal_symmetry,
        reference_frames_by_motion=reference_frames_by_motion,
        motion_phase_balanced=motion_phase_balanced,
        phase_bins=phase_bins,
    )
    train_sampler: WeightedRandomSampler | None = None
    if train_weights is not None:
        samples_per_epoch = int(samples_per_epoch_cfg) if samples_per_epoch_cfg is not None else len(train_ds)
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=samples_per_epoch,
            replacement=True,
        )
        split_metadata["sampling"]["samples_per_epoch"] = int(samples_per_epoch)
        split_metadata["sampling"]["sampler"] = "WeightedRandomSampler(replacement=True)"
    else:
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
    start_epoch = 0
    global_step = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
    metadata = {
        "config_path": str(config_path),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "model_config": vae_config.to_dict(),
        "train_count": len(train_ds),
        "validation_count": len(val_ds),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "mixed_precision": mixed_precision,
        "asset_manifest": str(data_cfg.get("asset_manifest")) if data_cfg.get("asset_manifest") else None,
        **split_metadata,
    }
    writer = _maybe_writer(output / "tensorboard")
    history_path = output / "training_history.jsonl"
    resume_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    latest_path = (output / "checkpoints" / "latest.pt").resolve()
    continue_history = resume_path == latest_path
    if start_epoch == 0 and resume_checkpoint is None and history_path.exists():
        history_path.unlink()
    elif history_path.exists() and not continue_history:
        history_path.unlink()
    history = _load_history_rows(history_path) if continue_history else []
    best_total_loss, best_action_mse, best_early_metric, no_improve_epochs = _best_state_from_rows(
        history,
        early_metric=early_metric,
        min_delta_relative=min_delta_relative,
    )
    started = time.time()

    def write_row(row: dict[str, Any]) -> None:
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if start_epoch > 0 and not history:
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
            "phase": "resume_validation",
            "val_total_loss": val_metrics["total_loss"],
            "val_action_mse": val_metrics["action_mse"],
            "val_kl": val_metrics["kl"],
            "val_latent_mean": val_metrics["latent_mean"],
            "val_latent_std": val_metrics["latent_std"],
            "per_motion_val_action_mse": val_metrics["per_motion_action_mse"],
            "worst_motion_val_action_mse": val_metrics["worst_motion_action_mse"],
            "worst_motion": val_metrics["worst_motion"],
        }
        history.append(row)
        write_row(row)
        best_total_loss = float(row["val_total_loss"])
        best_action_mse = float(row["val_action_mse"])
        best_early_metric = float(row[early_metric])
        _save_checkpoint(
            output / "checkpoints" / "latest.pt",
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
        _save_checkpoint(
            output / "checkpoints" / "best_by_total_loss.pt",
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
        _save_checkpoint(
            output / "checkpoints" / "best_by_action_mse.pt",
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
    for epoch in range(start_epoch + 1, max_epochs + 1):
        if early_enabled and no_improve_epochs >= patience:
            break
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
            "phase": "train_validation",
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
                writer.add_scalar(key, float(row[key]), epoch)
            for motion, value in row["per_motion_val_action_mse"].items():
                writer.add_scalar(f"per_motion_val_action_mse/{motion}", float(value), epoch)
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
        if epoch % save_latest_every == 0:
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
        if epoch in milestone_epochs:
            _save_checkpoint(
                output / "checkpoints" / f"epoch_{epoch}.pt",
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
    parser = argparse.ArgumentParser(description="Long-run paper VAE training with structured validation and early stopping.")
    parser.add_argument("--teacher-rollout", action="append", required=True)
    parser.add_argument("--config", default="configs/stage2/vae_paper_overnight.yaml")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/stage23/vae_overnight")
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(argv)
    for rollout in args.teacher_rollout:
        if not Path(rollout).is_file():
            raise SystemExit(f"--teacher-rollout does not exist: {rollout}")
    if not Path(args.config).is_file():
        raise SystemExit(f"--config does not exist: {args.config}")
    if args.resume_checkpoint and not Path(args.resume_checkpoint).is_file():
        raise SystemExit(f"--resume-checkpoint does not exist: {args.resume_checkpoint}")
    run_vae_overnight_training(
        teacher_rollouts=args.teacher_rollout,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
