"""Runtime training loops for Stage-2 VAE distillation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-2 training runtime") from exc

from beyondmimic_repro.contracts.dagger_dataset import load_dagger_dataset
from beyondmimic_repro.contracts.observation import DECODER_PROPRIO_DIM, ENCODER_REFERENCE_DIM
from beyondmimic_repro.stage2.models.conditional_action_vae import (
    PaperConditionalActionVAE,
    PaperVAEConfig,
    paper_vae_loss,
)


def _maybe_writer(log_dir: Path) -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-8, None)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32, copy=False)


def _rot6d_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    matrix = _quat_to_matrix_wxyz(quat)
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6).astype(np.float32, copy=False)


def _rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    matrix = _quat_to_matrix_wxyz(quat)
    return np.einsum("...ji,...j->...i", matrix, vec).astype(np.float32, copy=False)


def _previous_action(actions: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(actions, dtype=np.float32)
    prev[1:] = actions[:-1]
    return prev


def build_vae_arrays_from_teacher_rollout(
    path: str | Path,
    *,
    flatten: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build paper-shaped VAE arrays from a structured teacher rollout NPZ."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        policy_key = "policy_observation" if "policy_observation" in keys else "policy_obs"
        if {"encoder_reference_input", "decoder_proprio_input", "teacher_action"}.issubset(keys):
            arrays = {
                "encoder_reference_input": np.asarray(data["encoder_reference_input"], dtype=np.float32),
                "decoder_proprio_input": np.asarray(data["decoder_proprio_input"], dtype=np.float32),
                "teacher_action": np.asarray(data["teacher_action"], dtype=np.float32),
            }
            source = "precomputed_contract_arrays"
        elif {
            "ref_joint_pos",
            "ref_joint_vel",
            "teacher_action",
            "joint_pos",
            "joint_vel",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "body_pos_w",
            "body_quat_w",
            "ref_anchor_pos_w",
            "ref_anchor_quat_w",
        }.issubset(keys):
            body_names = [str(v) for v in data["body_names"]] if "body_names" in keys else []
            if "torso_link" in body_names:
                anchor_name = "torso_link"
                anchor_index = body_names.index(anchor_name)
            elif "Torso" in body_names:
                anchor_name = "Torso"
                anchor_index = body_names.index(anchor_name)
            elif "pelvis" in body_names:
                anchor_name = "pelvis"
                anchor_index = body_names.index(anchor_name)
            else:
                anchor_name = body_names[0] if body_names else "body_index_0"
                anchor_index = 0
            teacher_action = np.asarray(data["teacher_action"], dtype=np.float32)
            current_anchor_pos = np.asarray(data["body_pos_w"], dtype=np.float32)[..., anchor_index, :]
            current_anchor_quat = np.asarray(data["body_quat_w"], dtype=np.float32)[..., anchor_index, :]
            ref_anchor_pos = np.asarray(data["ref_anchor_pos_w"], dtype=np.float32)
            ref_anchor_quat = np.asarray(data["ref_anchor_quat_w"], dtype=np.float32)
            anchor_pos_error = ref_anchor_pos - current_anchor_pos
            anchor_rot_error = _quat_mul_wxyz(ref_anchor_quat, _quat_conjugate_wxyz(current_anchor_quat))
            encoder = np.concatenate(
                [
                    np.asarray(data["ref_joint_pos"], dtype=np.float32),
                    np.asarray(data["ref_joint_vel"], dtype=np.float32),
                    anchor_pos_error.astype(np.float32, copy=False),
                    _rot6d_from_quat_wxyz(anchor_rot_error),
                ],
                axis=-1,
            )
            root_quat = np.asarray(data["root_quat_w"], dtype=np.float32)
            gravity_w = np.zeros((*root_quat.shape[:-1], 3), dtype=np.float32)
            gravity_w[..., 2] = -1.0
            projected_gravity = _rotate_inverse_wxyz(root_quat, gravity_w)
            root_lin_vel_b = _rotate_inverse_wxyz(root_quat, np.asarray(data["root_lin_vel_w"], dtype=np.float32))
            root_ang_vel_b = _rotate_inverse_wxyz(root_quat, np.asarray(data["root_ang_vel_w"], dtype=np.float32))
            proprio = np.concatenate(
                [
                    projected_gravity,
                    root_lin_vel_b,
                    root_ang_vel_b,
                    np.asarray(data["joint_pos"], dtype=np.float32),
                    np.asarray(data["joint_vel"], dtype=np.float32),
                    _previous_action(teacher_action),
                ],
                axis=-1,
            )
            arrays = {
                "encoder_reference_input": encoder,
                "decoder_proprio_input": proprio,
                "teacher_action": teacher_action,
            }
            source = "structured_rollout_tensors"
        elif policy_key in keys and "teacher_action" in keys:
            obs = np.asarray(data[policy_key], dtype=np.float32)
            if obs.shape[-1] < ENCODER_REFERENCE_DIM + DECODER_PROPRIO_DIM:
                raise ValueError(
                    "teacher rollout lacks structured VAE tensors and policy observation is too short "
                    f"for fallback slicing: {obs.shape}"
                )
            arrays = {
                "encoder_reference_input": obs[..., :ENCODER_REFERENCE_DIM],
                "decoder_proprio_input": obs[..., ENCODER_REFERENCE_DIM : ENCODER_REFERENCE_DIM + DECODER_PROPRIO_DIM],
                "teacher_action": np.asarray(data["teacher_action"], dtype=np.float32),
            }
            source = "policy_observation_fallback_slice"
        else:
            raise ValueError(f"teacher rollout does not contain usable VAE training arrays: {sorted(keys)}")
        metadata = {
            "source": source,
            "input_path": str(path),
            "anchor_body_name": anchor_name if "anchor_name" in locals() else "precomputed_or_fallback",
            "quaternion_convention": "wxyz",
            "rot6d_convention": "first two rotation-matrix columns flattened row-major",
            "decoder_twist_frame": "root/body frame via inverse root quaternion",
            "projected_gravity_frame": "root/body frame",
            "joint_position_semantics": "Isaac rollout joint_pos tensor; verify relative-to-default against official observation manager before claims",
        }
    _validate_training_arrays(arrays)
    if flatten:
        return _flatten_training_arrays(arrays), metadata
    return arrays, metadata


def _validate_training_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc = np.asarray(arrays["encoder_reference_input"], dtype=np.float32)
    pro = np.asarray(arrays["decoder_proprio_input"], dtype=np.float32)
    act = np.asarray(arrays["teacher_action"], dtype=np.float32)
    if enc.shape[-1] != ENCODER_REFERENCE_DIM:
        raise ValueError(f"encoder_reference_input last dim must be {ENCODER_REFERENCE_DIM}, got {enc.shape}")
    if pro.shape[-1] != DECODER_PROPRIO_DIM:
        raise ValueError(f"decoder_proprio_input last dim must be {DECODER_PROPRIO_DIM}, got {pro.shape}")
    if act.shape[-1] != 29:
        raise ValueError(f"teacher_action last dim must be 29, got {act.shape}")
    if enc.shape[:-1] != pro.shape[:-1] or enc.shape[:-1] != act.shape[:-1]:
        raise ValueError(f"VAE training arrays leading shapes mismatch: {enc.shape}, {pro.shape}, {act.shape}")
    for name, value in [("encoder_reference_input", enc), ("decoder_proprio_input", pro), ("teacher_action", act)]:
        _finite(name, value)
    return enc, pro, act


def _flatten_training_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    enc, pro, act = _validate_training_arrays(arrays)
    return {
        "encoder_reference_input": enc.reshape(-1, enc.shape[-1]),
        "decoder_proprio_input": pro.reshape(-1, pro.shape[-1]),
        "teacher_action": act.reshape(-1, act.shape[-1]),
    }


class VAEArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], *, max_samples: int | None = None, seed: int = 0) -> None:
        sample_count = arrays["teacher_action"].shape[0]
        if max_samples is not None and max_samples < sample_count:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(sample_count, size=max_samples, replace=False))
        else:
            indices = slice(None)
        self.encoder = torch.from_numpy(np.asarray(arrays["encoder_reference_input"][indices], dtype=np.float32))
        self.proprio = torch.from_numpy(np.asarray(arrays["decoder_proprio_input"][indices], dtype=np.float32))
        self.action = torch.from_numpy(np.asarray(arrays["teacher_action"][indices], dtype=np.float32))

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder[idx], self.proprio[idx], self.action[idx]


def _vae_config_from_yaml(path: str | Path) -> PaperVAEConfig:
    cfg = _load_yaml(path)
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return PaperVAEConfig(
        encoder_input_dim=int(model.get("encoder_input_dim", ENCODER_REFERENCE_DIM)),
        decoder_proprio_dim=int(model.get("decoder_proprio_dim", DECODER_PROPRIO_DIM)),
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


def _split_dataset(dataset: Dataset[Any], validation_fraction: float, seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    if len(dataset) < 2 or validation_fraction <= 0.0:
        return dataset, torch.utils.data.Subset(dataset, [])
    val_count = max(1, int(round(len(dataset) * validation_fraction)))
    val_count = min(val_count, len(dataset) - 1)
    train_count = len(dataset) - val_count
    return random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(seed))


def _run_epoch(
    model: PaperConditionalActionVAE,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_accum: int,
    kl_coefficient: float,
    autocast_dtype: torch.dtype | None,
    scaler: Any | None,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (encoder, proprio, action) in enumerate(loader, start=1):
        encoder = encoder.to(device=device, non_blocking=True)
        proprio = proprio.to(device=device, non_blocking=True)
        action = action.to(device=device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred, mu, logvar, _ = model(encoder, proprio)
                loss, parts = paper_vae_loss(pred, action, mu, logvar, kl_coefficient=kl_coefficient)
                scaled_loss = loss / max(1, grad_accum)
            if not torch.isfinite(loss):
                raise FloatingPointError("VAE loss became NaN or Inf")
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
        count += batch
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["reconstruction_loss"] += float(parts["reconstruction_loss"].detach().cpu()) * batch
        totals["kl_loss"] += float(parts["kl_loss"].detach().cpu()) * batch
    return {key: value / max(1, count) for key, value in totals.items()}


def _save_checkpoint(
    path: Path,
    *,
    model: PaperConditionalActionVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: PaperVAEConfig,
    metadata: dict[str, Any],
    best_validation_loss: float,
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
            "best_validation_loss": best_validation_loss,
        },
        tmp,
    )
    tmp.replace(path)


def train_vae_runtime(
    *,
    arrays: dict[str, np.ndarray],
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    validation_fraction: float = 0.05,
    max_samples: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(config_path)
    vae_config = _vae_config_from_yaml(config_path)
    training_cfg = cfg.get("training", {})
    epochs = int(epochs if epochs is not None else training_cfg.get("epochs", 1))
    batch_size = int(batch_size if batch_size is not None else training_cfg.get("batch_size", 512))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", vae_config.gradient_accumulation_steps)))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()
    torch_device = torch.device(device)
    autocast_dtype: torch.dtype | None = None
    if torch_device.type == "cuda" and mixed_precision in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif torch_device.type == "cuda" and mixed_precision in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda" and autocast_dtype == torch.float16)
    dataset = VAEArrayDataset(arrays, max_samples=max_samples, seed=seed)
    train_ds, val_ds = _split_dataset(dataset, validation_fraction, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch_device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch_device.type == "cuda")
    model = PaperConditionalActionVAE(vae_config).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=vae_config.learning_rate)
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    runtime_metadata = dict(metadata or {})
    runtime_metadata.update(
        {
            "config_path": str(config_path),
            "sample_count": len(dataset),
            "train_count": len(train_ds),
            "validation_count": len(val_ds),
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "mixed_precision": mixed_precision,
        }
    )
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
    writer = _maybe_writer(output / "tensorboard")
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
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
        )
        global_step += len(train_loader)
        if len(val_ds):
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
            )
        else:
            val_metrics = {"loss": float("nan"), "reconstruction_loss": float("nan"), "kl_loss": float("nan")}
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "validation_loss": val_metrics["loss"],
            "validation_reconstruction_loss": val_metrics["reconstruction_loss"],
            "validation_kl_loss": val_metrics["kl_loss"],
        }
        history.append(row)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch" and math.isfinite(float(value)):
                    writer.add_scalar(key, float(value), epoch)
        _save_checkpoint(
            output / "checkpoints" / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=vae_config,
            metadata=runtime_metadata,
            best_validation_loss=best_validation_loss,
        )
        comparable = val_metrics["loss"] if math.isfinite(val_metrics["loss"]) else train_metrics["loss"]
        if comparable < best_validation_loss:
            best_validation_loss = comparable
            _save_checkpoint(
                output / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=vae_config,
                metadata=runtime_metadata,
                best_validation_loss=best_validation_loss,
            )
    if writer is not None:
        writer.close()
    summary = {
        "status": "trained",
        "epochs_completed": epochs,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "latest_checkpoint": str(output / "checkpoints" / "latest.pt"),
        "best_checkpoint": str(output / "checkpoints" / "best.pt"),
        "model_config": vae_config.to_dict(),
        "metadata": runtime_metadata,
        "history": history,
        "elapsed_s": time.time() - started,
    }
    (output / "training_history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def train_vae_bc_warmstart_runtime(
    *,
    teacher_rollout: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    arrays, metadata = build_vae_arrays_from_teacher_rollout(teacher_rollout)
    metadata.update({"training_source": "D0 teacher rollout BC warm start"})
    return train_vae_runtime(
        arrays=arrays,
        config_path=config_path,
        output_dir=output_dir,
        device=device,
        seed=seed,
        resume_checkpoint=resume_checkpoint,
        batch_size=batch_size,
        epochs=epochs,
        max_samples=max_samples,
        metadata=metadata,
    )


def train_vae_dagger_runtime(
    *,
    dagger_dataset: str | Path,
    vae_checkpoint: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    batch_size: int | None = None,
    epochs: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    payload, metadata = load_dagger_dataset(dagger_dataset)
    arrays = {
        "encoder_reference_input": payload["encoder_reference_input"],
        "decoder_proprio_input": payload["decoder_proprio_input"],
        "teacher_action": payload["teacher_action"],
    }
    runtime_metadata = {
        "training_source": "aggregated DAgger student-state dataset",
        "dagger_dataset": str(dagger_dataset),
        "dagger_metadata": metadata.to_dict(),
    }
    return train_vae_runtime(
        arrays=arrays,
        config_path=config_path,
        output_dir=output_dir,
        device=device,
        seed=seed,
        resume_checkpoint=vae_checkpoint,
        batch_size=batch_size,
        epochs=epochs,
        max_samples=max_samples,
        metadata=runtime_metadata,
    )
