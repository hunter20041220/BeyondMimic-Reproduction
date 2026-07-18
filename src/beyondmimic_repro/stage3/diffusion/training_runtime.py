"""Runtime training loop for Stage-3 state-latent diffusion."""

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
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-3 diffusion training runtime") from exc

from beyondmimic_repro.contracts.state_latent import load_state_latent_dataset
from beyondmimic_repro.stage3.diffusion.ema import ExponentialMovingAverage
from beyondmimic_repro.stage3.diffusion.noising import add_per_token_noise, construct_training_target
from beyondmimic_repro.stage3.diffusion.schedule import cosine_alpha_bar_schedule
from beyondmimic_repro.stage3.models.state_latent_transformer import StateLatentTransformer


def _maybe_writer(log_dir: Path) -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class StateLatentArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        states: np.ndarray,
        latents: np.ndarray,
        tokens: np.ndarray,
        *,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        count = int(tokens.shape[0])
        if max_samples is not None and max_samples < count:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(count, size=max_samples, replace=False))
        else:
            indices = slice(None)
        selected_tokens = np.asarray(tokens[indices], dtype=np.float32)
        selected_states = np.asarray(states[indices], dtype=np.float32)
        selected_latents = np.asarray(latents[indices], dtype=np.float32)
        selected_weight = None if sample_weight is None else np.asarray(sample_weight[indices], dtype=np.float32)
        if normalization_mean is not None and normalization_std is not None:
            mean = np.asarray(normalization_mean, dtype=np.float32)
            std = np.asarray(normalization_std, dtype=np.float32)
            if mean.shape != (selected_tokens.shape[-1],) or std.shape != (selected_tokens.shape[-1],):
                raise ValueError(
                    "normalization mean/std must match token_dim, got "
                    f"{mean.shape}, {std.shape}, token_dim={selected_tokens.shape[-1]}"
                )
            std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
            state_dim = selected_states.shape[-1]
            selected_tokens = (selected_tokens - mean) / std
            selected_states = (selected_states - mean[:state_dim]) / std[:state_dim]
            selected_latents = (selected_latents - mean[state_dim:]) / std[state_dim:]
        self.states = torch.from_numpy(selected_states)
        self.latents = torch.from_numpy(selected_latents)
        self.tokens = torch.from_numpy(selected_tokens)
        self.sample_weight = None if selected_weight is None else torch.from_numpy(np.clip(selected_weight, 1.0e-6, None))

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.latents[idx], self.tokens[idx]


def _split_dataset(dataset: Dataset[Any], validation_fraction: float, seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    if len(dataset) < 2 or validation_fraction <= 0.0:
        return dataset, torch.utils.data.Subset(dataset, [])
    val_count = max(1, int(round(len(dataset) * validation_fraction)))
    val_count = min(val_count, len(dataset) - 1)
    train_count = len(dataset) - val_count
    return random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(seed))


def _dataset_sample_weight(dataset: Dataset[Any]) -> torch.Tensor | None:
    if isinstance(dataset, torch.utils.data.Subset):
        parent_weight = getattr(dataset.dataset, "sample_weight", None)
        if parent_weight is None:
            return None
        return parent_weight[dataset.indices]
    weight = getattr(dataset, "sample_weight", None)
    return weight


def _build_model(token_dim: int, cfg: dict[str, Any]) -> StateLatentTransformer:
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    return StateLatentTransformer(
        token_dim=token_dim,
        sequence_length=int(cfg.get("sequence_length", 21)),
        denoising_steps=int(diffusion_cfg.get("denoising_steps", 20)),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        transformer_layers=int(model_cfg.get("transformer_layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _lr_for_step(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _run_epoch(
    model: StateLatentTransformer,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ema: ExponentialMovingAverage | None,
    alpha_bars: torch.Tensor,
    prediction_type: str,
    denoising_steps: int,
    grad_accum: int,
    autocast_dtype: torch.dtype | None,
    scaler: Any | None,
    max_grad_norm: float,
    base_lr: float,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    state_dim: int,
    loss_reduction: str,
    state_loss_weight: float,
    latent_loss_weight: float,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_state_loss = 0.0
    total_latent_loss = 0.0
    total_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (states, latents, clean_tokens) in enumerate(loader, start=1):
        states = states.to(device=device, non_blocking=True)
        latents = latents.to(device=device, non_blocking=True)
        clean_tokens = clean_tokens.to(device=device, non_blocking=True)
        k_state = torch.randint(0, denoising_steps, states.shape[:2], device=device)
        k_latent = torch.randint(0, denoising_steps, latents.shape[:2], device=device)
        noise_states = torch.randn_like(states)
        noise_latents = torch.randn_like(latents)
        noise_tokens = torch.cat([noise_states, noise_latents], dim=-1)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                noisy_tokens = add_per_token_noise(
                    states,
                    latents,
                    noise_states,
                    noise_latents,
                    k_state,
                    k_latent,
                    alpha_bars,
                )
                diffusion_steps = torch.stack([k_state, k_latent], dim=-1)
                pred = model(noisy_tokens, diffusion_steps)
                target = construct_training_target(clean_tokens, noise_tokens, prediction_type=prediction_type)
                sq_error = (pred - target) ** 2
                state_loss = torch.mean(sq_error[..., :state_dim])
                latent_loss = torch.mean(sq_error[..., state_dim:])
                if loss_reduction == "token_mean":
                    loss = torch.mean(sq_error)
                elif loss_reduction == "group_mean":
                    loss = float(state_loss_weight) * state_loss + float(latent_loss_weight) * latent_loss
                else:
                    raise ValueError("loss.reduction must be 'token_mean' or 'group_mean'")
                scaled_loss = loss / max(1, grad_accum)
            if not torch.isfinite(loss):
                raise FloatingPointError("diffusion loss became NaN or Inf")
            if training:
                lr = _lr_for_step(base_lr, global_step, total_steps, warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
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
                    if ema is not None:
                        ema.update(model)
                    global_step += 1
        batch = int(clean_tokens.shape[0])
        total_loss += float(loss.detach().cpu()) * batch
        total_state_loss += float(state_loss.detach().cpu()) * batch
        total_latent_loss += float(latent_loss.detach().cpu()) * batch
        total_count += batch
    count = max(1, total_count)
    return {
        "loss": total_loss / count,
        "state_loss": total_state_loss / count,
        "latent_loss": total_latent_loss / count,
    }, global_step


def _save_checkpoint(
    path: Path,
    *,
    model: StateLatentTransformer,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    epoch: int,
    global_step: int,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
            "metadata": metadata,
            "best_validation_loss": best_validation_loss,
        },
        tmp,
    )
    tmp.replace(path)


def train_state_latent_diffusion_runtime(
    *,
    dataset_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    prediction_type: str,
    device: str,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    max_samples: int | None = None,
    validation_fraction: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(config_path)
    payload, dataset_metadata = load_state_latent_dataset(dataset_path)
    states = np.asarray(payload["states"], dtype=np.float32)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    tokens = np.asarray(payload["tokens"], dtype=np.float32)
    if not np.all(np.isfinite(tokens)):
        raise ValueError("state-latent tokens contain NaN or Inf")
    sample_weight = np.asarray(payload["sample_weight"], dtype=np.float32) if "sample_weight" in payload else None
    if sample_weight is not None:
        if sample_weight.shape != (tokens.shape[0],):
            raise ValueError(f"sample_weight must have shape {(tokens.shape[0],)}, got {sample_weight.shape}")
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0.0):
            raise ValueError("sample_weight must be finite and positive")
    training_cfg = cfg.get("training", {})
    diffusion_cfg = cfg.get("diffusion", {})
    ema_cfg = cfg.get("ema", {})
    normalization_cfg = cfg.get("normalization", {})
    loss_cfg = cfg.get("loss", {})
    prediction_type = prediction_type or str(cfg.get("prediction_type", "x0"))
    epochs = int(epochs if epochs is not None else training_cfg.get("epochs", 1))
    batch_size = int(
        batch_size
        if batch_size is not None
        else training_cfg.get("per_device_batch_size", training_cfg.get("batch_size", 128))
    )
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-3))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    warmup_steps = int(training_cfg.get("warmup_steps", training_cfg.get("warmup", 10_000)))
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()
    use_sample_weight = bool(training_cfg.get("use_sample_weight", sample_weight is not None))
    resume_from_ema = bool(training_cfg.get("resume_from_ema", False))
    reset_optimizer_on_resume = bool(training_cfg.get("reset_optimizer_on_resume", False))
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 50))
    early_min_delta_relative = float(early_cfg.get("min_delta_relative", 0.001))
    checkpoint_cfg = cfg.get("checkpoint", {})
    milestone_epochs = {int(v) for v in checkpoint_cfg.get("milestone_epochs", [])}
    torch_device = torch.device(device)
    autocast_dtype: torch.dtype | None = None
    if torch_device.type == "cuda" and mixed_precision in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif torch_device.type == "cuda" and mixed_precision in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda" and autocast_dtype == torch.float16)
    denoising_steps = int(diffusion_cfg.get("denoising_steps", 20))
    alpha_bars = cosine_alpha_bar_schedule(denoising_steps, device=torch_device)
    normalization_enabled = bool(normalization_cfg.get("enabled", False))
    normalization_mean: np.ndarray | None = None
    normalization_std: np.ndarray | None = None
    if normalization_enabled:
        if "normalization_mean" not in payload or "normalization_std" not in payload:
            raise ValueError("normalization.enabled requires dataset normalization_mean and normalization_std")
        normalization_mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
        normalization_std = np.asarray(payload["normalization_std"], dtype=np.float32)
        if normalization_mean.shape != (tokens.shape[-1],) or normalization_std.shape != (tokens.shape[-1],):
            raise ValueError(
                "dataset normalization_mean/std must have shape "
                f"({tokens.shape[-1]},), got {normalization_mean.shape}, {normalization_std.shape}"
            )
        normalization_std = np.where(normalization_std < 1.0e-6, 1.0, normalization_std).astype(np.float32)
    loss_reduction = str(loss_cfg.get("reduction", "token_mean"))
    state_loss_weight = float(loss_cfg.get("state_weight", 1.0))
    latent_loss_weight = float(loss_cfg.get("latent_weight", 1.0))
    dataset = StateLatentArrayDataset(
        states,
        latents,
        tokens,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        sample_weight=sample_weight,
        max_samples=max_samples,
        seed=seed,
    )
    train_ds, val_ds = _split_dataset(dataset, validation_fraction, seed)
    train_weight = _dataset_sample_weight(train_ds) if use_sample_weight else None
    train_sampler = None
    shuffle_train = True
    if train_weight is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_weight.double(),
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle_train = False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch_device.type == "cuda")
    model = _build_model(token_dim=tokens.shape[-1], cfg=cfg).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    ema = ExponentialMovingAverage(
        model,
        power=float(ema_cfg.get("power", 0.75)),
        max_decay=float(ema_cfg.get("max", ema_cfg.get("max_decay", 0.9999))),
    )
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=torch_device)
        ema_state = checkpoint.get("ema_state_dict", {})
        shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
        if resume_from_ema and shadow:
            model_state = {key: value.to(torch_device) for key, value in checkpoint["model_state_dict"].items()}
            model_state.update({key: value.to(torch_device) for key, value in shadow.items()})
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and not reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            ema.power = float(ema_state.get("power", ema.power))
            ema.max_decay = float(ema_state.get("max_decay", ema.max_decay))
            ema.num_updates = int(ema_state.get("num_updates", ema.num_updates))
            if resume_from_ema and shadow:
                ema.shadow = {k: v.detach().clone().to(torch_device) for k, v in model.state_dict().items() if torch.is_floating_point(v)}
            else:
                ema.shadow = {k: v.to(torch_device) for k, v in ema_state.get("shadow", {}).items()}
        start_epoch = 0 if reset_optimizer_on_resume else int(checkpoint.get("epoch", 0))
        global_step = 0 if reset_optimizer_on_resume else int(checkpoint.get("global_step", 0))
        best_validation_loss = math.inf if reset_optimizer_on_resume else float(checkpoint.get("best_validation_loss", math.inf))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    metadata = {
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "dataset_metadata": {
            "schema_version": dataset_metadata.schema_version,
            "source": dataset_metadata.source,
            "frequency_hz": dataset_metadata.frequency_hz,
            "past_steps": dataset_metadata.past_steps,
            "future_steps": dataset_metadata.future_steps,
            "state_schema": dataset_metadata.state_schema,
            "latent_source": dataset_metadata.latent_source,
        },
        "sample_count": len(dataset),
        "train_count": len(train_ds),
        "validation_count": len(val_ds),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "prediction_type": prediction_type,
        "mixed_precision": mixed_precision,
        "normalization_enabled": normalization_enabled,
        "sample_weight_enabled": bool(train_weight is not None),
        "loss_reduction": loss_reduction,
        "state_loss_weight": state_loss_weight,
        "latent_loss_weight": latent_loss_weight,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_from_ema": resume_from_ema,
        "reset_optimizer_on_resume": reset_optimizer_on_resume,
    }
    if normalization_enabled:
        metadata["normalization_mean"] = normalization_mean.tolist() if normalization_mean is not None else None
        metadata["normalization_std"] = normalization_std.tolist() if normalization_std is not None else None
    writer = _maybe_writer(output / "tensorboard")
    history: list[dict[str, float | int]] = []
    history_path = output / "training_history.jsonl"
    if start_epoch == 0:
        history_path.write_text("", encoding="utf-8")
    started = time.time()
    bad_epochs = 0
    stopped_early = False
    stop_reason: str | None = None
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        train_metrics, global_step = _run_epoch(
            model,
            train_loader,
            device=torch_device,
            optimizer=optimizer,
            ema=ema,
            alpha_bars=alpha_bars,
            prediction_type=prediction_type,
            denoising_steps=denoising_steps,
            grad_accum=grad_accum,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            base_lr=base_lr,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            state_dim=states.shape[-1],
            loss_reduction=loss_reduction,
            state_loss_weight=state_loss_weight,
            latent_loss_weight=latent_loss_weight,
        )
        if len(val_ds):
            val_metrics, _ = _run_epoch(
                model,
                val_loader,
                device=torch_device,
                optimizer=None,
                ema=None,
                alpha_bars=alpha_bars,
                prediction_type=prediction_type,
                denoising_steps=denoising_steps,
                grad_accum=1,
                autocast_dtype=autocast_dtype,
                scaler=None,
                max_grad_norm=max_grad_norm,
                base_lr=base_lr,
                global_step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                state_dim=states.shape[-1],
                loss_reduction=loss_reduction,
                state_loss_weight=state_loss_weight,
                latent_loss_weight=latent_loss_weight,
            )
        else:
            val_metrics = {"loss": float("nan"), "state_loss": float("nan"), "latent_loss": float("nan")}
        row = {
            "elapsed_s": time.time() - started,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_metrics["loss"],
            "train_state_loss": train_metrics["state_loss"],
            "train_latent_loss": train_metrics["latent_loss"],
            "validation_loss": val_metrics["loss"],
            "validation_state_loss": val_metrics["state_loss"],
            "validation_latent_loss": val_metrics["latent_loss"],
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if writer is not None:
            writer.add_scalar("train_loss", train_metrics["loss"], epoch)
            writer.add_scalar("train_state_loss", train_metrics["state_loss"], epoch)
            writer.add_scalar("train_latent_loss", train_metrics["latent_loss"], epoch)
            if math.isfinite(val_metrics["loss"]):
                writer.add_scalar("validation_loss", val_metrics["loss"], epoch)
                writer.add_scalar("validation_state_loss", val_metrics["state_loss"], epoch)
                writer.add_scalar("validation_latent_loss", val_metrics["latent_loss"], epoch)
        _save_checkpoint(
            output / "checkpoints" / "latest.pt",
            model=model,
            optimizer=optimizer,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
            metadata=metadata,
            best_validation_loss=best_validation_loss,
        )
        comparable = val_metrics["loss"] if math.isfinite(val_metrics["loss"]) else train_metrics["loss"]
        previous_best = best_validation_loss
        relative_delta = math.inf
        if math.isfinite(previous_best) and previous_best > 0.0:
            relative_delta = (previous_best - comparable) / previous_best
        improved_for_early_stop = comparable < previous_best and (
            not math.isfinite(previous_best) or relative_delta >= early_min_delta_relative
        )
        if comparable < best_validation_loss:
            best_validation_loss = comparable
            _save_checkpoint(
                output / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                metadata=metadata,
                best_validation_loss=best_validation_loss,
            )
        if epoch in milestone_epochs:
            _save_checkpoint(
                output / "checkpoints" / f"epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                metadata=metadata,
                best_validation_loss=best_validation_loss,
            )
        if early_enabled:
            if improved_for_early_stop:
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= early_patience:
                stopped_early = True
                stop_reason = (
                    f"validation did not improve by relative {early_min_delta_relative:.6g} "
                    f"for {early_patience} epochs"
                )
                print(json.dumps({"epoch": epoch, "status": "early_stopped", "reason": stop_reason}, sort_keys=True), flush=True)
                break
    if writer is not None:
        writer.close()
    summary = {
        "status": "trained",
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "latest_checkpoint": str(output / "checkpoints" / "latest.pt"),
        "best_checkpoint": str(output / "checkpoints" / "best.pt"),
        "metadata": metadata,
        "history": history,
        "elapsed_s": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
