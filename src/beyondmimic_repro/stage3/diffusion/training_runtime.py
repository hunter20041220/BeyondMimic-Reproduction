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
    from torch.utils.data import DataLoader, Dataset, random_split
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
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        count = int(tokens.shape[0])
        if max_samples is not None and max_samples < count:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(count, size=max_samples, replace=False))
        else:
            indices = slice(None)
        self.states = torch.from_numpy(np.asarray(states[indices], dtype=np.float32))
        self.latents = torch.from_numpy(np.asarray(latents[indices], dtype=np.float32))
        self.tokens = torch.from_numpy(np.asarray(tokens[indices], dtype=np.float32))

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
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
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
                loss = torch.mean((pred - target) ** 2)
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
        total_count += batch
    return {"loss": total_loss / max(1, total_count)}, global_step


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
    training_cfg = cfg.get("training", {})
    diffusion_cfg = cfg.get("diffusion", {})
    ema_cfg = cfg.get("ema", {})
    prediction_type = prediction_type or str(cfg.get("prediction_type", "x0"))
    epochs = int(epochs if epochs is not None else training_cfg.get("epochs", 1))
    batch_size = int(batch_size if batch_size is not None else training_cfg.get("per_device_batch_size", 128))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-3))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    warmup_steps = int(training_cfg.get("warmup_steps", training_cfg.get("warmup", 10_000)))
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()
    torch_device = torch.device(device)
    autocast_dtype: torch.dtype | None = None
    if torch_device.type == "cuda" and mixed_precision in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif torch_device.type == "cuda" and mixed_precision in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda" and autocast_dtype == torch.float16)
    denoising_steps = int(diffusion_cfg.get("denoising_steps", 20))
    alpha_bars = cosine_alpha_bar_schedule(denoising_steps, device=torch_device)
    dataset = StateLatentArrayDataset(states, latents, tokens, max_samples=max_samples, seed=seed)
    train_ds, val_ds = _split_dataset(dataset, validation_fraction, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch_device.type == "cuda")
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
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            ema.power = float(ema_state.get("power", ema.power))
            ema.max_decay = float(ema_state.get("max_decay", ema.max_decay))
            ema.num_updates = int(ema_state.get("num_updates", ema.num_updates))
            ema.shadow = {k: v.to(torch_device) for k, v in ema_state.get("shadow", {}).items()}
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
    total_steps = max(1, epochs * len(train_loader))
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
    }
    writer = _maybe_writer(output / "tensorboard")
    history: list[dict[str, float | int]] = []
    started = time.time()
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
            )
        else:
            val_metrics = {"loss": float("nan")}
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "validation_loss": val_metrics["loss"]}
        history.append(row)
        if writer is not None:
            writer.add_scalar("train_loss", train_metrics["loss"], epoch)
            if math.isfinite(val_metrics["loss"]):
                writer.add_scalar("validation_loss", val_metrics["loss"], epoch)
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
    if writer is not None:
        writer.close()
    summary = {
        "status": "trained",
        "epochs_completed": epochs,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "latest_checkpoint": str(output / "checkpoints" / "latest.pt"),
        "best_checkpoint": str(output / "checkpoints" / "best.pt"),
        "metadata": metadata,
        "history": history,
        "elapsed_s": time.time() - started,
    }
    (output / "training_history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
