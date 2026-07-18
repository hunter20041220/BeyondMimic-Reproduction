"""Reverse denoising and guided sampling contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guided sampler") from exc


GuidanceCost = Callable[[torch.Tensor, dict[str, Any]], tuple[torch.Tensor, dict[str, torch.Tensor]]]


def guided_sample(
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noisy_trajectory: torch.Tensor,
    diffusion_steps: torch.Tensor,
    conditioning: dict[str, Any] | None = None,
    guidance_costs: list[GuidanceCost] | None = None,
    guidance_weights: list[float] | None = None,
    masks: dict[str, torch.Tensor] | None = None,
    *,
    guidance_scale: float = 1.0,
    gradient_clip_norm: float = 1.0,
    alpha_bars: torch.Tensor | None = None,
    denoising_steps: int | None = None,
    prediction_type: str = "x0",
    initial_noise_scale: float = 1.0,
    token_mean: torch.Tensor | None = None,
    token_std: torch.Tensor | None = None,
    return_denormalized: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run a simple denoising update plus task gradients.

    The model update produces the prior trajectory. Guidance gradients are
    applied to the predicted trajectory, then observed masks are clamped.
    """
    conditioning = conditioning or {}
    guidance_costs = guidance_costs or []
    guidance_weights = guidance_weights or [1.0] * len(guidance_costs)
    masks = masks or {}
    if len(guidance_costs) != len(guidance_weights):
        raise ValueError("guidance_costs and guidance_weights length mismatch")
    if alpha_bars is not None:
        return _guided_ddim_sample(
            denoiser,
            noisy_trajectory,
            diffusion_steps,
            conditioning,
            guidance_costs,
            guidance_weights,
            masks,
            guidance_scale=guidance_scale,
            gradient_clip_norm=gradient_clip_norm,
            alpha_bars=alpha_bars,
            denoising_steps=denoising_steps,
            prediction_type=prediction_type,
            initial_noise_scale=initial_noise_scale,
            token_mean=token_mean,
            token_std=token_std,
            return_denormalized=return_denormalized,
        )
    predicted = denoiser(noisy_trajectory, diffusion_steps)
    diagnostics: dict[str, torch.Tensor] = {"prior_update_norm": torch.linalg.norm(predicted - noisy_trajectory).detach()}
    if guidance_costs:
        guided_input = noisy_trajectory.detach().clone().requires_grad_(True)
        guided_prediction = denoiser(guided_input, diffusion_steps)
        total_cost = torch.zeros(guided_prediction.shape[0], dtype=guided_prediction.dtype, device=guided_prediction.device)
        for fn, weight in zip(guidance_costs, guidance_weights, strict=True):
            cost, diag = fn(guided_prediction, conditioning)
            total_cost = total_cost + float(weight) * cost
            diagnostics.update({f"guidance_{len(diagnostics)}_{key}": value.detach() for key, value in diag.items()})
        grad = torch.autograd.grad(total_cost.sum(), guided_input)[0]
        grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).clamp(min=1e-12)
        scale = torch.clamp(gradient_clip_norm / grad_norm, max=1.0).view(-1, 1, 1)
        guided_input = guided_input - guidance_scale * grad * scale
        predicted = denoiser(guided_input.detach(), diffusion_steps)
        diagnostics["guidance_cost"] = total_cost.detach()
        diagnostics["guidance_grad_norm"] = grad_norm.detach()
    if "observed_mask" in masks and "observed_values" in masks:
        mask = masks["observed_mask"].to(dtype=torch.bool, device=predicted.device)
        observed = masks["observed_values"].to(dtype=predicted.dtype, device=predicted.device)
        predicted = torch.where(mask, observed, predicted)
    return predicted, diagnostics


def _guided_ddim_sample(
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trajectory_template: torch.Tensor,
    diffusion_steps: torch.Tensor,
    conditioning: dict[str, Any],
    guidance_costs: list[GuidanceCost],
    guidance_weights: list[float],
    masks: dict[str, torch.Tensor],
    *,
    guidance_scale: float,
    gradient_clip_norm: float,
    alpha_bars: torch.Tensor,
    denoising_steps: int | None,
    prediction_type: str,
    initial_noise_scale: float,
    token_mean: torch.Tensor | None,
    token_std: torch.Tensor | None,
    return_denormalized: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DDIM-style reverse sampling for x0-trained state-latent denoisers."""
    if prediction_type != "x0":
        raise ValueError("multi-step guided sampling currently supports prediction_type='x0'")
    alpha = alpha_bars.to(device=trajectory_template.device, dtype=trajectory_template.dtype).flatten()
    steps = int(denoising_steps or alpha.shape[0])
    if steps <= 0 or alpha.shape[0] < steps:
        raise ValueError(f"invalid denoising_steps={steps} for alpha_bars shape {tuple(alpha.shape)}")
    if initial_noise_scale < 0.0:
        raise ValueError(f"initial_noise_scale must be non-negative, got {initial_noise_scale}")
    if diffusion_steps.shape != trajectory_template.shape[:2] + (2,):
        raise ValueError(f"diffusion_steps must be [B,T,2], got {tuple(diffusion_steps.shape)}")
    if (token_mean is None) != (token_std is None):
        raise ValueError("token_mean and token_std must be provided together")
    normalizer: tuple[torch.Tensor, torch.Tensor] | None = None
    if token_mean is not None and token_std is not None:
        mean = token_mean.to(device=trajectory_template.device, dtype=trajectory_template.dtype).view(1, 1, -1)
        std = token_std.to(device=trajectory_template.device, dtype=trajectory_template.dtype).view(1, 1, -1)
        if mean.shape[-1] != trajectory_template.shape[-1] or std.shape[-1] != trajectory_template.shape[-1]:
            raise ValueError(
                "token_mean/token_std must match token_dim, got "
                f"{mean.shape[-1]}, {std.shape[-1]}, token_dim={trajectory_template.shape[-1]}"
            )
        normalizer = (mean, std.clamp_min(1.0e-6))

    def to_cost_space(tokens: torch.Tensor) -> torch.Tensor:
        if normalizer is None:
            return tokens
        mean, std = normalizer
        return tokens * std + mean

    observed_mask = masks.get("observed_mask")
    observed_values = masks.get("observed_values")
    observed_step_mask = masks.get("observed_step_mask")
    if observed_step_mask is not None:
        observed_step_mask = observed_step_mask.to(dtype=torch.bool, device=trajectory_template.device)
        if observed_step_mask.shape != trajectory_template.shape[:2] + (2,):
            raise ValueError(
                "observed_step_mask must have shape [B,T,2], got "
                f"{tuple(observed_step_mask.shape)} for trajectory {tuple(trajectory_template.shape)}"
            )
    base_steps = diffusion_steps.to(device=trajectory_template.device, dtype=torch.long).clamp(0, steps - 1)
    initial_sample = trajectory_template + float(initial_noise_scale) * torch.randn_like(trajectory_template)
    if observed_mask is not None and observed_values is not None:
        mask = observed_mask.to(dtype=torch.bool, device=trajectory_template.device)
        observed = observed_values.to(dtype=trajectory_template.dtype, device=trajectory_template.device)
        sample = torch.where(mask, observed, initial_sample)
    else:
        mask = None
        observed = None
        sample = initial_sample

    diagnostics: dict[str, torch.Tensor] = {
        "initial_noise_scale": torch.as_tensor(initial_noise_scale, dtype=trajectory_template.dtype, device=trajectory_template.device)
    }
    last_predicted = sample
    for k in range(steps - 1, -1, -1):
        step_tokens = torch.full(
            sample.shape[:2] + (2,),
            k,
            dtype=torch.long,
            device=sample.device,
        )
        if observed_step_mask is not None:
            step_tokens = torch.where(observed_step_mask, base_steps, step_tokens)
        current = sample.detach().clone().requires_grad_(bool(guidance_costs))
        predicted = denoiser(current, step_tokens)
        if guidance_costs:
            predicted_for_cost = to_cost_space(predicted)
            total_cost = torch.zeros(predicted.shape[0], dtype=predicted.dtype, device=predicted.device)
            for fn, weight in zip(guidance_costs, guidance_weights, strict=True):
                cost, diag = fn(predicted_for_cost, conditioning)
                total_cost = total_cost + float(weight) * cost
                if k == 0:
                    diagnostics.update({f"guidance_{len(diagnostics)}_{key}": value.detach() for key, value in diag.items()})
            grad = torch.autograd.grad(total_cost.sum(), current)[0]
            grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).clamp(min=1e-12)
            scale = torch.clamp(gradient_clip_norm / grad_norm, max=1.0).view(-1, 1, 1)
            guided_current = current - guidance_scale * grad * scale
            predicted = denoiser(guided_current.detach(), step_tokens)
            current_for_update = guided_current.detach()
            diagnostics["guidance_cost"] = total_cost.detach()
            diagnostics["guidance_grad_norm"] = grad_norm.detach()
        else:
            current_for_update = current.detach()

        a_t = alpha[k].clamp(min=1.0e-6, max=1.0)
        if k == 0:
            sample = predicted
        else:
            a_prev = alpha[k - 1].clamp(min=1.0e-6, max=1.0)
            eps = (current_for_update - torch.sqrt(a_t) * predicted) / torch.sqrt((1.0 - a_t).clamp(min=1.0e-6))
            sample = torch.sqrt(a_prev) * predicted + torch.sqrt((1.0 - a_prev).clamp(min=0.0)) * eps
        if mask is not None and observed is not None:
            sample = torch.where(mask, observed, sample)
        last_predicted = predicted.detach()

    diagnostics["prior_update_norm"] = torch.linalg.norm(last_predicted - trajectory_template).detach()
    if return_denormalized and normalizer is not None:
        sample = to_cost_space(sample)
    return sample, diagnostics


def extract_current_latent(predicted_tokens: torch.Tensor, state_dim: int, current_index: int = 4) -> torch.Tensor:
    """Return z_current for receding-horizon VAE decoding."""
    if predicted_tokens.ndim != 3:
        raise ValueError("predicted_tokens must be [B,T,D]")
    return predicted_tokens[:, current_index, state_dim:]
