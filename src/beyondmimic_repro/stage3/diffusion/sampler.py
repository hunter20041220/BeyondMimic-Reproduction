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
    predicted = denoiser(noisy_trajectory, diffusion_steps)
    diagnostics: dict[str, torch.Tensor] = {"prior_update_norm": torch.linalg.norm(predicted - noisy_trajectory).detach()}
    if guidance_costs:
        guided = predicted.detach().clone().requires_grad_(True)
        total_cost = torch.zeros(guided.shape[0], dtype=guided.dtype, device=guided.device)
        for fn, weight in zip(guidance_costs, guidance_weights, strict=True):
            cost, diag = fn(guided, conditioning)
            total_cost = total_cost + float(weight) * cost
            diagnostics.update({f"guidance_{len(diagnostics)}_{key}": value.detach() for key, value in diag.items()})
        grad = torch.autograd.grad(total_cost.sum(), guided)[0]
        grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).clamp(min=1e-12)
        scale = torch.clamp(gradient_clip_norm / grad_norm, max=1.0).view(-1, 1, 1)
        predicted = guided - guidance_scale * grad * scale
        diagnostics["guidance_cost"] = total_cost.detach()
        diagnostics["guidance_grad_norm"] = grad_norm.detach()
    if "observed_mask" in masks and "observed_values" in masks:
        mask = masks["observed_mask"].to(dtype=torch.bool, device=predicted.device)
        observed = masks["observed_values"].to(dtype=predicted.dtype, device=predicted.device)
        predicted = torch.where(mask, observed, predicted)
    return predicted, diagnostics


def extract_current_latent(predicted_tokens: torch.Tensor, state_dim: int, current_index: int = 4) -> torch.Tensor:
    """Return z_current for receding-horizon VAE decoding."""
    if predicted_tokens.ndim != 3:
        raise ValueError("predicted_tokens must be [B,T,D]")
    return predicted_tokens[:, current_index, state_dim:]
