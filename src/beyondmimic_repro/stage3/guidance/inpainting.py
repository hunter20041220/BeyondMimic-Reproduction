"""Inpainting/keyframe guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def inpainting_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Selected future state/body/keyframe constraints."""
    target = torch.as_tensor(context["target"], dtype=predicted_trajectory.dtype, device=predicted_trajectory.device)
    mask = torch.as_tensor(context["mask"], dtype=predicted_trajectory.dtype, device=predicted_trajectory.device)
    if target.shape != predicted_trajectory.shape or mask.shape != predicted_trajectory.shape:
        raise ValueError("target and mask must match predicted_trajectory shape")
    diff = (predicted_trajectory - target) * mask
    denom = mask.sum(dim=(1, 2)).clamp(min=1.0)
    cost = diff.square().sum(dim=(1, 2)) / denom
    return cost, {"masked_error": diff.detach()}
