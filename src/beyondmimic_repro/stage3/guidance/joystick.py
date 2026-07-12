"""Joystick velocity guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def joystick_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Planar root velocity tracking across the future horizon."""
    xy = predicted_trajectory[..., :2]
    dt = float(context.get("dt", 1.0))
    target = torch.as_tensor(context["target_velocity_xy"], dtype=xy.dtype, device=xy.device)
    velocity = (xy[:, 1:] - xy[:, :-1]) / dt
    diff = velocity - target.view(1, 1, 2)
    weights = torch.as_tensor(context.get("horizon_weights", torch.ones(diff.shape[1], device=xy.device)), dtype=xy.dtype, device=xy.device)
    cost = (diff.square().sum(dim=-1) * weights.view(1, -1)).mean(dim=-1)
    return cost, {"velocity_error_mean": diff.mean(dim=1)}
