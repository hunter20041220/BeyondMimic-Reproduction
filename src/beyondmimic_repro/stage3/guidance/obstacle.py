"""Obstacle SDF guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def relaxed_barrier(distance: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Differentiable relaxed -log SDF barrier."""
    d = torch.as_tensor(delta, dtype=distance.dtype, device=distance.device)
    safe = -torch.log(torch.clamp(distance, min=delta))
    relaxed = -torch.log(d) + 0.5 * (((distance - 2.0 * d) / d) ** 2 - 1.0)
    return torch.where(distance >= d, safe, relaxed)


def obstacle_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SDF + relaxed barrier over planar root path."""
    start_index = int(context.get("cost_start_index", 0))
    position_slice = context.get("position_slice", (0, 2))
    pos_start, pos_end = position_slice
    xy = predicted_trajectory[:, start_index:, int(pos_start) : int(pos_end)]
    center = torch.as_tensor(context["obstacle_xy"], dtype=xy.dtype, device=xy.device)
    if center.ndim == 1:
        center = center.view(1, 1, 2)
    elif center.ndim == 2:
        center = center[:, None, :]
    radius = float(context.get("radius", 0.2))
    delta = float(context.get("delta", 0.1))
    distance = torch.linalg.norm(xy - center, dim=-1) - radius
    cost = relaxed_barrier(distance, delta=delta).mean(dim=-1)
    return cost, {"min_sdf": distance.min(dim=-1).values}
