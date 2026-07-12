"""Waypoint guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def waypoint_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distance-dependent final waypoint position/velocity cost."""
    xy = predicted_trajectory[..., :2]
    waypoint = torch.as_tensor(context["waypoint_xy"], dtype=xy.dtype, device=xy.device).view(1, 2)
    final_error = xy[:, -1] - waypoint
    distance = torch.linalg.norm(final_error, dim=-1)
    velocity_weight = float(context.get("velocity_weight", 0.0))
    cost = distance.square()
    if velocity_weight:
        final_velocity = xy[:, -1] - xy[:, -2]
        cost = cost + velocity_weight * final_velocity.square().sum(dim=-1)
    return cost, {"distance": distance}
