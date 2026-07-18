"""Waypoint guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def waypoint_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper waypoint cost over the controllable horizon."""
    start_index = int(context.get("cost_start_index", 0))
    position_slice = context.get("position_slice", (0, 2))
    pos_start, pos_end = position_slice
    xy = predicted_trajectory[:, start_index:, int(pos_start) : int(pos_end)]
    waypoint = torch.as_tensor(context["waypoint_xy"], dtype=xy.dtype, device=xy.device)
    if waypoint.ndim == 1:
        waypoint = waypoint.view(1, 1, 2)
    elif waypoint.ndim == 2:
        waypoint = waypoint[:, None, :]
    error = xy - waypoint
    distance = torch.linalg.norm(error, dim=-1)
    position_weight = 1.0 - torch.exp(-2.0 * distance)
    velocity_weight = torch.exp(-2.0 * distance)
    velocity_slice = context.get("velocity_slice", (9, 11))
    if velocity_slice is None:
        dt = float(context.get("dt", 1.0))
        velocity = torch.zeros_like(xy)
        if xy.shape[1] > 1:
            velocity[:, 1:] = (xy[:, 1:] - xy[:, :-1]) / dt
    else:
        start, end = velocity_slice
        velocity = predicted_trajectory[:, start_index:, int(start) : int(end)]
    velocity_offset = context.get("velocity_offset_xy")
    if velocity_offset is not None:
        offset = torch.as_tensor(velocity_offset, dtype=velocity.dtype, device=velocity.device)
        if offset.ndim == 1:
            offset = offset.view(1, 1, 2)
        elif offset.ndim == 2:
            offset = offset[:, None, :]
        velocity = velocity + offset
    cost = position_weight * error.square().sum(dim=-1) + velocity_weight * velocity.square().sum(dim=-1)
    cost = cost.mean(dim=-1)
    return cost, {
        "distance": distance[:, -1],
        "mean_distance": distance.mean(dim=-1),
        "mean_velocity_norm": torch.linalg.norm(velocity, dim=-1).mean(dim=-1),
    }
