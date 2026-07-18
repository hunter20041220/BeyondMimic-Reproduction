"""Joystick velocity guidance."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


def joystick_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Planar root velocity tracking across the controllable horizon."""
    start_index = int(context.get("cost_start_index", 0))
    velocity_slice = context.get("velocity_slice", (9, 11))
    if velocity_slice is None:
        xy = predicted_trajectory[:, start_index:, :2]
        dt = float(context.get("dt", 1.0))
        velocity = (xy[:, 1:] - xy[:, :-1]) / dt
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
    if bool(context.get("speed_only", False)):
        speed = torch.linalg.norm(velocity, dim=-1)
        if "target_speed" in context:
            target_speed = torch.as_tensor(context["target_speed"], dtype=velocity.dtype, device=velocity.device)
        else:
            target_velocity = torch.as_tensor(context["target_velocity_xy"], dtype=velocity.dtype, device=velocity.device)
            target_speed = torch.linalg.norm(target_velocity, dim=-1)
        if target_speed.ndim == 0:
            target_speed = target_speed.view(1, 1)
        elif target_speed.ndim == 1:
            target_speed = target_speed.view(1, -1) if target_speed.shape[0] == speed.shape[1] else target_speed[:, None]
        diff_speed = speed - target_speed
        weights = torch.as_tensor(
            context.get("horizon_weights", torch.ones(diff_speed.shape[1], device=velocity.device)),
            dtype=velocity.dtype,
            device=velocity.device,
        )
        cost = (diff_speed.square() * weights.view(1, -1)).mean(dim=-1)
        return cost, {"speed_error_mean": diff_speed.mean(dim=1), "speed_mean": speed.mean(dim=1)}

    target = torch.as_tensor(context["target_velocity_xy"], dtype=velocity.dtype, device=velocity.device)
    if target.ndim == 1:
        target = target.view(1, 1, 2)
    elif target.ndim == 2:
        target = target[:, None, :]
    diff = velocity - target
    weights = torch.as_tensor(
        context.get("horizon_weights", torch.ones(diff.shape[1], device=velocity.device)),
        dtype=velocity.dtype,
        device=velocity.device,
    )
    cost = (diff.square().sum(dim=-1) * weights.view(1, -1)).mean(dim=-1)
    return cost, {"velocity_error_mean": diff.mean(dim=1)}


def turn_rate_guidance_cost(predicted_trajectory: torch.Tensor, context: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Root yaw-rate tracking across the controllable horizon."""
    start_index = int(context.get("cost_start_index", 0))
    angular_index = context.get("angular_velocity_z_index", 14)
    if angular_index is None:
        raise ValueError("turn-rate guidance requires angular_velocity_z_index")
    turn_rate = predicted_trajectory[:, start_index:, int(angular_index)]
    target = torch.as_tensor(context["target_turn_rate_z"], dtype=turn_rate.dtype, device=turn_rate.device)
    if target.ndim == 0:
        target = target.view(1, 1)
    elif target.ndim == 1:
        target = target.view(1, -1) if target.shape[0] == turn_rate.shape[1] else target[:, None]
    diff = turn_rate - target
    weights = torch.as_tensor(
        context.get("horizon_weights", torch.ones(diff.shape[1], device=turn_rate.device)),
        dtype=turn_rate.dtype,
        device=turn_rate.device,
    )
    cost = (diff.square() * weights.view(1, -1)).mean(dim=-1)
    return cost, {"turn_rate_error_mean": diff.mean(dim=1), "turn_rate_mean": turn_rate.mean(dim=1)}
