"""Physical root-velocity guidance for state-latent diffusion rollouts."""

from __future__ import annotations

from typing import Any, Callable

import torch


ROOT_LINEAR_VELOCITY_SLICE_99D = (9, 11)


def smooth_walk_run_walk_velocity(
    step: int,
    total_steps: int,
    walk_velocity_xy: tuple[float, float],
    run_velocity_xy: tuple[float, float],
) -> tuple[float, float]:
    """Cosine walk->run->walk command schedule with fixed phase boundaries."""
    if total_steps <= 1:
        return walk_velocity_xy
    x = min(max(float(step) / float(max(1, total_steps - 1)), 0.0), 1.0)
    if x < 0.30:
        blend = 0.0
    elif x < 0.42:
        u = (x - 0.30) / 0.12
        blend = 0.5 - 0.5 * torch.cos(torch.tensor(u * torch.pi)).item()
    elif x < 0.62:
        blend = 1.0
    elif x < 0.74:
        u = (x - 0.62) / 0.12
        blend = 0.5 + 0.5 * torch.cos(torch.tensor(u * torch.pi)).item()
    else:
        blend = 0.0
    walk = torch.tensor(walk_velocity_xy, dtype=torch.float64)
    run = torch.tensor(run_velocity_xy, dtype=torch.float64)
    velocity = walk + float(blend) * (run - walk)
    return (float(velocity[0]), float(velocity[1]))


def projection_pseudoinverse(projection_matrix: torch.Tensor | None) -> torch.Tensor | None:
    """Return the projected-state pseudoinverse as [99, projected_dim]."""
    if projection_matrix is None:
        return None
    return torch.linalg.pinv(projection_matrix.float())


def physical_root_velocity_xy(
    tokens: torch.Tensor,
    *,
    state_dim: int,
    velocity_slice: tuple[int, int],
    velocity_is_relative: bool,
    current_velocity_xy: torch.Tensor | None,
    projection_inverse: torch.Tensor | None,
) -> torch.Tensor:
    """Extract physical planar root velocity from denormalized diffusion tokens.

    For paper_projected states, this first maps the 163-D projected state back
    to the 99-D yaw-centric paper state and then reads root linear velocity.
    """
    state = tokens[..., :state_dim]
    if projection_inverse is not None:
        inverse = projection_inverse.to(device=state.device, dtype=state.dtype)
        hybrid_state = torch.matmul(state, inverse.T)
        velocity = hybrid_state[..., ROOT_LINEAR_VELOCITY_SLICE_99D[0] : ROOT_LINEAR_VELOCITY_SLICE_99D[1]]
    else:
        start, end = velocity_slice
        velocity = state[..., int(start) : int(end)]
    if velocity_is_relative and current_velocity_xy is not None:
        offset = current_velocity_xy.to(device=velocity.device, dtype=velocity.dtype)
        if offset.ndim == 2:
            offset = offset[:, None, :]
        velocity = velocity + offset
    return velocity


def scheduled_physical_velocity_cost(
    predicted_tokens: torch.Tensor,
    context: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper S5 joystick cost over physical planar root velocity.

    The state stores root velocity relative to the current root velocity in the
    current character-yaw frame.  ``physical_root_velocity_xy`` first recovers
    the 99-D paper state, then adds the current velocity offset so the cost is
    applied to absolute future velocity in that same current-yaw frame.
    """
    start_index = int(context.get("cost_start_index", 0))
    velocity = physical_root_velocity_xy(
        predicted_tokens,
        state_dim=int(context["state_dim"]),
        velocity_slice=tuple(context["velocity_slice"]),
        velocity_is_relative=bool(context.get("velocity_is_relative", False)),
        current_velocity_xy=context.get("current_velocity_xy"),
        projection_inverse=context.get("projection_inverse"),
    )[:, start_index:]
    target = torch.as_tensor(
        context["target_velocity_xy_schedule"],
        dtype=velocity.dtype,
        device=velocity.device,
    )
    if target.ndim == 2:
        target = target[None, :, :]
    target = target[:, : velocity.shape[1], :]
    velocity = velocity[:, : target.shape[1], :]
    weights = torch.as_tensor(
        context.get("horizon_weights", torch.ones(target.shape[1], dtype=velocity.dtype, device=velocity.device)),
        dtype=velocity.dtype,
        device=velocity.device,
    )
    if bool(context.get("speed_only", False)):
        speed = torch.linalg.norm(velocity, dim=-1)
        target_speed = torch.linalg.norm(target, dim=-1)
        diff_speed = speed - target_speed
        cost = 0.5 * (diff_speed.square() * weights.view(1, -1)).sum(dim=-1)
        heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        return cost, {
            "physical_speed_error_mean": diff_speed.mean(dim=1),
            "physical_speed_mean": speed.mean(dim=1),
            "physical_velocity_x_mean": velocity[..., 0].mean(dim=1),
            "physical_velocity_y_mean": velocity[..., 1].mean(dim=1),
            "physical_heading_mean": heading.mean(dim=1),
        }
    diff = velocity - target
    # BeyondMimic supplement S5: G = 0.5 * sum_i ||V_xy,i - g_v||^2.
    cost = 0.5 * (diff.square().sum(dim=-1) * weights.view(1, -1)).sum(dim=-1)
    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    target_heading = torch.atan2(target[..., 1], target[..., 0])
    return cost, {
        "physical_velocity_error_mean": diff.square().sum(dim=-1).mean(dim=1),
        "physical_velocity_error_x_mean": diff[..., 0].mean(dim=1),
        "physical_velocity_error_y_mean": diff[..., 1].mean(dim=1),
        "physical_velocity_x_mean": velocity[..., 0].mean(dim=1),
        "physical_velocity_y_mean": velocity[..., 1].mean(dim=1),
        "physical_speed_mean": torch.linalg.norm(velocity, dim=-1).mean(dim=1),
        "physical_heading_mean": heading.mean(dim=1),
        "physical_target_velocity_x_mean": target[..., 0].mean(dim=1),
        "physical_target_velocity_y_mean": target[..., 1].mean(dim=1),
        "physical_target_speed_mean": torch.linalg.norm(target, dim=-1).mean(dim=1),
        "physical_target_heading_mean": target_heading.mean(dim=1),
    }


def diagnostic_guided_sample(
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trajectory_template: torch.Tensor,
    diffusion_steps: torch.Tensor,
    conditioning: dict[str, Any],
    masks: dict[str, torch.Tensor],
    *,
    guidance_scale: float,
    gradient_clip_norm: float,
    alpha_bars: torch.Tensor,
    denoising_steps: int,
    initial_noise_scale: float,
    token_mean: torch.Tensor | None,
    token_std: torch.Tensor | None,
    return_denormalized: bool,
    current_index: int,
    state_dim: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run same-noise unguided/guided DDIM and expose guidance diagnostics."""
    if (token_mean is None) != (token_std is None):
        raise ValueError("token_mean and token_std must be provided together")
    alpha = alpha_bars.to(device=trajectory_template.device, dtype=trajectory_template.dtype).flatten()
    steps = int(denoising_steps)
    if steps <= 0 or alpha.shape[0] < steps:
        raise ValueError(f"invalid denoising_steps={steps} for alpha_bars shape={tuple(alpha.shape)}")

    normalizer: tuple[torch.Tensor, torch.Tensor] | None = None
    if token_mean is not None and token_std is not None:
        mean = token_mean.to(device=trajectory_template.device, dtype=trajectory_template.dtype).view(1, 1, -1)
        std = token_std.to(device=trajectory_template.device, dtype=trajectory_template.dtype).view(1, 1, -1)
        normalizer = (mean, std.clamp_min(1.0e-6))

    def to_cost_space(tokens: torch.Tensor) -> torch.Tensor:
        if normalizer is None:
            return tokens
        mean, std = normalizer
        return tokens * std + mean

    observed_mask = masks.get("observed_mask")
    observed_values = masks.get("observed_values")
    observed_step_mask = masks.get("observed_step_mask")
    mask = observed_mask.to(dtype=torch.bool, device=trajectory_template.device) if observed_mask is not None else None
    observed = (
        observed_values.to(dtype=trajectory_template.dtype, device=trajectory_template.device)
        if observed_values is not None
        else None
    )
    if observed_step_mask is not None:
        observed_step_mask = observed_step_mask.to(dtype=torch.bool, device=trajectory_template.device)
    base_steps = diffusion_steps.to(device=trajectory_template.device, dtype=torch.long).clamp(0, steps - 1)
    base_noise = torch.randn_like(trajectory_template)

    def run_once(apply_guidance: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = trajectory_template + float(initial_noise_scale) * base_noise
        if mask is not None and observed is not None:
            sample = torch.where(mask, observed, sample)
        diagnostics: dict[str, torch.Tensor] = {}
        last_predicted = sample.detach()
        for k in range(steps - 1, -1, -1):
            step_tokens = torch.full(sample.shape[:2] + (2,), k, dtype=torch.long, device=sample.device)
            if observed_step_mask is not None:
                step_tokens = torch.where(observed_step_mask, base_steps, step_tokens)
            current = sample.detach().clone().requires_grad_(apply_guidance)
            predicted = denoiser(current, step_tokens)
            current_for_update = current.detach()
            if apply_guidance:
                predicted_for_cost = to_cost_space(predicted)
                cost, diag = scheduled_physical_velocity_cost(predicted_for_cost, conditioning)
                grad = torch.autograd.grad(cost.sum(), current)[0]
                grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).clamp(min=1.0e-12)
                latent_grad = grad[:, current_index, state_dim:]
                current_latent_grad_norm = torch.linalg.norm(latent_grad, dim=-1)
                scale = torch.clamp(float(gradient_clip_norm) / grad_norm, max=1.0).view(-1, 1, 1)
                guided_current = current - float(guidance_scale) * grad * scale
                predicted = denoiser(guided_current.detach(), step_tokens)
                current_for_update = guided_current.detach()
                diagnostics["guidance_cost"] = cost.detach()
                diagnostics["guidance_grad_norm"] = grad_norm.detach()
                diagnostics["current_latent_grad_norm"] = current_latent_grad_norm.detach()
                if k == 0:
                    diagnostics.update({key: value.detach() for key, value in diag.items()})
            a_t = alpha[k].clamp(min=1.0e-6, max=1.0)
            if k == 0:
                sample = predicted
            else:
                a_prev = alpha[k - 1].clamp(min=1.0e-6, max=1.0)
                eps = (current_for_update - torch.sqrt(a_t) * predicted) / torch.sqrt(
                    (1.0 - a_t).clamp(min=1.0e-6)
                )
                sample = torch.sqrt(a_prev) * predicted + torch.sqrt((1.0 - a_prev).clamp(min=0.0)) * eps
            if mask is not None and observed is not None:
                sample = torch.where(mask, observed, sample)
            last_predicted = predicted.detach()
        diagnostics["prior_update_norm"] = torch.linalg.norm(last_predicted - trajectory_template).detach()
        if return_denormalized and normalizer is not None:
            sample = to_cost_space(sample)
        return sample, diagnostics

    unguided, _ = run_once(False)
    guided, diagnostics = run_once(True)
    future_start = int(conditioning.get("cost_start_index", min(current_index + 1, guided.shape[1] - 1)))
    before_velocity = physical_root_velocity_xy(
        unguided,
        state_dim=state_dim,
        velocity_slice=tuple(conditioning["velocity_slice"]),
        velocity_is_relative=bool(conditioning.get("velocity_is_relative", False)),
        current_velocity_xy=conditioning.get("current_velocity_xy"),
        projection_inverse=conditioning.get("projection_inverse"),
    )[:, future_start:]
    after_velocity = physical_root_velocity_xy(
        guided,
        state_dim=state_dim,
        velocity_slice=tuple(conditioning["velocity_slice"]),
        velocity_is_relative=bool(conditioning.get("velocity_is_relative", False)),
        current_velocity_xy=conditioning.get("current_velocity_xy"),
        projection_inverse=conditioning.get("projection_inverse"),
    )[:, future_start:]
    if before_velocity.shape[1] > 0:
        diagnostics["future_velocity_before_xy_mean"] = before_velocity.mean(dim=1).detach()
        diagnostics["future_velocity_after_xy_mean"] = after_velocity.mean(dim=1).detach()
        diagnostics["future_speed_before_mean"] = torch.linalg.norm(before_velocity, dim=-1).mean(dim=1).detach()
        diagnostics["future_speed_after_mean"] = torch.linalg.norm(after_velocity, dim=-1).mean(dim=1).detach()
        diagnostics["future_heading_before_mean"] = torch.atan2(
            before_velocity[..., 1], before_velocity[..., 0]
        ).mean(dim=1).detach()
        diagnostics["future_heading_after_mean"] = torch.atan2(
            after_velocity[..., 1], after_velocity[..., 0]
        ).mean(dim=1).detach()
    current_latent_before = unguided[:, current_index, state_dim:]
    current_latent_after = guided[:, current_index, state_dim:]
    diagnostics["current_latent_diff_norm"] = torch.linalg.norm(current_latent_after - current_latent_before, dim=-1).detach()
    diagnostics["current_latent_before_norm"] = torch.linalg.norm(current_latent_before, dim=-1).detach()
    diagnostics["current_latent_after_norm"] = torch.linalg.norm(current_latent_after, dim=-1).detach()
    diagnostics["unguided_tokens"] = unguided.detach()
    return guided, diagnostics
