from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from beyondmimic_repro.stage3.diffusion.ema import ExponentialMovingAverage
from beyondmimic_repro.stage3.diffusion.noising import add_per_token_noise, construct_training_target
from beyondmimic_repro.stage3.diffusion.schedule import linear_beta_schedule, alpha_bars_from_betas
from beyondmimic_repro.stage3.guidance.composition import composed_guidance_cost
from beyondmimic_repro.stage3.guidance.inpainting import inpainting_guidance_cost
from beyondmimic_repro.stage3.guidance.joystick import joystick_guidance_cost
from beyondmimic_repro.stage3.guidance.obstacle import obstacle_guidance_cost
from beyondmimic_repro.stage3.guidance.waypoint import waypoint_guidance_cost
from beyondmimic_repro.stage3.models.state_latent_transformer import StateLatentTransformer


def test_per_token_diffusion_steps_and_targets() -> None:
    states = torch.ones(1, 2, 3)
    latents = torch.ones(1, 2, 2) * 2.0
    zeros_s = torch.zeros_like(states)
    zeros_l = torch.zeros_like(latents)
    alpha_bars = torch.tensor([1.0, 0.25])
    noisy = add_per_token_noise(
        states,
        latents,
        zeros_s,
        zeros_l,
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 0]]),
        alpha_bars,
    )
    assert torch.allclose(noisy[0, 0, :3], torch.ones(3))
    assert torch.allclose(noisy[0, 1, :3], torch.ones(3) * 0.5)
    assert construct_training_target(noisy, torch.zeros_like(noisy), prediction_type="x0").shape == noisy.shape
    assert torch.count_nonzero(construct_training_target(noisy, torch.ones_like(noisy), prediction_type="epsilon")) == noisy.numel()


def test_transformer_forward_shape_and_ema() -> None:
    model = StateLatentTransformer(token_dim=5, sequence_length=3, denoising_steps=4, embedding_dim=16, attention_heads=4, transformer_layers=1)
    x = torch.zeros(2, 3, 5)
    steps = torch.zeros(2, 3, 2, dtype=torch.long)
    assert model(x, steps).shape == x.shape
    ema = ExponentialMovingAverage(model)
    old = {k: v.clone() for k, v in ema.shadow.items()}
    for param in model.parameters():
        param.data.add_(1.0)
    ema.update(model)
    assert any(not torch.allclose(old[k], ema.shadow[k]) for k in old)


def test_guidance_costs_have_finite_gradients() -> None:
    traj = torch.zeros(2, 5, 6, requires_grad=True)
    traj.data[:, :, 0] = torch.linspace(0.0, 0.4, 5)
    costs = []
    costs.append(joystick_guidance_cost(traj, {"target_velocity_xy": torch.tensor([0.1, 0.0]), "dt": 1.0})[0])
    costs.append(waypoint_guidance_cost(traj, {"waypoint_xy": torch.tensor([0.4, 0.0])})[0])
    costs.append(obstacle_guidance_cost(traj, {"obstacle_xy": torch.tensor([2.0, 2.0]), "radius": 0.1})[0])
    target = torch.zeros_like(traj)
    mask = torch.zeros_like(traj)
    mask[:, -1, :2] = 1.0
    costs.append(inpainting_guidance_cost(traj, {"target": target, "mask": mask})[0])
    composed, _ = composed_guidance_cost(
        traj,
        {
            "objectives": [
                ("joystick", joystick_guidance_cost, {"target_velocity_xy": torch.tensor([0.1, 0.0]), "dt": 1.0}, 1.0),
                ("waypoint", waypoint_guidance_cost, {"waypoint_xy": torch.tensor([0.4, 0.0])}, 0.5),
            ]
        },
    )
    total = sum(cost.sum() for cost in costs) + composed.sum()
    total.backward()
    assert torch.isfinite(traj.grad).all()


def test_schedule_alpha_bars() -> None:
    betas = linear_beta_schedule(4)
    bars = alpha_bars_from_betas(betas)
    assert bars.shape == (4,)
    assert torch.all(bars > 0)
