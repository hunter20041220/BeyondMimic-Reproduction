"""Diffusion + VAE receding-horizon frontend."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use DiffusionVAEPolicyFrontend") from exc

from beyondmimic_repro.contracts.action import validate_normalized_action
from beyondmimic_repro.stage2.dagger.interfaces import RobotState
from beyondmimic_repro.stage2.models.conditional_action_vae import PaperConditionalActionVAE
from beyondmimic_repro.stage3.diffusion.sampler import extract_current_latent, guided_sample


class DiffusionVAEPolicyFrontend:
    """Plan state-latent trajectory, execute only the current decoded action."""

    def __init__(
        self,
        denoiser: torch.nn.Module,
        vae: PaperConditionalActionVAE,
        *,
        state_dim: int,
        device: str | torch.device = "cpu",
        current_index: int = 4,
    ) -> None:
        self.denoiser = denoiser.to(device).eval()
        self.vae = vae.to(device).eval()
        self.state_dim = int(state_dim)
        self.device = torch.device(device)
        self.current_index = int(current_index)

    def act(self, history: np.ndarray, robot_state: RobotState, task_context: dict[str, object]) -> np.ndarray:
        noisy = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        steps = torch.zeros(noisy.shape[:2] + (2,), dtype=torch.long, device=self.device)
        with torch.enable_grad():
            predicted, _ = guided_sample(self.denoiser, noisy, steps, task_context, [], [])
        z_current = extract_current_latent(predicted, self.state_dim, self.current_index)
        proprio = torch.as_tensor(robot_state.policy_observation[-self.vae.config.decoder_proprio_dim :], dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.vae.decode(z_current, proprio).squeeze(0).cpu().numpy()
        return validate_normalized_action(action)
