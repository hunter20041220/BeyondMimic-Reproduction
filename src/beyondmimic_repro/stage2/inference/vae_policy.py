"""VAE policy frontend that outputs normalized actions only."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use VAEPolicyFrontend") from exc

from beyondmimic_repro.contracts.action import validate_normalized_action
from beyondmimic_repro.stage2.dagger.interfaces import RobotState
from beyondmimic_repro.stage2.models.conditional_action_vae import PaperConditionalActionVAE


class VAEPolicyFrontend:
    """Decode the current latent/action for a robot backend."""

    def __init__(self, model: PaperConditionalActionVAE, *, device: str | torch.device = "cpu", deterministic: bool = True) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.model.eval()

    def act(self, reference_input: np.ndarray, robot_state: RobotState, previous_action: np.ndarray) -> np.ndarray:
        """Return one normalized 29-D action, never torque or PD targets."""
        del previous_action
        ref = torch.as_tensor(reference_input, dtype=torch.float32, device=self.device).reshape(1, -1)
        proprio = torch.as_tensor(robot_state.policy_observation[-self.model.config.decoder_proprio_dim :], dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            mu, logvar = self.model.encode(ref)
            latent = mu if self.deterministic else self.model.reparameterize(mu, logvar)
            action = self.model.decode(latent, proprio).squeeze(0).cpu().numpy()
        return validate_normalized_action(action)
