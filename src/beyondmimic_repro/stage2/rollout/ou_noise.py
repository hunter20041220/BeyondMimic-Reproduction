"""Reproducible Ornstein-Uhlenbeck action perturbation."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use OrnsteinUhlenbeckNoise") from exc


class OrnsteinUhlenbeckNoise:
    """Batch OU process with partial reset support."""

    def __init__(self, theta: float = 0.8, mu: float = 0.0, sigma: float = 0.1, dt: float = 0.02, seed: int | None = None) -> None:
        if theta <= 0.0 or sigma < 0.0 or dt <= 0.0:
            raise ValueError("theta and dt must be positive; sigma must be nonnegative")
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.seed = seed
        self.state: torch.Tensor | None = None
        self.generator: torch.Generator | None = None

    def reset(self, batch_size: int, action_dim: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Reset all environments and return the current noise state."""
        dev = torch.device(device)
        self.generator = torch.Generator(device=dev)
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        self.state = torch.full((batch_size, action_dim), self.mu, dtype=torch.float32, device=dev)
        return self.state.clone()

    def reset_mask(self, mask: torch.Tensor) -> None:
        """Reset selected batch rows to the OU mean."""
        if self.state is None:
            raise RuntimeError("call reset before reset_mask")
        mask = mask.to(device=self.state.device, dtype=torch.bool)
        if mask.shape != (self.state.shape[0],):
            raise ValueError(f"reset mask must be [{self.state.shape[0]}], got {tuple(mask.shape)}")
        self.state[mask] = self.mu

    def step(self, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Advance OU noise by one control step."""
        if self.state is None or self.generator is None:
            raise RuntimeError("call reset before step")
        if reset_mask is not None:
            self.reset_mask(reset_mask)
        drift = self.theta * (self.mu - self.state) * self.dt
        diffusion = self.sigma * (self.dt**0.5) * torch.randn(
            self.state.shape,
            dtype=self.state.dtype,
            device=self.state.device,
            generator=self.generator,
        )
        self.state = (self.state + drift + diffusion).to(torch.float32)
        return self.state.clone()
