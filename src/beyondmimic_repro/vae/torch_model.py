"""Small conditional action VAE used by the release training script."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the torch extra to use beyondmimic_repro.vae.torch_model") from exc


class ConditionalActionVAE(nn.Module):
    """Encode state-action pairs and decode actions conditioned on state."""

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat([state, action], dim=-1))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-10.0, max=10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([state, latent], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(state, action)
        latent = self.reparameterize(mu, logvar)
        return self.decode(state, latent), mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
    recon_loss = torch.mean((recon - target) ** 2)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
