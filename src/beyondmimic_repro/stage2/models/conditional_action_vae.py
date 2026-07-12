"""Paper-semantics conditional action VAE.

Encoder input is reference-motion information. Decoder conditioning is current
robot proprioception. This differs from the legacy state-action VAE baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the torch extra to use Stage-2 VAE modules") from exc

from beyondmimic_repro.contracts.action import ACTION_DIM
from beyondmimic_repro.contracts.observation import DECODER_PROPRIO_DIM, ENCODER_REFERENCE_DIM


@dataclass(frozen=True)
class PaperVAEConfig:
    """Default Stage-2 VAE configuration from the paper contract."""

    encoder_input_dim: int = ENCODER_REFERENCE_DIM
    decoder_proprio_dim: int = DECODER_PROPRIO_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = 32
    encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    activation: str = "ELU"
    learning_rate: float = 5e-4
    kl_coefficient: float = 0.01
    gradient_accumulation_steps: int = 15
    joint_position_semantics: str = "relative_to_default"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["encoder_hidden_dims"] = list(self.encoder_hidden_dims)
        data["decoder_hidden_dims"] = list(self.decoder_hidden_dims)
        return data


def _activation(name: str) -> nn.Module:
    if name == "ELU":
        return nn.ELU()
    if name == "SiLU":
        return nn.SiLU()
    if name == "GELU":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name!r}")


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), _activation(activation)])
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class PaperConditionalActionVAE(nn.Module):
    """Encode reference motion and decode normalized actions from proprioception."""

    def __init__(self, config: PaperVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or PaperVAEConfig()
        self.encoder = _mlp(
            self.config.encoder_input_dim,
            self.config.encoder_hidden_dims,
            self.config.latent_dim * 2,
            self.config.activation,
        )
        self.decoder = _mlp(
            self.config.latent_dim + self.config.decoder_proprio_dim,
            self.config.decoder_hidden_dims,
            self.config.action_dim,
            self.config.activation,
        )

    def encode(self, encoder_reference_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_reference_input.shape[-1] != self.config.encoder_input_dim:
            raise ValueError(
                f"encoder_reference_input last dim must be {self.config.encoder_input_dim}, "
                f"got {tuple(encoder_reference_input.shape)}"
            )
        stats = self.encoder(encoder_reference_input)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent: torch.Tensor, decoder_proprio_input: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latent last dim must be {self.config.latent_dim}, got {tuple(latent.shape)}")
        if decoder_proprio_input.shape[-1] != self.config.decoder_proprio_dim:
            raise ValueError(
                f"decoder_proprio_input last dim must be {self.config.decoder_proprio_dim}, "
                f"got {tuple(decoder_proprio_input.shape)}"
            )
        return self.decoder(torch.cat([latent, decoder_proprio_input], dim=-1))

    def forward(
        self,
        encoder_reference_input: torch.Tensor,
        decoder_proprio_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(encoder_reference_input)
        latent = self.reparameterize(mu, logvar)
        action = self.decode(latent, decoder_proprio_input)
        return action, mu, logvar, latent


def paper_vae_loss(
    predicted_action: torch.Tensor,
    teacher_action: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_coefficient: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return total, reconstruction, and KL losses."""
    recon = torch.mean((predicted_action - teacher_action) ** 2)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon + kl_coefficient * kl
    return total, {"reconstruction_loss": recon, "kl_loss": kl, "total_loss": total}
