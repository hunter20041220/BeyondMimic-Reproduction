"""Legacy epsilon-prediction transformer denoiser for smoke tests."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the torch extra to use beyondmimic_repro.diffusion.torch_denoiser") from exc


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for integer diffusion steps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half - 1, 1))
        )
        angles = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class StateLatentTransformerDenoiser(nn.Module):
    """Legacy epsilon-prediction baseline; Stage-3 defaults to x0 prediction."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.input = nn.Linear(token_dim, hidden_dim)
        self.time = nn.Sequential(SinusoidalTimeEmbedding(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Linear(hidden_dim, token_dim)

    def forward(self, noisy_tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        hidden = self.input(noisy_tokens)
        hidden = hidden + self.time(timesteps)[:, None, :]
        return self.output(self.transformer(hidden))


LegacyEpsilonDenoiser = StateLatentTransformerDenoiser
