"""Transformer that supports independent state/latent diffusion steps."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use StateLatentTransformer") from exc


class StateLatentTransformer(nn.Module):
    """Predict clean or epsilon state-latent trajectory tokens."""

    def __init__(
        self,
        token_dim: int,
        sequence_length: int = 21,
        denoising_steps: int = 20,
        embedding_dim: int = 512,
        attention_heads: int = 8,
        transformer_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.sequence_length = int(sequence_length)
        self.denoising_steps = int(denoising_steps)
        self.input_proj = nn.Linear(token_dim, embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
        self.state_step_embed = nn.Embedding(denoising_steps, embedding_dim)
        self.latent_step_embed = nn.Embedding(denoising_steps, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.output_proj = nn.Linear(embedding_dim, token_dim)

    def forward(self, noisy_tokens: torch.Tensor, diffusion_steps: torch.Tensor) -> torch.Tensor:
        if noisy_tokens.ndim != 3:
            raise ValueError(f"noisy_tokens must be [B,T,D], got {tuple(noisy_tokens.shape)}")
        if diffusion_steps.shape != noisy_tokens.shape[:2] + (2,):
            raise ValueError(f"diffusion_steps must be [B,T,2], got {tuple(diffusion_steps.shape)}")
        steps = diffusion_steps.to(device=noisy_tokens.device, dtype=torch.long).clamp(0, self.denoising_steps - 1)
        hidden = self.input_proj(noisy_tokens)
        hidden = hidden + self.pos_embed[:, : noisy_tokens.shape[1]]
        hidden = hidden + self.state_step_embed(steps[..., 0]) + self.latent_step_embed(steps[..., 1])
        return self.output_proj(self.encoder(hidden))
