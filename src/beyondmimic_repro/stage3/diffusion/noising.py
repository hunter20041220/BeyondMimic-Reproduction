"""Per-token state/latent diffusion noising and target construction."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use Stage-3 noising") from exc


def _gather_alpha(alpha_bars: torch.Tensor, steps: torch.Tensor, feature_dim: int) -> torch.Tensor:
    idx = steps.to(dtype=torch.long, device=alpha_bars.device).clamp(min=0, max=alpha_bars.shape[0] - 1)
    values = alpha_bars[idx].to(dtype=torch.float32)
    return values.unsqueeze(-1).expand(*values.shape, feature_dim)


def add_per_token_noise(
    clean_states: torch.Tensor,
    clean_latents: torch.Tensor,
    noise_states: torch.Tensor,
    noise_latents: torch.Tensor,
    k_state: torch.Tensor,
    k_latent: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Apply independent diffusion steps to state and latent tokens."""
    if clean_states.shape != noise_states.shape or clean_latents.shape != noise_latents.shape:
        raise ValueError("clean/noise state and latent tensors must match")
    if clean_states.shape[:2] != clean_latents.shape[:2]:
        raise ValueError("state and latent tensors must share [B,T]")
    if k_state.shape != clean_states.shape[:2] or k_latent.shape != clean_states.shape[:2]:
        raise ValueError("k_state and k_latent must have shape [B,T]")
    a_state = _gather_alpha(alpha_bars, k_state, clean_states.shape[-1]).to(clean_states.device)
    a_latent = _gather_alpha(alpha_bars, k_latent, clean_latents.shape[-1]).to(clean_latents.device)
    noisy_state = torch.sqrt(a_state) * clean_states + torch.sqrt(1.0 - a_state) * noise_states
    noisy_latent = torch.sqrt(a_latent) * clean_latents + torch.sqrt(1.0 - a_latent) * noise_latents
    return torch.cat([noisy_state, noisy_latent], dim=-1)


def construct_training_target(
    clean_tokens: torch.Tensor,
    noise_tokens: torch.Tensor,
    *,
    prediction_type: str = "x0",
) -> torch.Tensor:
    """Build x0 default or epsilon legacy training targets."""
    if clean_tokens.shape != noise_tokens.shape:
        raise ValueError("clean_tokens and noise_tokens shapes must match")
    if prediction_type == "x0":
        return clean_tokens
    if prediction_type == "epsilon":
        return noise_tokens
    raise ValueError("prediction_type must be 'x0' or 'epsilon'")


def apply_inpainting_mask(noisy: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Clamp observed history/current/keyframe entries."""
    if noisy.shape != clean.shape or mask.shape != noisy.shape:
        raise ValueError("noisy, clean, and mask must share shape")
    return torch.where(mask.to(dtype=torch.bool, device=noisy.device), clean, noisy)
