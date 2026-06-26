#!/usr/bin/env python3
"""Train a compact transformer diffusion denoiser on state-latent windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beyondmimic_repro.data.state_latent import load_state_latent_tokens
from beyondmimic_repro.utils.io import write_json


def cosine_alpha_bar(timesteps: np.ndarray, total_steps: int) -> np.ndarray:
    s = 0.008
    x = (timesteps / total_steps + s) / (1.0 + s)
    return np.cos(x * np.pi / 2.0) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/state_latent/train_windows.npz")
    parser.add_argument("--output", default="checkpoints/diffusion/denoiser_latest.pt")
    parser.add_argument("--metrics", default="outputs/metrics/diffusion_metrics.json")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from beyondmimic_repro.diffusion.torch_denoiser import StateLatentTransformerDenoiser
    except ImportError as exc:
        raise SystemExit("Install torch first: pip install -r requirements/torch.txt") from exc

    tokens = load_state_latent_tokens(args.dataset)
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)
    model = StateLatentTransformerDenoiser(tokens.shape[-1], args.hidden_dim, args.depth, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for (clean_batch,) in loader:
            clean_batch = clean_batch.to(device)
            batch = clean_batch.shape[0]
            t = torch.randint(1, args.diffusion_steps, (batch,), device=device)
            alpha = torch.from_numpy(cosine_alpha_bar(t.detach().cpu().numpy(), args.diffusion_steps)).float().to(device)
            noise = torch.randn_like(clean_batch)
            noisy = alpha.sqrt()[:, None, None] * clean_batch + (1.0 - alpha).sqrt()[:, None, None] * noise
            pred = model(noisy, t)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "noise_mse": float(np.mean(losses))})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_dim": tokens.shape[-1],
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "heads": args.heads,
            "diffusion_steps": args.diffusion_steps,
        },
        output,
    )
    summary = {"status": "ok", "checkpoint": str(output), "dataset": args.dataset, "history": history}
    write_json(args.metrics, summary)
    print(summary)


if __name__ == "__main__":
    main()
