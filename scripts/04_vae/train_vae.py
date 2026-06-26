#!/usr/bin/env python3
"""Train a small conditional action VAE on teacher rollout windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beyondmimic_repro.rollout.schema import load_teacher_rollout
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/teacher_rollouts/teacher_rollout_train.npz")
    parser.add_argument("--output", default="checkpoints/vae/vae_latest.pt")
    parser.add_argument("--metrics", default="outputs/metrics/vae_metrics.json")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from beyondmimic_repro.vae.torch_model import ConditionalActionVAE, vae_loss
    except ImportError as exc:
        raise SystemExit("Install torch first: pip install -r requirements/torch.txt") from exc

    states, actions, _ = load_teacher_rollout(args.dataset)
    flat_states = states.reshape(-1, states.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    dataset = TensorDataset(torch.from_numpy(flat_states), torch.from_numpy(flat_actions))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)
    model = ConditionalActionVAE(states.shape[-1], actions.shape[-1], args.latent_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for state_batch, action_batch in loader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            recon, mu, logvar = model(state_batch, action_batch)
            loss = vae_loss(recon, action_batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": states.shape[-1],
            "action_dim": actions.shape[-1],
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
        },
        output,
    )
    summary = {"status": "ok", "checkpoint": str(output), "dataset": args.dataset, "history": history}
    write_json(args.metrics, summary)
    print(summary)


if __name__ == "__main__":
    main()
