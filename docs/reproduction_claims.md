# Reproduction Claims

This repository is a clean public-resource reproduction and engineering
release. It is not the official BeyondMimic repository.

What is included:

- A documented pipeline matching the method structure: rollout dataset, VAE,
  state-latent windows, transformer denoiser, and test-time guidance.
- Dependency-light schema, data, state, diffusion, and guidance utilities.
- Torch implementations for a conditional action VAE and transformer denoiser.
- A synthetic smoke pipeline that runs without proprietary checkpoints.
- Public-motion LAFAN1-style preparation and rollout fallback scripts.

What is not claimed:

- Official BeyondMimic checkpoints.
- Official unreleased DAgger logs.
- Real-robot deployment.
- Paper-level Fig. 5/Fig. 6 metric reproduction.
- Equivalence to the authors' private training code.

The development/debug history is intentionally kept in the original
[`BeyondMimic`](https://github.com/hunter20041220/BeyondMimic) repository.
This repository is the cleaned release surface.
