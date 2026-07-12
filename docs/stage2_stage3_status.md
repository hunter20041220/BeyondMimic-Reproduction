# Stage-2/Stage-3 Status

This branch organizes H20 Stage-2/3 code for transfer to an RTX 4090 Isaac Lab host.

H20 evidence supports:

- Source inventory, package contracts, configs, and CPU synthetic tests.
- Offline VAE, DAgger, state-latent, diffusion, guidance, and controller interfaces.
- Preservation of legacy/smoke simplified baselines.

H20 evidence does not support:

- Completed paper-faithful DAgger.
- VAE Isaac closed-loop success.
- VAE rollout dataset collected in Isaac.
- Guided diffusion Isaac closed-loop success.
- Fig. 5 or walk-to-run reproduction.

All Isaac closed-loop entrypoints are marked: `not validated on H20; requires RTX 4090 + Isaac Sim runtime`.

Primary code paths:

- Contracts: `src/beyondmimic_repro/contracts/`
- Stage-2: `src/beyondmimic_repro/stage2/`
- Stage-3: `src/beyondmimic_repro/stage3/`
- Isaac adapter: `src/beyondmimic_repro/adapters/isaac/`
- MuJoCo adapter contract: `src/beyondmimic_repro/adapters/mujoco/`
- Legacy baselines: `src/beyondmimic_repro/legacy/`
