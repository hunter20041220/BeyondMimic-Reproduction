# Limitations

- The release code provides a public-data fallback for teacher rollouts. It does
  not bundle an official tracking policy checkpoint.
- The state-latent builder includes a deterministic linear encoder for smoke
  tests. Use the trained VAE checkpoint for reported experiments.
- Guidance evaluation is implemented at the trajectory-token level. Full
  closed-loop simulator evaluation requires IsaacLab/MuJoCo integration and
  task-specific rendering.
- Large data, videos, logs, and checkpoints are intentionally excluded from git.
