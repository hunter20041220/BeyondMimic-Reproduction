# Data Format

## Retargeted LAFAN1 CSV

The public motion entrypoint assumes no-header CSV rows with 36 values:

```text
0:3    root position xyz
3:7    root quaternion xyzw
7:36   Unitree G1 29-D action / joint vector
```

Prepare:

```bash
python scripts/01_data/prepare_lafan1.py \
  --input data/raw/lafan1 \
  --output data/processed/lafan1_g1.npz
```

## Teacher Rollout NPZ

```text
states  float32 [N, T, state_dim]
actions float32 [N, T, 29]
names   str     [N]
```

The current public fallback constructs `state=[root, action, action_velocity]`.
A full tracking-policy rollout can replace this with simulator observations
while preserving the same `states/actions/names` contract.

## State-Latent NPZ

```text
tokens  float32 [N, T, state_dim + latent_dim]
states  float32 [N, T, state_dim]
latents float32 [N, T, latent_dim]
```

The smoke path uses a deterministic linear encoder if no trained VAE checkpoint
is available. For reported experiments, generate latents with the trained VAE.
