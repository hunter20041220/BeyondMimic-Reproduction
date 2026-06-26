# Pipeline

The repository is organized around the reproduction chain:

```text
motion tracking policy rollout dataset
  -> DAgger-style conditional action VAE
  -> state-latent trajectory windows
  -> transformer diffusion denoiser
  -> test-time guidance
  -> joystick / waypoint / obstacle / inpainting evaluation
```

## 1. Motion Tracking Rollout Dataset

Expected file:

```text
data/teacher_rollouts/teacher_rollout_train.npz
```

Required arrays:

```text
states  [N, T, state_dim]
actions [N, T, 29]
names   [N]
```

The release repository provides a public-motion fallback:

```bash
python scripts/03_teacher_rollout/collect_teacher_rollout.py \
  --input data/processed/lafan1_g1.npz \
  --output data/teacher_rollouts/teacher_rollout_train.npz
```

An IsaacLab/RSL-RL tracking policy can write the same schema after closed-loop
rollout.

Schema-only evaluation:

```bash
python scripts/02_tracking/eval_tracking_policy.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz
```

## 2. Conditional Action VAE

```bash
python scripts/04_vae/train_vae.py \
  --dataset data/teacher_rollouts/teacher_rollout_train.npz \
  --output checkpoints/vae/vae_latest.pt
```

The VAE encodes state-action pairs and decodes 29-D G1 actions conditioned on
state.

## 3. State-Latent Trajectories

```bash
python scripts/05_state_latent/build_state_latent_dataset.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz \
  --output data/state_latent/train_windows.npz
```

The output stores:

```text
tokens  [N, T, state_dim + latent_dim]
states  [N, T, state_dim]
latents [N, T, latent_dim]
```

## 4. Transformer Diffusion Denoiser

```bash
python scripts/06_diffusion/train_denoiser.py \
  --dataset data/state_latent/train_windows.npz \
  --output checkpoints/diffusion/denoiser_latest.pt
```

The denoiser predicts DDPM noise over state-latent trajectory tokens.

## 5. Test-Time Guidance

```bash
python scripts/07_guidance/eval_guidance.py \
  --dataset data/state_latent/train_windows.npz
```

Implemented guidance costs:

```text
joystick       root XY velocity target
waypoint       final root XY target
obstacle       SDF-style barrier
inpainting     keyframe reconstruction
composed       waypoint + obstacle + joystick
```

## 6. Output Summary

Generated metrics are local artifacts and are ignored by git. To inspect a run:

```bash
python scripts/08_visualization/summarize_outputs.py
```
