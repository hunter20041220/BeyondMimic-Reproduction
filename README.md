# BeyondMimic-Reproduction

This repository provides a clean, research-oriented reproduction codebase for a
BeyondMimic-style whole-body motion pipeline. It is an independent reproduction
surface, not the official BeyondMimic implementation.
https://arxiv.org/pdf/2508.08241
```text
motion data
  -> teacher rollout contracts
  -> conditional action VAE
  -> state-latent trajectory windows
  -> Transformer diffusion denoiser
  -> test-time guidance
```

## Highlights

- Dependency-light smoke pipeline for data contracts, VAE inputs, diffusion
  shapes, and guidance costs.
- Modular implementation under `src/beyondmimic_repro` with stage-specific
  command-line entrypoints under `scripts`.
- Configured experiments for tracking data preparation, VAE training,
  state-latent diffusion, and guidance tasks.
- Optional Isaac Sim / Isaac Lab entrypoints are isolated behind runtime import
  boundaries so the package can be imported and tested without a simulator.

## Installation

```bash
git clone https://github.com/hunter20041220/BeyondMimic-Reproduction.git
cd BeyondMimic-Reproduction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,dev]"
```

Install PyTorch separately for VAE or diffusion training:

```bash
pip install -r requirements/torch.txt
pip install -e .
```

Install simulator-specific dependencies only in an environment where Isaac Sim,
Isaac Lab, and the required task package are available.

## Quick Start

The following smoke pipeline uses synthetic data and does not require external
datasets, checkpoints, or a simulator:

```bash
python scripts/00_setup/check_environment.py
python scripts/01_data/make_synthetic_fixture.py
python scripts/03_teacher_rollout/collect_teacher_rollout.py \
  --input data/processed/synthetic_lafan1_g1.npz \
  --output data/teacher_rollouts/teacher_rollout_train.npz
python scripts/02_tracking/eval_tracking_policy.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz
python scripts/05_state_latent/build_state_latent_dataset.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz \
  --output data/state_latent/train_windows.npz
python scripts/07_guidance/eval_guidance.py \
  --dataset data/state_latent/train_windows.npz
python scripts/08_visualization/summarize_outputs.py
pytest
```

## Data

Place retargeted LAFAN1 G1 CSV files under:

```text
data/raw/lafan1/
```

Each row is expected to contain:

```text
root position xyz       3
root quaternion xyzw    4
G1 action / joints     29
total                  36
```

Prepare the dataset and build teacher rollout windows:

```bash
python scripts/01_data/prepare_lafan1.py \
  --input data/raw/lafan1 \
  --output data/processed/lafan1_g1.npz
python scripts/03_teacher_rollout/collect_teacher_rollout.py \
  --input data/processed/lafan1_g1.npz \
  --output data/teacher_rollouts/teacher_rollout_train.npz
```

## Training And Evaluation

Train a conditional action VAE:

```bash
python scripts/04_vae/train_vae.py \
  --dataset data/teacher_rollouts/teacher_rollout_train.npz \
  --output checkpoints/vae/vae_latest.pt
```

Build state-latent windows:

```bash
python scripts/05_state_latent/build_state_latent_dataset.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz \
  --output data/state_latent/train_windows.npz
```

Train the Transformer diffusion denoiser:

```bash
python scripts/06_diffusion/train_denoiser.py \
  --dataset data/state_latent/train_windows.npz \
  --output checkpoints/diffusion/denoiser_latest.pt
```

Evaluate offline guidance costs:

```bash
python scripts/07_guidance/eval_guidance.py \
  --dataset data/state_latent/train_windows.npz
```

Isaac-based closed-loop commands are available under `scripts/09_isaac`. These
entrypoints require a working Isaac Sim / Isaac Lab installation and the
corresponding whole-body tracking task.

## Repository Layout

```text
configs/       experiment defaults
scripts/       command-line entrypoints by pipeline stage
src/           importable implementation
tests/         regression tests
requirements/  optional dependency groups
data/          local datasets, ignored by git
checkpoints/   local weights, ignored by git
outputs/       metrics, logs, figures, and videos, ignored by git
```

The stage numbering follows the reproduction pipeline:

```text
00 setup checks
01 public motion data
02 tracking-rollout schema evaluation
03 teacher rollout collection
04 conditional action VAE
05 state-latent trajectory construction
06 Transformer diffusion denoiser
07 test-time guidance
08 output summaries and visualization hooks
09 optional Isaac runtime entrypoints
```

## Scope

Included: schema utilities, motion data preparation, teacher-rollout dataset
format, conditional action VAE, state-latent dataset builder, Transformer
denoiser, and trajectory-level guidance costs.

Not included: official BeyondMimic checkpoints, private training logs,
real-robot deployment code, or official benchmark claims.

## Citation

If this repository is useful for your research, please cite the BeyondMimic
paper and this reproduction repository.
