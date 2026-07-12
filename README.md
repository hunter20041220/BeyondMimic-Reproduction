# BeyondMimic-Reproduction

Clean public-resource reproduction code for a BeyondMimic-style pipeline:

```text
motion tracking policy rollout dataset
  -> DAgger-style VAE encoder / decoder
  -> state-latent trajectories
  -> Transformer diffusion denoiser
  -> test-time guidance
  -> joystick / waypoint / obstacle avoidance / inpainting
```

This is a cleaned release repository, not the official BeyondMimic codebase.
The original development/audit repository keeps the full commit history and
progress logs; this repository keeps the runnable research-code surface.

Development history and audit trail:
[hunter20041220/BeyondMimic](https://github.com/hunter20041220/BeyondMimic).

## Installation

```bash
git clone https://github.com/hunter20041220/BeyondMimic-Reproduction.git
cd BeyondMimic-Reproduction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,dev]"
```

For VAE and diffusion training:

```bash
pip install -r requirements/torch.txt
pip install -e .
```

## Quick Smoke Pipeline

This path needs no external dataset or simulator:

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

## Public Motion Data

Place retargeted LAFAN1 G1 CSV files under:

```text
data/raw/lafan1/
```

Expected row layout:

```text
root position xyz       3
root quaternion xyzw    4
G1 action / joints     29
total                  36
```

Prepare:

```bash
python scripts/01_data/prepare_lafan1.py \
  --input data/raw/lafan1 \
  --output data/processed/lafan1_g1.npz
```

Then build rollout windows:

```bash
python scripts/03_teacher_rollout/collect_teacher_rollout.py \
  --input data/processed/lafan1_g1.npz \
  --output data/teacher_rollouts/teacher_rollout_train.npz
```

## Training

## Stage-2/Stage-3 H20 Release Branch

This branch adds the organized Stage-2/3 release surface for RTX 4090 transfer:

```text
contracts/
stage2/       paper VAE, DAgger, OU rollout, VAE frontend
stage3/       character-frame state, state-latent, x0 diffusion, guidance
adapters/     Isaac import boundary and MuJoCo/controller contracts
legacy/       smoke/simplified baselines kept for comparison
```

Important truthfulness boundary:

```text
not validated on H20
requires RTX 4090 + Isaac Sim runtime
```

applies to DAgger closed-loop, VAE closed-loop, VAE rollout collection, and guided diffusion closed-loop.

Start with:

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py --help
python scripts/04_vae/train_vae_bc_warmstart.py --help
python scripts/05_state_latent/build_from_vae_rollout.py --help
python scripts/06_diffusion/train_state_latent_diffusion.py --help
python scripts/09_isaac/collect_dagger_round.py --help
```

See:

- `docs/rtx4090_handoff_guide.md`
- `docs/rtx4090_handoff_guide_zh.md`
- `docs/stage2_stage3_status.md`
- `docs/data_contracts_stage2_stage3.md`
- `docs/transfer_to_4090.md`
- `docs/paper_alignment_matrix.md`

Conditional action VAE:

```bash
python scripts/04_vae/train_vae.py \
  --dataset data/teacher_rollouts/teacher_rollout_train.npz \
  --output checkpoints/vae/vae_latest.pt
```

State-latent windows:

```bash
python scripts/05_state_latent/build_state_latent_dataset.py \
  --rollout data/teacher_rollouts/teacher_rollout_train.npz \
  --output data/state_latent/train_windows.npz
```

Transformer diffusion denoiser:

```bash
python scripts/06_diffusion/train_denoiser.py \
  --dataset data/state_latent/train_windows.npz \
  --output checkpoints/diffusion/denoiser_latest.pt
```

Guidance metrics:

```bash
python scripts/07_guidance/eval_guidance.py \
  --dataset data/state_latent/train_windows.npz
```

## Repository Layout

```text
configs/      experiment defaults
scripts/      command-line entrypoints by pipeline stage
src/          importable implementation
data/         local datasets, ignored by git
checkpoints/  local weights, ignored by git
outputs/      metrics, logs, figures, videos, ignored by git
docs/         method, data format, environment, claims
tests/        dependency-light regression tests
```

The stage numbering follows the reproduction chain:

```text
00 setup checks
01 public motion data
02 tracking-rollout schema evaluation
03 teacher rollout collection
04 conditional action VAE
05 state-latent trajectory construction
06 transformer diffusion denoiser
07 test-time guidance
08 output summaries / visualization hooks
```

## Scope

Included: schema utilities, public motion preparation, teacher-rollout dataset
format, conditional action VAE, transformer denoiser, state-latent dataset
builder, and trajectory-level guidance costs.

Not included: official BeyondMimic checkpoints, unreleased DAgger logs,
real-robot deployment, or paper-level official metric claims. See
`docs/reproduction_claims.md`.
