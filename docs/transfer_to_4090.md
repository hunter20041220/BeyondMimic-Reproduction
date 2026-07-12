# Transfer To RTX 4090 Isaac Lab Host

Clone:

```bash
git clone https://github.com/hunter20041220/BeyondMimic-Reproduction.git
cd BeyondMimic-Reproduction
git checkout stage2-stage3-complete-h20
```

Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pip install -r requirements/torch.txt
pip install -r requirements/isaaclab.txt
```

Install Isaac Sim / Isaac Lab and the `whole_body_tracking` task on the 4090 host. Use the version required by that task checkout. Run all Isaac scripts from that activated environment.

Place assets locally:

```text
data/motions/                 motion.npz files
data/teacher_rollouts/        D0 rollout NPZ files
checkpoints/tracking/         teacher .pt files
checkpoints/vae/              VAE checkpoints
checkpoints/diffusion/        diffusion checkpoints
configs/local/teacher_map.json
```

Audit teacher map:

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py \
  --teacher-map configs/local/teacher_map.json \
  --data-root data \
  --checkpoint-root checkpoints/tracking \
  --require-files \
  --output-dir outputs/4090_audit
```

Walk1 single-motion DAgger:

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --device cuda:0 \
  --output-dir outputs/dagger_walk1
```

VAE closed-loop evaluation:

```bash
python scripts/09_isaac/eval_vae_closed_loop.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_closed_loop
```

VAE OU rollout:

```bash
python scripts/09_isaac/collect_vae_rollout.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_rollout_ou
```

Build state-latent dataset:

```bash
python scripts/05_state_latent/build_from_vae_rollout.py \
  --vae-rollout outputs/vae_rollout_ou/vae_rollout.npz \
  --output data/state_latent/vae_rollout_state_latent.npz \
  --output-dir outputs/state_latent_build
```

Diffusion training:

```bash
python scripts/06_diffusion/train_state_latent_diffusion.py \
  --state-latent-dataset data/state_latent/vae_rollout_state_latent.npz \
  --config configs/stage3/diffusion_engineering_50hz.yaml \
  --device cuda:0 \
  --output-dir outputs/diffusion_train
```

Guided diffusion evaluation:

```bash
python scripts/09_isaac/eval_velocity_guidance.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --diffusion-checkpoint checkpoints/diffusion/diffusion_latest.pt \
  --num-envs 512 \
  --device cuda:0 \
  --output-dir outputs/velocity_guidance
```

50 Hz vs 25 Hz:

- `configs/stage3/diffusion_engineering_50hz.yaml`: 0.32 s future horizon.
- `configs/stage3/diffusion_paper_25hz.yaml`: 0.64 s future horizon.

Common failures:

- Missing Isaac runtime: run on the 4090 host after Isaac Lab setup.
- Missing `whole_body_tracking`: install/register the task before running entrypoints.
- Teacher map points to H20 paths: pass `--data-root` and `--checkpoint-root`.
- Checkpoint/action dimension mismatch: validate joint order and action scale before rollout.
