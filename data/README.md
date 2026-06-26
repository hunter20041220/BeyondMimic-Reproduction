# Data Layout

This repository does not commit large motion datasets, rollout shards, or
state-latent windows. Place or generate them under:

```text
data/raw/                 downloaded public motion files
data/processed/           normalized motion arrays and metadata
data/teacher_rollouts/    tracking-policy rollout datasets
data/state_latent/        trajectory windows for diffusion training
```

Recommended first pass:

```bash
python scripts/01_data/prepare_lafan1.py \
  --input data/raw/lafan1 \
  --output data/processed/lafan1_g1.npz
```

For a dependency-light smoke test:

```bash
python scripts/01_data/make_synthetic_fixture.py
```
