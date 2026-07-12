# Data Contracts For Stage-2/Stage-3

Reference motion:

```text
motion.npz
```

Reference motion is trajectory data. It is not a policy and is not a teacher rollout.

Teacher closed-loop rollout:

```text
teacher_rollout.npz
```

This is a Stage-1 teacher running from historical states. It can be used for D0 warm start only.

DAgger aggregated dataset:

```text
D0 union D1 union D2 ...
```

This contains student-visited states and teacher labels for those student states.

VAE rollout dataset:

```text
trained VAE student closed-loop rollout
```

This is the rollout distribution that Stage-3 should learn from.

State-latent trajectory dataset:

```text
actual VAE rollout state + VAE latent
```

It must not be built by directly encoding teacher rollout and presenting it as final Stage-3 data. The legacy teacher-direct path emits a warning and is preserved only as a baseline.

Schemas:

- Teacher assets: `src/beyondmimic_repro/contracts/teacher_assets.py`
- DAgger: `schema_version = "stage2-dagger-v1"`
- VAE rollout: `schema_version = "stage2-vae-rollout-v1"`
- State-latent: `schema_version = "stage3-state-latent-v1"`
