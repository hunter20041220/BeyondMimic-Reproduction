# Stage-2 VAE And DAgger

The paper-style VAE is implemented as `PaperConditionalActionVAE`.

Encoder input:

```text
reference joint position     29
reference joint velocity     29
anchor position error         3
anchor orientation Rot6D      6
encoder input                67
```

Decoder conditioning:

```text
latent z                     32
projected gravity             3
IMU/root twist                6
current joint position       29
current joint velocity       29
previous normalized action   29
decoder total               128
```

The decoder outputs normalized 29-D actions. It does not output torque or scaled PD targets.

DAgger semantics:

- `scripts/04_vae/train_vae_bc_warmstart.py` is D0 offline BC warm start.
- `scripts/04_vae/train_vae_dagger.py` updates from aggregated DAgger data.
- `scripts/04_vae/collect_dagger_isaac.py` is an Isaac migration entrypoint.
- During DAgger collection, the student action is executed. The teacher action is only a label.

The joint-position convention is explicit in config as `joint_position_semantics`. The release default is `relative_to_default`; the 4090 validation run must confirm whether the Isaac task exports absolute or default-relative joint positions.
