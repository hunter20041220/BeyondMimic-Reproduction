# Paper Alignment Matrix

| Paper requirement | Current implementation | Code path | Status | Offline tested | Isaac closed-loop | Known deviation | Next step |
|---|---|---|---|---|---|---|---|
| Teacher asset map | Relocatable schema and validation | `contracts/teacher_assets.py` | Implemented | Synthetic-tested | Requires Isaac | Does not upload assets | Validate real 4090 paths |
| D0 warm start | Separate BC entrypoint with warning | `scripts/04_vae/train_vae_bc_warmstart.py` | Implemented | Argument-tested | Not applicable | Training body is runtime-ready scaffold | Run on real D0 |
| DAgger collection | Student executes, teacher labels | `stage2/dagger/collector_core.py` | Implemented | Synthetic-tested | Requires Isaac | No H20 closed loop | Collect D1 on 4090 |
| Paper VAE encoder/decoder | 67-D encoder, 96-D proprio, 32-D latent, 29-D action | `stage2/models/conditional_action_vae.py` | Implemented | Synthetic-tested | Requires Isaac | Joint-position convention must be confirmed | Validate observation builder |
| OU perturbation | Reproducible OU with partial reset | `stage2/rollout/ou_noise.py` | Implemented | Synthetic-tested | Requires Isaac | No real rollout acceptance yet | Run VAE OU rollout |
| VAE rollout acceptance | 2.5 s noise, 5.0 s survival | `stage2/rollout/acceptance.py` | Implemented | Synthetic-tested | Requires Isaac | No H20 Isaac survival evidence | Evaluate on 4090 |
| State-latent source | Requires VAE rollout | `stage3/datasets/state_latent_builder.py` | Implemented | Synthetic-tested | Requires Isaac | Existing teacher-direct path is legacy | Build from accepted VAE rollout |
| Character-yaw state | Transform utilities and tested invariance | `stage3/representation/character_frame.py` | Implemented | Synthetic-tested | Requires Isaac | Full body schema depends runtime tensors | Validate Isaac tensor mapping |
| Emphasis projection | Save/load P and P_inv | `stage3/representation/emphasis_projection.py` | Implemented | Synthetic-tested | Not applicable | None known | Persist projection with datasets |
| x0 diffusion | Clean trajectory target default | `stage3/diffusion/noising.py` | Implemented | Synthetic-tested | Requires Isaac | Training not run in this branch | Train on 4090 dataset |
| Per-token noise | `k_state` and `k_latent` per timestep | `stage3/diffusion/noising.py` | Implemented | Synthetic-tested | Not applicable | None known | Use in training loop |
| Guidance costs | Joystick, waypoint, obstacle, inpainting, composition | `stage3/guidance/` | Implemented | Synthetic-tested | Requires Isaac | No guided closed-loop claim | Evaluate on 4090 |
| Controller frontend | Normalized action contract | `adapters/mujoco/shared_controller_contract.py` | Implemented | Synthetic-tested | Requires Isaac | Backend-specific mapping still runtime work | Validate action scale/joint order |
| Isaac entrypoints | AppLauncher import boundary | `scripts/09_isaac/` | Implemented | Help/import-boundary only | Requires Isaac | H20 cannot launch Isaac | Run on 4090 |
