# Stage-2/Stage-3 Transfer Manifest

- **release_branch**: stage2-stage3-complete-h20
- **base_commit**: e4f6a41
- **final_commit**: code_release_commit=d7cdf6d; latest_branch_commit=see git log
- **full_find_file_count**: 93072
- **source_inventory_count**: 3726
- **copied_file_count**: 2975
- **new_file_count**: 107
- **legacy_file_count**: 81
- **test_count**: 8 passed, 2 skipped
- **test_result**: python3 -m compileall src scripts: passed; python3 -m pytest -q: 8 passed, 2 skipped; key CLI --help: passed; ruff: not installed

## Implemented Modules

- teacher asset contract
- paper Stage-2 VAE interface
- DAgger dataset and collection semantics
- OU rollout noise and acceptance
- VAE rollout state-latent builder
- character-yaw state utilities
- emphasis projection persistence
- per-token state/latent diffusion noising
- x0 and epsilon target construction
- StateLatentTransformer
- EMA
- differentiable guidance costs
- controller normalized-action contract
- Isaac entrypoint import boundary

## Requires Isaac

- collect_dagger_round
- eval_vae_closed_loop
- collect_vae_rollout
- eval_diffusion_closed_loop
- eval_velocity_guidance
- validate_teacher_checkpoint against real task/checkpoints

## Known Limitations

- Torch is optional and unavailable in current system Python, so Torch tests skipped
- DAgger closed-loop not validated on H20
- VAE closed-loop not validated on H20
- VAE rollout dataset not collected on H20
- Guided diffusion closed-loop not validated on H20

## Files Deliberately Not Uploaded

- checkpoints
- ONNX
- NPZ datasets
- videos
- wandb logs
- Isaac/Omni caches
- credentials

Tracked file count: 201. See JSON for per-file SHA256.
