# Stage-2/Stage-3 Release Summary

Repository: `https://github.com/hunter20041220/BeyondMimic-Reproduction`

Branch: `stage2-stage3-complete-h20`

Base commit: `e4f6a41`

Code release commit: `d7cdf6d`

Push status: pushed to `origin/stage2-stage3-complete-h20`.

Source files discovered by full find: `93072`

Source files indexed in inventory: `3726`

Source files semantically migrated/copied: `2975`

Tests passed: `8`

Tests skipped: `2` Torch-dependent tests skipped because Torch is not installed in the current system Python.

Verification:

- `python3 -m compileall src scripts` passed.
- `python3 -m pytest -q` passed with `8 passed, 2 skipped`.
- Key CLI `--help` checks passed.
- `ruff` was not installed.

Legacy implementations preserved:

- Legacy state-action `ConditionalActionVAE`
- Legacy teacher-rollout state-latent encoder
- Legacy epsilon-prediction denoiser

New Stage-2 modules:

- Teacher asset contract
- Paper VAE model/config
- DAgger schema and collector core
- OU rollout perturbation and acceptance
- VAE frontend and checkpoint format

New Stage-3 modules:

- Character-yaw state representation
- State-latent dataset from VAE rollout
- Per-token diffusion noising
- x0/epsilon target construction
- Transformer denoiser
- EMA
- Joystick, waypoint, obstacle, inpainting, composed guidance

Isaac integration entrypoints:

- `scripts/09_isaac/validate_teacher_checkpoint.py`
- `scripts/09_isaac/collect_dagger_round.py`
- `scripts/09_isaac/eval_vae_closed_loop.py`
- `scripts/09_isaac/collect_vae_rollout.py`
- `scripts/09_isaac/eval_diffusion_closed_loop.py`
- `scripts/09_isaac/eval_velocity_guidance.py`

Known limitations:

- DAgger closed-loop is not validated on H20.
- VAE Isaac closed-loop is not validated on H20.
- VAE rollout dataset is not collected on H20.
- Guided diffusion closed-loop is not validated on H20.
- Walk-to-run/Fig. 5 reproduction is not claimed.
