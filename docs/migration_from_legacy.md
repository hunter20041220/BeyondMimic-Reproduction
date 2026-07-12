# Migration From Legacy Baselines

Legacy/smoke code is preserved, not deleted.

Legacy baseline:

- `beyondmimic_repro.vae.torch_model.ConditionalActionVAE`
- `beyondmimic_repro.diffusion.torch_denoiser.StateLatentTransformerDenoiser`
- `beyondmimic_repro.data.state_latent.build_state_latent_dataset`

Release aliases:

- `LegacyConditionalActionVAE`
- `LegacyEpsilonDenoiser`
- `LegacyStateLatentTeacherEncoder`

These are simplified baselines:

- State-action VAE, not paper DAgger VAE.
- Teacher-rollout direct state-latent encoding, not VAE rollout data.
- Epsilon denoiser, not default x0 trajectory prediction.

New interfaces:

- `PaperConditionalActionVAE`
- `stage2-dagger-v1`
- `build_from_vae_rollout`
- `StateLatentTransformer`
- Torch guidance costs in `src/beyondmimic_repro/stage3/guidance/`
