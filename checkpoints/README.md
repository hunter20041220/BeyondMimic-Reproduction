# Checkpoints

Large checkpoints are intentionally not tracked by git.

```text
checkpoints/tracking/     tracking policy checkpoints
checkpoints/vae/          conditional action VAE checkpoints
checkpoints/diffusion/    transformer denoiser checkpoints
```

The scripts default to these paths but accept explicit `--checkpoint` or config
overrides.
