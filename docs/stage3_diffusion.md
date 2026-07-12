# Stage-3 State-Latent Diffusion

The default Stage-3 implementation predicts clean trajectories:

```text
prediction_type: x0
```

The legacy epsilon target is still supported for comparison and is not the default paper contract.

Window semantics are explicit:

```text
past_steps = 4
include_current = True
future_steps = 16
sequence_length = 21
```

The 50 Hz engineering gate has a 0.32 s future horizon. The 25 Hz paper-faithful config has a 0.64 s future horizon.

State representation:

- Root features are expressed relative to the current character-yaw frame.
- Body position and velocity are expressed in each timestep local root/yaw frame.
- Quaternion and Rot6D conventions are explicit.
- Emphasis projection is saved and reloaded; it is not regenerated randomly during inference.

Diffusion supports per-token noising:

```text
k_state  [B, sequence_length]
k_latent [B, sequence_length]
```

Guidance costs are differentiable Torch functions for joystick, waypoint, obstacle, inpainting, and composition.
