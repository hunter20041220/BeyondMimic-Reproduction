# RTX 4090 Stage-2/Stage-3 Handoff Guide

This document is the memory-free handoff guide for a new RTX 4090 server.
It explains what each released code area does, what its inputs and outputs are,
and how to continue the real Isaac Lab validation work.

Truth boundary:

- This branch organizes code, schemas, configs, CLI entrypoints, and CPU tests.
- H20 did not validate Isaac closed-loop DAgger, VAE rollout, or guided diffusion.
- Any closed-loop result must be generated again on the RTX 4090 Isaac Sim host.

## 1. Clone And Install

```bash
git clone https://github.com/hunter20041220/BeyondMimic-Reproduction.git
cd BeyondMimic-Reproduction
git checkout stage2-stage3-complete-h20

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pip install -r requirements/torch.txt
pip install -r requirements/isaaclab.txt
```

Then install Isaac Sim, Isaac Lab, and the `whole_body_tracking` task in the
same environment. The Isaac scripts are intentionally import-safe before Isaac
starts, but the real task package must exist on the 4090 host.

## 2. What Is Not In This Repo

The repo deliberately does not contain:

- `.pt`, `.pth`, `.ckpt` teacher/student/diffusion checkpoints
- `.onnx` exports
- `.npz` teacher rollouts, VAE rollouts, or state-latent datasets
- videos, W&B logs, TensorBoard event files, Isaac/Omni caches
- credentials

Put runtime assets on the 4090 server under local paths such as:

```text
data/motions/
data/teacher_rollouts/
data/vae_rollouts/
data/state_latent/
checkpoints/tracking/
checkpoints/vae/
checkpoints/diffusion/
configs/local/teacher_map.json
outputs/
```

## 3. Top-Level Code Map

```text
src/beyondmimic_repro/contracts/      data schemas and validation
src/beyondmimic_repro/stage2/         VAE distillation and DAgger interfaces
src/beyondmimic_repro/stage3/         state-latent diffusion and guidance
src/beyondmimic_repro/adapters/       Isaac/MuJoCo runtime boundaries
src/beyondmimic_repro/legacy/         old smoke/simplified baselines
scripts/03_teacher_rollout/           teacher asset audit CLI
scripts/04_vae/                       Stage-2 VAE/DAgger CLIs
scripts/05_state_latent/              Stage-3 dataset CLIs
scripts/06_diffusion/                 diffusion training/eval CLIs
scripts/07_guidance/                  offline guidance CLIs
scripts/09_isaac/                     RTX 4090 Isaac entrypoints
configs/stage2/                       Stage-2 default configs
configs/stage3/                       Stage-3 default configs
docs/                                 status, contracts, transfer, inventory
```

## 4. Contracts Layer

### `contracts/action.py`

Purpose:

- Defines the normalized 29-D action contract.
- Policy frontends output normalized action only.
- Backends convert normalized action to PD target/torque.

Main interface:

```python
validate_normalized_action(action, action_dim=29) -> np.ndarray
```

Input:

- `action`: array with trailing shape `[29]` or `[B, 29]`

Output:

- finite `float32` action array

Rejects:

- wrong action dimension
- NaN/Inf

### `contracts/observation.py`

Purpose:

- Defines paper VAE input dimensions.

Important constants:

```text
ENCODER_REFERENCE_DIM = 67
DECODER_PROPRIO_DIM = 96
POLICY_OBSERVATION_DIM = 160
```

Paper encoder input:

```text
reference joint position  29
reference joint velocity  29
anchor position error      3
anchor orientation Rot6D   6
total                     67
```

Paper decoder proprio input:

```text
projected gravity          3
IMU/root twist             6
current joint position    29
current joint velocity    29
previous action           29
total                     96
```

Together with latent `z[32]`, decoder total input is `128`.

### `contracts/teacher_assets.py`

Purpose:

- Loads and validates teacher metadata without hardcoded H20 paths.
- Supports relocating assets with `--data-root` and `--checkpoint-root`.

Main dataclass:

```python
TeacherAssets(
    motion_name,
    checkpoint_path,
    onnx_path,
    teacher_rollout_path,
    motion_file,
    task_name,
    frequency_hz,
    checkpoint_iteration,
    joint_names,
    body_names,
    anchor_body_name,
    checkpoint_sha256,
    onnx_sha256,
    motion_sha256,
)
```

Main interfaces:

```python
load_teacher_map(path, data_root=None, checkpoint_root=None) -> dict[str, TeacherAssets]
validate_teacher_assets(assets, require_files=False, expected_hz=None) -> list[str]
```

CLI:

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py \
  --teacher-map configs/local/teacher_map.json \
  --data-root data \
  --checkpoint-root checkpoints/tracking \
  --require-files \
  --output-dir outputs/teacher_asset_audit
```

Input:

- JSON teacher map
- optional relocated data/checkpoint roots

Output:

- `outputs/teacher_asset_audit/teacher_asset_audit.json`

Use this first on the 4090 server.

### `contracts/teacher_rollout.py`

Purpose:

- Defines D0 teacher closed-loop rollout schema.
- Teacher rollout is only warm-start data, not final DAgger.

Required arrays:

```text
policy_observation [T, E, obs_dim]
teacher_action     [T, E, 29]
```

Main interfaces:

```python
save_teacher_rollout_contract(path, payload, metadata)
load_teacher_rollout_contract(path) -> (payload, metadata)
```

### `contracts/dagger_dataset.py`

Purpose:

- Defines real DAgger aggregated dataset schema.
- This is student-visited states plus teacher labels.

Schema version:

```text
stage2-dagger-v1
```

Required sample arrays include:

```text
encoder_reference_input [N, 67]
decoder_proprio_input   [N, 96]
student_mu              [N, 32]
student_logvar          [N, 32]
student_latent          [N, 32]
student_action          [N, 29]
teacher_action          [N, 29]
policy_observation      [N, obs_dim]
root_state              [N, root_dim]
joint_position          [N, 29]
joint_velocity          [N, 29]
previous_action         [N, 29]
reward                  [N]
done                    [N]
body_position_error
body_orientation_error
joint_position_error
joint_velocity_error
```

Main interfaces:

```python
validate_dagger_dataset(payload)
save_dagger_dataset(path, payload, metadata)
load_dagger_dataset(path) -> (payload, metadata)
merge_dagger_rounds([d0, d1, ...], output_path)
```

CLI:

```bash
python scripts/04_vae/merge_dagger_rounds.py \
  --round data/dagger/D0.npz \
  --round data/dagger/D1.npz \
  --output data/dagger/aggregated.npz
```

### `contracts/vae_rollout.py`

Purpose:

- Defines trained VAE student closed-loop rollout.
- This is the correct source for Stage-3 data.

Required arrays:

```text
actual_state    [E, T, state_dim]
latent          [E, T, 32]
clean_action    [E, T, 29]
executed_action [E, T, 29]
accepted        [E, T]
episode_id
time_index
```

Main interfaces:

```python
save_vae_rollout(path, payload, metadata)
load_vae_rollout(path) -> (payload, metadata)
```

### `contracts/state_latent.py`

Purpose:

- Defines Stage-3 state-latent trajectory dataset.
- Enforces that paper-faithful data comes from VAE rollout.

Schema version:

```text
stage3-state-latent-v1
```

Default window:

```text
past_steps = 4
include_current = True
future_steps = 16
sequence_length = 21
```

Required arrays:

```text
states       [N, 21, state_dim]
latents      [N, 21, 32]
tokens       [N, 21, state_dim + 32]
valid_mask   [N, 21]
episode_id   [N]
motion_id    [N]
time_index   [N]
frequency_hz [N]
```

## 5. Stage-2 Code

### `stage2/models/conditional_action_vae.py`

Purpose:

- Paper-style VAE, not the old state-action VAE.

Main class:

```python
PaperConditionalActionVAE(PaperVAEConfig)
```

Important methods:

```python
encode(encoder_reference_input[*, 67]) -> (mu[*, 32], logvar[*, 32])
decode(latent[*, 32], decoder_proprio_input[*, 96]) -> action[*, 29]
forward(reference[*, 67], proprio[*, 96]) -> (action, mu, logvar, latent)
```

Training loss:

```python
paper_vae_loss(predicted_action, teacher_action, mu, logvar, kl_coefficient=0.01)
```

Config:

```text
configs/stage2/vae_paper.yaml
```

### `stage2/dagger/interfaces.py`

Purpose:

- Defines abstract protocol for student, teacher, and backend.

Important types:

```python
RobotState(
    policy_observation,
    root_state,
    joint_position,
    joint_velocity,
    previous_action,
)

StudentPolicy.act(reference_input, robot_state, previous_action) -> action[29]
TeacherPolicy.label(robot_state, reference_frame_index) -> action[29]
RobotBackend.apply_normalized_action(action[29]) -> None
```

### `stage2/dagger/collector_core.py`

Purpose:

- Implements correct DAgger semantics.

Important rule:

```text
student action is executed
teacher action is only supervision
```

Main interface:

```python
collect_dagger_steps(
    backend,
    student,
    teacher,
    reference_inputs[T, 67],
    start_reference_frame=0,
) -> list[DAggerCollectionStep]
```

Output row contains:

```text
student_action
teacher_action
policy_observation
joint_position
joint_velocity
reference_frame_index
```

### `stage2/rollout/ou_noise.py`

Purpose:

- Reproducible Ornstein-Uhlenbeck action perturbation.

Main interface:

```python
noise = OrnsteinUhlenbeckNoise(theta=0.8, mu=0.0, sigma=0.1, dt=0.02, seed=...)
noise.reset(batch_size, action_dim, device)
noise.step(reset_mask=None) -> noise[batch_size, action_dim]
noise.reset_mask(mask[batch_size])
```

Use during VAE rollout:

```text
clean_action + ou_noise = executed_action
```

### `stage2/rollout/acceptance.py`

Purpose:

- Encodes VAE rollout acceptance rule.

Rule:

```text
execute noisy action for 2.5 s
continue closed-loop validation to 5.0 s
if failure before 5.0 s, reject episode
```

Main interface:

```python
evaluate_rollout_acceptance(survival_time_s, failed, rejection_reason)
```

Output:

```text
accepted
rejection_reason
survival_time
```

### `stage2/inference/vae_policy.py`

Purpose:

- Runtime frontend that returns normalized action only.

Main interface:

```python
VAEPolicyFrontend(model).act(reference_input, robot_state, previous_action) -> action[29]
```

Backend must convert action to PD target/torque.

## 6. Stage-2 Scripts

### D0 BC warm start

```bash
python scripts/04_vae/train_vae_bc_warmstart.py \
  --teacher-rollout data/teacher_rollouts/D0.npz \
  --config configs/stage2/vae_paper.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_bc_walk1
```

Input:

- teacher rollout D0

Output:

- VAE checkpoint once training body is connected
- summary JSON

This prints:

```text
This is offline BC warm start, not a completed DAgger distillation.
```

### DAgger update

```bash
python scripts/04_vae/train_vae_dagger.py \
  --dagger-dataset data/dagger/aggregated.npz \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --config configs/stage2/dagger_walk1_50hz.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_dagger_update
```

Input:

- aggregated DAgger NPZ
- existing VAE checkpoint

Output:

- updated VAE checkpoint once training body is connected
- summary JSON

### Isaac DAgger collection

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --device cuda:0 \
  --output-dir outputs/dagger_D1
```

Input:

- teacher map
- VAE checkpoint
- Isaac task/runtime

Output expected on 4090:

- DAgger round NPZ matching `stage2-dagger-v1`
- local logs/summary

H20 did not run this.

## 7. Stage-3 Representation

### `stage3/representation/character_frame.py`

Purpose:

- Character-yaw-centric transforms.

Main interfaces:

```python
world_to_character_yaw(points_w[..., 3], origin_w[3], yaw_w) -> points_c[..., 3]
character_yaw_to_world(points_c[..., 3], origin_w[3], yaw_w) -> points_w[..., 3]
build_character_yaw_state_window(...) -> state_window[T, state_dim]
```

### `stage3/representation/trajectory_state.py`

Purpose:

- Makes timing explicit.

Main helpers:

```python
paper_time_window_50hz() -> sequence_length 21, future horizon 0.32 s
paper_time_window_25hz() -> sequence_length 21, future horizon 0.64 s
```

### `stage3/representation/emphasis_projection.py`

Purpose:

- Builds and saves paper-style emphasis projection.
- Do not randomly regenerate projection at inference.

Main interfaces:

```python
build_projection_matrix(seed, schema) -> {A, B, P, P_inv, schema, state_schema_hash}
apply_projection(states, P) -> projected_states
apply_inverse_projection(projected_states, P_inv) -> states
save_projection(path, payload)
load_projection(path)
```

## 8. Stage-3 Dataset Code

### `stage3/datasets/state_latent_builder.py`

Purpose:

- Builds Stage-3 data from VAE closed-loop rollout.

Main interface:

```python
build_from_vae_rollout(
    vae_rollout_path,
    output_path,
    metadata=None,
) -> summary
```

Input:

- VAE rollout NPZ with actual state and latent

Output:

- state-latent NPZ with `states`, `latents`, `tokens`, masks, IDs, frequency

CLI:

```bash
python scripts/05_state_latent/build_from_vae_rollout.py \
  --vae-rollout data/vae_rollouts/vae_walk1_ou.npz \
  --output data/state_latent/walk1_state_latent.npz \
  --output-dir outputs/state_latent_build
```

Legacy warning:

- Direct teacher-rollout encoding is preserved only as legacy and is not the paper-faithful Stage-3 path.

## 9. Stage-3 Diffusion Code

### `stage3/diffusion/noising.py`

Purpose:

- Adds per-token state/latent noise.
- Builds x0 or epsilon training target.

Main interfaces:

```python
add_per_token_noise(
    clean_states[B, T, state_dim],
    clean_latents[B, T, latent_dim],
    noise_states[B, T, state_dim],
    noise_latents[B, T, latent_dim],
    k_state[B, T],
    k_latent[B, T],
    alpha_bars,
) -> noisy_tokens[B, T, state_dim + latent_dim]

construct_training_target(clean_tokens, noise_tokens, prediction_type="x0")
apply_inpainting_mask(noisy, clean, mask)
```

Default:

```text
prediction_type = x0
```

Legacy:

```text
prediction_type = epsilon
```

### `stage3/models/state_latent_transformer.py`

Purpose:

- Transformer denoiser with separate state/latent diffusion step embeddings.

Main interface:

```python
model = StateLatentTransformer(
    token_dim,
    sequence_length=21,
    denoising_steps=20,
    embedding_dim=512,
    attention_heads=8,
    transformer_layers=6,
)

model(noisy_tokens[B, T, token_dim], diffusion_steps[B, T, 2]) -> prediction[B, T, token_dim]
```

`diffusion_steps[..., 0]` is `k_state`.
`diffusion_steps[..., 1]` is `k_latent`.

### `stage3/diffusion/ema.py`

Purpose:

- Maintains EMA shadow weights.

Main interface:

```python
ema = ExponentialMovingAverage(model, power=0.75, max_decay=0.9999)
ema.update(model)
ema.state_dict()
```

### `stage3/diffusion/sampler.py`

Purpose:

- Defines guided sampling contract.
- Output is a planned trajectory; runtime should execute only current latent.

Main interfaces:

```python
guided_sample(
    denoiser,
    noisy_trajectory,
    diffusion_steps,
    conditioning,
    guidance_costs,
    guidance_weights,
    masks,
    guidance_scale=1.0,
    gradient_clip_norm=1.0,
) -> (predicted_tokens, diagnostics)

extract_current_latent(predicted_tokens, state_dim, current_index=4) -> z_current[B, latent_dim]
```

Important:

- Do not open-loop play the whole diffusion trajectory.
- Use `z_current -> VAE decoder -> normalized action`.

### Diffusion training CLI

```bash
python scripts/06_diffusion/train_state_latent_diffusion.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --config configs/stage3/diffusion_engineering_50hz.yaml \
  --prediction-type x0 \
  --device cuda:0 \
  --output-dir outputs/diffusion_walk1
```

Input:

- state-latent NPZ
- config

Output:

- diffusion checkpoint once training body is connected
- summary JSON

Configs:

```text
configs/stage3/diffusion_engineering_50hz.yaml
configs/stage3/diffusion_paper_25hz.yaml
```

Use 50 Hz for current engineering gate. Use 25 Hz only when data is truly sampled/validated at 25 Hz.

## 10. Stage-3 Guidance Code

All guidance costs are Torch-differentiable and return:

```python
cost[B], diagnostics = guidance_cost(predicted_trajectory[B, T, D], context)
```

### `guidance/joystick.py`

Purpose:

- Track planar root velocity across future horizon.

Input context:

```python
{
  "target_velocity_xy": torch.tensor([vx, vy]),
  "dt": 0.02,
  "horizon_weights": optional
}
```

### `guidance/waypoint.py`

Purpose:

- Move final/root trajectory toward waypoint.

Input context:

```python
{
  "waypoint_xy": torch.tensor([x, y]),
  "velocity_weight": optional
}
```

### `guidance/obstacle.py`

Purpose:

- SDF + relaxed barrier obstacle cost.

Input context:

```python
{
  "obstacle_xy": torch.tensor([x, y]),
  "radius": 0.2,
  "delta": 0.1
}
```

### `guidance/inpainting.py`

Purpose:

- Enforce selected keyframes/body/state dimensions.

Input context:

```python
{
  "target": target_tensor[B, T, D],
  "mask": mask_tensor[B, T, D]
}
```

### `guidance/composition.py`

Purpose:

- Weighted sum of multiple guidance objectives.

Input context:

```python
{
  "objectives": [
    ("joystick", joystick_guidance_cost, joystick_context, 1.0),
    ("waypoint", waypoint_guidance_cost, waypoint_context, 0.5),
  ]
}
```

Offline CLI examples:

```bash
python scripts/07_guidance/eval_guidance_offline.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --guidance-config configs/stage3/guidance_velocity.yaml \
  --output-dir outputs/guidance_offline

python scripts/07_guidance/sample_velocity_guided.py \
  --diffusion-checkpoint checkpoints/diffusion/diffusion_latest.pt \
  --guidance-config configs/stage3/guidance_velocity.yaml \
  --device cuda:0 \
  --output-dir outputs/sample_velocity
```

## 11. Policy Frontends And Backends

### Stage-2 frontend

```python
VAEPolicyFrontend.act(reference_input, robot_state, previous_action) -> normalized_action[29]
```

### Stage-3 frontend

```python
DiffusionVAEPolicyFrontend.act(history, robot_state, task_context) -> normalized_action[29]
```

Meaning:

- Frontends never output torque.
- Frontends never directly set root pose.
- Frontends only output normalized 29-D actions.

### Backend responsibility

The backend converts normalized action to control:

```text
target_joint_pos = default_joint_pos + action_scale * action
torque = Kp * (target_joint_pos - joint_pos) - Kd * joint_vel
torque = clip(torque)
physics step
```

Code:

```text
src/beyondmimic_repro/adapters/mujoco/shared_controller_contract.py
src/beyondmimic_repro/adapters/mujoco/robot_backend_protocol.py
```

Before any claim, validate:

- joint name order
- qpos/qvel mapping
- default pose
- action scale
- stiffness/damping
- control dt and simulation dt
- quaternion convention
- anchor yaw/position alignment

## 12. Isaac Adapter Entry Points

These scripts are for the 4090 Isaac host:

```text
scripts/09_isaac/validate_teacher_checkpoint.py
scripts/09_isaac/collect_dagger_round.py
scripts/09_isaac/eval_vae_closed_loop.py
scripts/09_isaac/collect_vae_rollout.py
scripts/09_isaac/eval_diffusion_closed_loop.py
scripts/09_isaac/eval_velocity_guidance.py
```

They support:

```text
--task-name
--num-envs
--device
--teacher-map
--vae-checkpoint
--diffusion-checkpoint
--motion-file
--output-dir
--headless / --no-headless
--dry-run
```

They are designed so Isaac imports happen after AppLauncher startup. If Isaac
is unavailable, they fail clearly instead of silently using fake simulation.

Dry-run:

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --dry-run \
  --output-dir outputs/isaac_dry_run
```

Real VAE rollout collection:

```bash
python scripts/09_isaac/collect_vae_rollout.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_rollout_walk1
```

Expected real output:

- VAE rollout NPZ matching `stage2-vae-rollout-v1`
- acceptance/rejection summary
- local logs

The current branch supplies the interface and import boundary. The 4090 server
must connect the task-specific tensor extraction/action application details.

## 13. Legacy Baselines

Old public simplified code is preserved under:

```text
src/beyondmimic_repro/legacy/
```

Aliases:

```python
LegacyConditionalActionVAE
LegacyStateLatentTeacherEncoder
LegacyEpsilonDenoiser
```

These are not paper-faithful:

- legacy VAE encodes state-action pairs
- legacy state-latent builder encodes teacher rollout directly
- legacy denoiser predicts epsilon by default

Use these only for smoke tests or ablation comparison, not final claims.

## 14. Recommended 4090 Work Order

### Step 1: Audit environment

```bash
python scripts/00_setup/check_environment.py
python -m compileall src scripts
python -m pytest -q
```

### Step 2: Create local teacher map

Create `configs/local/teacher_map.json` that points to local files.

Minimum shape:

```json
{
  "teachers": [
    {
      "motion_name": "walk1",
      "checkpoint_path": "walk1_model30000_official.pt",
      "onnx_path": "walk1_model30000_official.onnx",
      "teacher_rollout_path": "walk1_D0_teacher_rollout.npz",
      "motion_file": "walk1_motion.npz",
      "task_name": "Tracking-Flat-G1-v0",
      "frequency_hz": 50.0,
      "checkpoint_iteration": 30000,
      "joint_names": [],
      "body_names": [],
      "anchor_body_name": "pelvis"
    }
  ]
}
```

Then audit:

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py \
  --teacher-map configs/local/teacher_map.json \
  --data-root data \
  --checkpoint-root checkpoints/tracking \
  --require-files \
  --output-dir outputs/teacher_asset_audit
```

### Step 3: Validate teacher checkpoint in Isaac

```bash
python scripts/09_isaac/validate_teacher_checkpoint.py \
  --teacher-map configs/local/teacher_map.json \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 128 \
  --device cuda:0 \
  --output-dir outputs/teacher_validate
```

Only after this can the server say the teacher asset loads in Isaac.

### Step 4: D0 VAE warm start

```bash
python scripts/04_vae/train_vae_bc_warmstart.py \
  --teacher-rollout data/teacher_rollouts/walk1_D0_teacher_rollout.npz \
  --config configs/stage2/vae_paper.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D0
```

This is only BC warm start.

### Step 5: Collect real DAgger round

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --device cuda:0 \
  --output-dir outputs/dagger_D1
```

Then merge:

```bash
python scripts/04_vae/merge_dagger_rounds.py \
  --round data/dagger/D0.npz \
  --round outputs/dagger_D1/dagger_round.npz \
  --output data/dagger/aggregated_D0_D1.npz
```

Then update VAE:

```bash
python scripts/04_vae/train_vae_dagger.py \
  --dagger-dataset data/dagger/aggregated_D0_D1.npz \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --config configs/stage2/dagger_walk1_50hz.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D1
```

### Step 6: Evaluate VAE closed loop

```bash
python scripts/09_isaac/eval_vae_closed_loop.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D1.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_closed_loop
```

Only after this passes can you claim VAE closed-loop evidence.

### Step 7: Collect accepted VAE rollout

```bash
python scripts/09_isaac/collect_vae_rollout.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D1.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_rollout_ou
```

Output must include actual state, latent, clean action, executed action,
acceptance, rejection reason, and survival time.

### Step 8: Build Stage-3 state-latent data

```bash
python scripts/05_state_latent/build_from_vae_rollout.py \
  --vae-rollout outputs/vae_rollout_ou/vae_rollout.npz \
  --output data/state_latent/walk1_state_latent.npz \
  --output-dir outputs/state_latent_build
```

Audit:

```bash
python scripts/05_state_latent/audit_state_latent_dataset.py \
  --dataset data/state_latent/walk1_state_latent.npz \
  --output-dir outputs/state_latent_audit
```

### Step 9: Train diffusion

```bash
python scripts/06_diffusion/train_state_latent_diffusion.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --config configs/stage3/diffusion_engineering_50hz.yaml \
  --prediction-type x0 \
  --device cuda:0 \
  --output-dir outputs/diffusion_train
```

### Step 10: Offline diffusion/guidance checks

```bash
python scripts/06_diffusion/eval_denoising_offline.py \
  --diffusion-checkpoint checkpoints/diffusion/diffusion_latest.pt \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --device cuda:0 \
  --output-dir outputs/diffusion_eval

python scripts/07_guidance/eval_guidance_offline.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --guidance-config configs/stage3/guidance_velocity.yaml \
  --output-dir outputs/guidance_offline
```

### Step 11: Isaac guided diffusion closed loop

```bash
python scripts/09_isaac/eval_velocity_guidance.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D1.pt \
  --diffusion-checkpoint checkpoints/diffusion/diffusion_latest.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 512 \
  --device cuda:0 \
  --output-dir outputs/velocity_guidance_eval
```

Only after this passes can you claim guided diffusion closed-loop evidence.

## 15. What To Report After 4090 Runs

Report these separately:

- teacher checkpoint loads in Isaac
- DAgger round collected
- VAE BC warm start trained
- VAE DAgger update trained
- VAE closed-loop survival metrics
- VAE OU rollout acceptance rate
- state-latent dataset built from VAE rollout
- diffusion offline denoising metrics
- guidance offline gradients/costs
- guided diffusion closed-loop metrics

Do not report these unless actually validated:

- completed paper-faithful DAgger
- VAE Isaac stable walking
- guided diffusion closed-loop success
- walk-to-run reproduction
- paper Fig. 5 reproduction

## 16. Where To Look First If Something Breaks

Teacher map/path errors:

- `src/beyondmimic_repro/contracts/teacher_assets.py`
- `scripts/03_teacher_rollout/audit_teacher_assets.py`

Observation/action shape errors:

- `contracts/observation.py`
- `contracts/action.py`
- `stage2/models/conditional_action_vae.py`

DAgger semantic errors:

- `stage2/dagger/collector_core.py`
- verify student action, not teacher action, is executed

State-latent source errors:

- `contracts/vae_rollout.py`
- `contracts/state_latent.py`
- `stage3/datasets/state_latent_builder.py`

Diffusion shape/noise errors:

- `stage3/diffusion/noising.py`
- `stage3/models/state_latent_transformer.py`

Guidance gradient problems:

- `stage3/guidance/*.py`
- check finite gradients and no accidental detach

Isaac import/runtime errors:

- `adapters/isaac/contracts.py`
- ensure AppLauncher starts before task imports

Controller mismatch:

- `adapters/mujoco/shared_controller_contract.py`
- verify joint order, action scale, Kp/Kd, qpos/qvel mapping

