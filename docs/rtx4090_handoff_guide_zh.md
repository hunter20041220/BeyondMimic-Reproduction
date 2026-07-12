# RTX 4090 Stage-2/Stage-3 中文交接说明

这份文档给另一台 RTX 4090 服务器使用。它不依赖当前聊天记录，clone 仓库后按本文执行即可继续真正的 Isaac Lab DAgger、VAE rollout 和 guided diffusion 验证。

最重要的真实性边界：

- H20 已完成：代码整理、接口定义、schema、配置、CPU synthetic tests、迁移入口。
- H20 未完成：Isaac closed-loop DAgger、VAE closed-loop、VAE rollout 采集、guided diffusion closed-loop。
- 4090 上跑通前，不要宣称 DAgger 完成、VAE 稳定行走、guided diffusion 成功或复现论文 Fig. 5。

## 1. 拉代码和安装

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

4090 服务器还需要安装：

- Isaac Sim
- Isaac Lab
- `whole_body_tracking` 任务包
- 本地 teacher checkpoints、motion 文件、rollout 数据

## 2. 仓库里有什么，不包括什么

仓库包括：

- Stage-2/3 的 Python package 代码
- 数据契约和 shape/NaN/schema 检查
- CLI 脚本
- 50 Hz/25 Hz 配置
- 4090 Isaac adapter 入口
- legacy baseline 标识
- 测试和文档

仓库不包括：

- `.pt/.pth/.ckpt` 权重
- `.onnx` 导出
- `.npz` rollout/dataset
- 视频
- W&B/TensorBoard 日志
- Isaac/Omni cache
- GitHub token 或其他凭证

推荐在 4090 放置：

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

## 3. 总体目录说明

```text
src/beyondmimic_repro/contracts/   所有数据和资产契约
src/beyondmimic_repro/stage2/      VAE、DAgger、OU rollout、VAE frontend
src/beyondmimic_repro/stage3/      state-latent、diffusion、guidance、diffusion frontend
src/beyondmimic_repro/adapters/    Isaac/MuJoCo 接入边界
src/beyondmimic_repro/legacy/      旧版 smoke/simplified baseline
scripts/03_teacher_rollout/        teacher asset 审计
scripts/04_vae/                    VAE/DAgger 脚本
scripts/05_state_latent/           state-latent 数据脚本
scripts/06_diffusion/              diffusion 训练/评估脚本
scripts/07_guidance/               guidance 离线脚本
scripts/09_isaac/                  4090 Isaac 入口
configs/stage2/                    Stage-2 配置
configs/stage3/                    Stage-3 配置
docs/                              交接、状态、inventory、迁移说明
```

## 4. contracts：数据和资产接口

### `contracts/action.py`

作用：统一动作含义。

核心接口：

```python
validate_normalized_action(action, action_dim=29) -> np.ndarray
```

输入：

- `action[..., 29]`
- normalized action，不是 torque，不是 PD target

输出：

- finite `float32` action

错误：

- 维度不是 29
- 有 NaN/Inf

### `contracts/observation.py`

作用：固定论文式 VAE 的输入维度。

Encoder 输入 67 维：

```text
reference joint position  29
reference joint velocity  29
anchor position error      3
anchor orientation Rot6D   6
```

Decoder proprio 输入 96 维：

```text
projected gravity          3
IMU/root twist             6
current joint position    29
current joint velocity    29
previous action           29
```

Decoder 实际输入是：

```text
latent z 32 + proprio 96 = 128
```

### `contracts/teacher_assets.py`

作用：读取 `teacher_map.json`，并把 H20 绝对路径迁移成 4090 本地路径。

核心结构：

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

核心接口：

```python
load_teacher_map(path, data_root=None, checkpoint_root=None)
validate_teacher_assets(assets, require_files=False, expected_hz=None)
```

用法：

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py \
  --teacher-map configs/local/teacher_map.json \
  --data-root data \
  --checkpoint-root checkpoints/tracking \
  --require-files \
  --output-dir outputs/teacher_asset_audit
```

输出：

```text
outputs/teacher_asset_audit/teacher_asset_audit.json
```

### `contracts/teacher_rollout.py`

作用：定义 D0 teacher rollout。它只能作为 BC warm start，不等于 DAgger 完成。

必须包含：

```text
policy_observation [T, E, obs_dim]
teacher_action     [T, E, 29]
```

核心接口：

```python
save_teacher_rollout_contract(path, payload, metadata)
load_teacher_rollout_contract(path)
```

### `contracts/dagger_dataset.py`

作用：定义真正 DAgger aggregated dataset。

schema：

```text
stage2-dagger-v1
```

含义：

```text
D0 union D1 union D2 ...
student 自己走到的状态 + teacher 对这些 student state 的 action label
```

关键数组：

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
```

核心接口：

```python
validate_dagger_dataset(payload)
save_dagger_dataset(path, payload, metadata)
load_dagger_dataset(path)
merge_dagger_rounds([d0, d1, ...], output_path)
```

CLI 合并：

```bash
python scripts/04_vae/merge_dagger_rounds.py \
  --round data/dagger/D0.npz \
  --round data/dagger/D1.npz \
  --output data/dagger/aggregated.npz
```

### `contracts/vae_rollout.py`

作用：定义 trained VAE student closed-loop rollout。这是 Stage-3 的正确数据源。

必须包含：

```text
actual_state    [E, T, state_dim]
latent          [E, T, 32]
clean_action    [E, T, 29]
executed_action [E, T, 29]
accepted        [E, T]
episode_id
time_index
```

核心接口：

```python
save_vae_rollout(path, payload, metadata)
load_vae_rollout(path)
```

### `contracts/state_latent.py`

作用：定义 Stage-3 state-latent dataset。

schema：

```text
stage3-state-latent-v1
```

默认时间窗口：

```text
past_steps = 4
include_current = True
future_steps = 16
sequence_length = 21
```

必须包含：

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

注意：正式 Stage-3 数据必须来自 VAE rollout，不能直接把 teacher rollout encode 后冒充。

## 5. Stage-2：VAE 和 DAgger

### `stage2/models/conditional_action_vae.py`

作用：论文语义 VAE。

核心类：

```python
PaperConditionalActionVAE(PaperVAEConfig)
```

接口：

```python
encode(reference_input[..., 67]) -> mu[..., 32], logvar[..., 32]
decode(latent[..., 32], proprio[..., 96]) -> action[..., 29]
forward(reference_input, proprio) -> action, mu, logvar, latent
```

输出：

```text
normalized 29-D action
```

不是 torque，也不是 `default_joint_pos + action_scale * action`。

配置：

```text
configs/stage2/vae_paper.yaml
```

### `stage2/dagger/interfaces.py`

作用：定义 student、teacher、robot backend 的协议。

核心概念：

```python
RobotState(
    policy_observation,
    root_state,
    joint_position,
    joint_velocity,
    previous_action,
)
```

接口：

```python
StudentPolicy.act(reference_input, robot_state, previous_action) -> action[29]
TeacherPolicy.label(robot_state, reference_frame_index) -> action[29]
RobotBackend.apply_normalized_action(action[29]) -> None
```

### `stage2/dagger/collector_core.py`

作用：DAgger 采集核心语义。

最关键规则：

```text
执行 student_action
teacher_action 只作为监督标签
绝对不能用 teacher_action 接管 rollout
```

接口：

```python
collect_dagger_steps(
    backend,
    student,
    teacher,
    reference_inputs[T, 67],
    start_reference_frame=0,
) -> list[DAggerCollectionStep]
```

每个 step 输出：

```text
student_action
teacher_action
policy_observation
joint_position
joint_velocity
reference_frame_index
```

### `stage2/rollout/ou_noise.py`

作用：VAE rollout 时给 action 加 OU noise。

接口：

```python
noise = OrnsteinUhlenbeckNoise(theta=0.8, mu=0.0, sigma=0.1, dt=0.02, seed=...)
noise.reset(batch_size, action_dim, device)
noise.step(reset_mask=None) -> noise[batch_size, action_dim]
noise.reset_mask(mask)
```

执行动作：

```text
executed_action = clean_action + ou_noise
```

### `stage2/rollout/acceptance.py`

作用：VAE rollout 接收/拒绝规则。

规则：

```text
带噪 action 执行 2.5 s
继续闭环检查到 5.0 s
5.0 s 前失败则拒绝
```

接口：

```python
evaluate_rollout_acceptance(survival_time_s, failed, rejection_reason)
```

输出：

```text
accepted
rejection_reason
survival_time
```

### `stage2/inference/vae_policy.py`

作用：VAE runtime frontend。

接口：

```python
VAEPolicyFrontend(model).act(reference_input, robot_state, previous_action) -> action[29]
```

只输出 normalized action，后端负责 PD target 和 torque。

## 6. Stage-2 脚本怎么用

### D0 BC warm start

```bash
python scripts/04_vae/train_vae_bc_warmstart.py \
  --teacher-rollout data/teacher_rollouts/D0.npz \
  --config configs/stage2/vae_paper.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D0
```

输入：

- teacher rollout D0

输出：

- 初始 VAE checkpoint
- summary JSON

语义：

- 这是 offline BC warm start
- 不是完整 DAgger

### DAgger update

```bash
python scripts/04_vae/train_vae_dagger.py \
  --dagger-dataset data/dagger/aggregated.npz \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --config configs/stage2/dagger_walk1_50hz.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D1
```

输入：

- aggregated DAgger NPZ
- 已有 VAE checkpoint

输出：

- updated VAE checkpoint
- summary JSON

### Isaac DAgger collection

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --device cuda:0 \
  --output-dir outputs/dagger_D1
```

输入：

- teacher map
- student VAE checkpoint
- Isaac task/runtime

输出：

- DAgger round NPZ
- summary/logs

H20 没跑过这个，必须在 4090 验证。

## 7. Stage-3：state-latent、diffusion、guidance

### `stage3/representation/character_frame.py`

作用：character-yaw-centric 坐标变换。

接口：

```python
world_to_character_yaw(points_w[..., 3], origin_w[3], yaw_w) -> points_c[..., 3]
character_yaw_to_world(points_c[..., 3], origin_w[3], yaw_w) -> points_w[..., 3]
build_character_yaw_state_window(...) -> state_window[T, state_dim]
```

用途：

- root future/history 都相对当前 character yaw frame
- body position/velocity 在每个时间步自己的 local root/yaw frame

### `stage3/representation/trajectory_state.py`

作用：明确时间窗口，不用模糊 `horizon=20`。

```python
paper_time_window_50hz() -> 21 steps, future 0.32 s
paper_time_window_25hz() -> 21 steps, future 0.64 s
```

### `stage3/representation/emphasis_projection.py`

作用：emphasis projection 的构建、保存、读取。

接口：

```python
build_projection_matrix(seed, schema) -> {A, B, P, P_inv, schema, hash}
apply_projection(states, P)
apply_inverse_projection(projected_states, P_inv)
save_projection(path, payload)
load_projection(path)
```

注意：

- 训练和推理必须读取同一个 projection
- 不要每次推理随机重建

### `stage3/datasets/state_latent_builder.py`

作用：从 VAE rollout 构建 Stage-3 数据。

接口：

```python
build_from_vae_rollout(vae_rollout_path, output_path, metadata=None)
```

CLI：

```bash
python scripts/05_state_latent/build_from_vae_rollout.py \
  --vae-rollout outputs/vae_rollout_ou/vae_rollout.npz \
  --output data/state_latent/walk1_state_latent.npz \
  --output-dir outputs/state_latent_build
```

输入：

- VAE closed-loop rollout NPZ

输出：

- state-latent NPZ

### `stage3/diffusion/noising.py`

作用：per-token diffusion noising 和训练 target。

接口：

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

默认：

```text
prediction_type = x0
```

legacy 支持：

```text
prediction_type = epsilon
```

### `stage3/models/state_latent_transformer.py`

作用：Transformer denoiser。

接口：

```python
model = StateLatentTransformer(
    token_dim,
    sequence_length=21,
    denoising_steps=20,
    embedding_dim=512,
    attention_heads=8,
    transformer_layers=6,
)

model(noisy_tokens[B, T, token_dim], diffusion_steps[B, T, 2])
  -> prediction[B, T, token_dim]
```

其中：

```text
diffusion_steps[..., 0] = k_state
diffusion_steps[..., 1] = k_latent
```

### `stage3/diffusion/sampler.py`

作用：guided sampling 合约。

接口：

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
) -> predicted_tokens, diagnostics

extract_current_latent(predicted_tokens, state_dim, current_index=4) -> z_current
```

重要：

- diffusion 规划一段 trajectory
- 实际控制时只取当前 latent
- `z_current -> VAE decoder -> normalized action`
- 不要整段 open-loop 播放 diffusion trajectory

### diffusion 训练

```bash
python scripts/06_diffusion/train_state_latent_diffusion.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --config configs/stage3/diffusion_engineering_50hz.yaml \
  --prediction-type x0 \
  --device cuda:0 \
  --output-dir outputs/diffusion_train
```

配置：

```text
configs/stage3/diffusion_engineering_50hz.yaml  # 当前工程 gate
configs/stage3/diffusion_paper_25hz.yaml        # paper-faithful timing
```

## 8. guidance 接口

所有 guidance cost 统一：

```python
cost[B], diagnostics = guidance_cost(predicted_trajectory[B, T, D], context)
```

### `guidance/joystick.py`

目标：未来 horizon 的 planar root velocity tracking。

context：

```python
{
  "target_velocity_xy": torch.tensor([vx, vy]),
  "dt": 0.02,
  "horizon_weights": optional,
}
```

### `guidance/waypoint.py`

目标：到达 waypoint。

context：

```python
{
  "waypoint_xy": torch.tensor([x, y]),
  "velocity_weight": optional,
}
```

### `guidance/obstacle.py`

目标：SDF + relaxed barrier 避障。

context：

```python
{
  "obstacle_xy": torch.tensor([x, y]),
  "radius": 0.2,
  "delta": 0.1,
}
```

### `guidance/inpainting.py`

目标：约束历史/current/future keyframe。

context：

```python
{
  "target": target_tensor[B, T, D],
  "mask": mask_tensor[B, T, D],
}
```

### `guidance/composition.py`

目标：多个 guidance 加权组合。

context：

```python
{
  "objectives": [
    ("joystick", joystick_guidance_cost, joystick_context, 1.0),
    ("waypoint", waypoint_guidance_cost, waypoint_context, 0.5),
  ]
}
```

## 9. Controller 和 backend

frontend 只输出：

```text
normalized action [29]
```

backend 负责：

```text
target_joint_pos = default_joint_pos + action_scale * action
torque = Kp * (target_joint_pos - joint_pos) - Kd * joint_vel
torque clipping
physics step
```

代码：

```text
src/beyondmimic_repro/adapters/mujoco/shared_controller_contract.py
src/beyondmimic_repro/adapters/mujoco/robot_backend_protocol.py
```

4090 上必须检查：

- joint_names 顺序
- qpos/qvel mapping
- default pose
- action_scale
- stiffness/damping
- torque limit
- control_dt/simulation_dt
- quaternion convention
- anchor yaw/position alignment

## 10. Isaac entrypoints

脚本：

```text
scripts/09_isaac/validate_teacher_checkpoint.py
scripts/09_isaac/collect_dagger_round.py
scripts/09_isaac/eval_vae_closed_loop.py
scripts/09_isaac/collect_vae_rollout.py
scripts/09_isaac/eval_diffusion_closed_loop.py
scripts/09_isaac/eval_velocity_guidance.py
```

共同参数：

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

设计原则：

- 先启动 Isaac AppLauncher
- 再 import Isaac tasks
- 没有 Isaac runtime 就清楚报错
- 不做 synthetic fallback

dry run：

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_latest.pt \
  --dry-run \
  --output-dir outputs/isaac_dry_run
```

## 11. 推荐 4090 执行顺序

1. 环境检查

```bash
python scripts/00_setup/check_environment.py
python -m compileall src scripts
python -m pytest -q
```

2. 准备并审计 teacher map

```bash
python scripts/03_teacher_rollout/audit_teacher_assets.py \
  --teacher-map configs/local/teacher_map.json \
  --data-root data \
  --checkpoint-root checkpoints/tracking \
  --require-files \
  --output-dir outputs/teacher_asset_audit
```

3. Isaac 中验证 teacher checkpoint

```bash
python scripts/09_isaac/validate_teacher_checkpoint.py \
  --teacher-map configs/local/teacher_map.json \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 128 \
  --device cuda:0 \
  --output-dir outputs/teacher_validate
```

4. D0 VAE warm start

```bash
python scripts/04_vae/train_vae_bc_warmstart.py \
  --teacher-rollout data/teacher_rollouts/walk1_D0_teacher_rollout.npz \
  --config configs/stage2/vae_paper.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D0
```

5. 采集 DAgger D1

```bash
python scripts/09_isaac/collect_dagger_round.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 4096 \
  --device cuda:0 \
  --output-dir outputs/dagger_D1
```

6. 合并 DAgger 并训练更新 VAE

```bash
python scripts/04_vae/merge_dagger_rounds.py \
  --round data/dagger/D0.npz \
  --round outputs/dagger_D1/dagger_round.npz \
  --output data/dagger/aggregated_D0_D1.npz

python scripts/04_vae/train_vae_dagger.py \
  --dagger-dataset data/dagger/aggregated_D0_D1.npz \
  --vae-checkpoint checkpoints/vae/vae_D0.pt \
  --config configs/stage2/dagger_walk1_50hz.yaml \
  --device cuda:0 \
  --output-dir outputs/vae_D1
```

7. VAE closed-loop 验证

```bash
python scripts/09_isaac/eval_vae_closed_loop.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D1.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_closed_loop
```

8. 采集 VAE OU rollout

```bash
python scripts/09_isaac/collect_vae_rollout.py \
  --teacher-map configs/local/teacher_map.json \
  --vae-checkpoint checkpoints/vae/vae_D1.pt \
  --task-name Tracking-Flat-G1-v0 \
  --num-envs 1024 \
  --device cuda:0 \
  --output-dir outputs/vae_rollout_ou
```

9. 构建 Stage-3 state-latent

```bash
python scripts/05_state_latent/build_from_vae_rollout.py \
  --vae-rollout outputs/vae_rollout_ou/vae_rollout.npz \
  --output data/state_latent/walk1_state_latent.npz \
  --output-dir outputs/state_latent_build
```

10. 训练 diffusion

```bash
python scripts/06_diffusion/train_state_latent_diffusion.py \
  --state-latent-dataset data/state_latent/walk1_state_latent.npz \
  --config configs/stage3/diffusion_engineering_50hz.yaml \
  --prediction-type x0 \
  --device cuda:0 \
  --output-dir outputs/diffusion_train
```

11. guided diffusion closed-loop

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

## 12. legacy 代码怎么理解

legacy 在：

```text
src/beyondmimic_repro/legacy/
```

包含：

```text
LegacyConditionalActionVAE
LegacyStateLatentTeacherEncoder
LegacyEpsilonDenoiser
```

它们只是旧版 smoke/simplified baseline：

- legacy VAE 是 state-action VAE
- legacy state-latent 是 teacher rollout 直接 encode
- legacy denoiser 是 epsilon prediction

不能把 legacy 结果说成论文式 DAgger/VAE/diffusion。

## 13. 4090 跑完后应该汇报什么

分开汇报：

- teacher checkpoint 是否能在 Isaac 加载
- DAgger D1/D2 是否采集成功
- VAE BC warm start loss
- VAE DAgger update loss
- VAE closed-loop survival metrics
- VAE OU rollout acceptance rate
- state-latent dataset shape/schema
- diffusion offline denoising metrics
- guidance offline gradient/cost 是否 finite
- guided diffusion closed-loop metrics

不要提前汇报：

- DAgger 已完成
- VAE 已稳定行走
- guided diffusion 已闭环成功
- walk-to-run 已复现
- Fig. 5 已复现

## 14. 出问题先看哪里

teacher map/path：

```text
contracts/teacher_assets.py
scripts/03_teacher_rollout/audit_teacher_assets.py
```

VAE 输入维度：

```text
contracts/observation.py
stage2/models/conditional_action_vae.py
```

DAgger 执行语义：

```text
stage2/dagger/collector_core.py
```

VAE rollout/state-latent：

```text
contracts/vae_rollout.py
contracts/state_latent.py
stage3/datasets/state_latent_builder.py
```

diffusion shape/noise：

```text
stage3/diffusion/noising.py
stage3/models/state_latent_transformer.py
```

guidance 梯度：

```text
stage3/guidance/
```

Isaac import：

```text
adapters/isaac/contracts.py
```

controller mismatch：

```text
adapters/mujoco/shared_controller_contract.py
```
