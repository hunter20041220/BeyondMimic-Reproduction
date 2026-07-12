# Controller Interfaces

Policy frontends output normalized 29-D actions only.

Stage-2:

```python
VAEPolicyFrontend.act(reference_input, robot_state, previous_action) -> normalized_action
```

Stage-3:

```python
DiffusionVAEPolicyFrontend.act(history, robot_state, task_context) -> normalized_action
```

Backends own the conversion from normalized action to simulator control:

```text
default_joint_pos + action_scale * action
PD target
Kp/Kd
torque clipping
physics stepping
```

The shared backend protocol is in `src/beyondmimic_repro/adapters/mujoco/robot_backend_protocol.py`.

The controller metadata and PD target conversion are in `src/beyondmimic_repro/adapters/mujoco/shared_controller_contract.py`.

The adapter contract records joint-name mapping, qpos/qvel mapping, anchor yaw/position alignment, quaternion convention, default pose, action scale, stiffness/damping, control dt, and simulation dt. Runtime-specific implementations must validate these before closed-loop claims.
