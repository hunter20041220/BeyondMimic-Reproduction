"""MuJoCo runtime smoke for the shared normalized-action controller contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from beyondmimic_repro.adapters.mujoco.shared_controller_contract import (
    JointControllerMetadata,
    normalized_action_to_pd_target,
    pd_torque,
)


_ONE_DOF_XML = """
<mujoco model="beyondmimic_contract_smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
      <geom type="capsule" size="0.03 0.25" fromto="0 0 0 0 0 0.5" density="500"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-4 4"/>
  </actuator>
</mujoco>
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_mujoco_sim_to_sim_smoke(*, output: str | Path, steps: int = 200, action: float = 0.7) -> dict[str, Any]:
    """Run a deterministic MuJoCo PD-control smoke and write a JSON summary.

    This is a runtime gate for the shared controller contract, not a substitute
    for a full G1 XML/URDF sim-to-sim evaluation.
    """

    import mujoco

    if steps <= 0:
        raise ValueError("--steps must be positive")

    metadata = JointControllerMetadata(
        joint_names=("hinge",),
        default_joint_pos=np.zeros(1, dtype=np.float32),
        action_scale=np.array([0.45], dtype=np.float32),
        stiffness=np.array([24.0], dtype=np.float32),
        damping=np.array([1.2], dtype=np.float32),
        torque_limit=np.array([4.0], dtype=np.float32),
        control_dt=0.02,
        simulation_dt=0.002,
    )

    model = mujoco.MjModel.from_xml_string(_ONE_DOF_XML)
    data = mujoco.MjData(model)
    normalized_action = np.array([action], dtype=np.float32)
    target = normalized_action_to_pd_target(normalized_action, metadata)

    qpos_trace: list[float] = []
    torque_trace: list[float] = []
    substeps = max(1, round(metadata.control_dt / metadata.simulation_dt))
    for step_idx in range(steps):
        if step_idx == steps // 2:
            normalized_action *= -1.0
            target = normalized_action_to_pd_target(normalized_action, metadata)
        for _ in range(substeps):
            torque = pd_torque(target, data.qpos.copy(), data.qvel.copy(), metadata)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        qpos_trace.append(float(data.qpos[0]))
        torque_trace.append(float(data.ctrl[0]))

    qpos = np.asarray(qpos_trace, dtype=np.float32)
    torque = np.asarray(torque_trace, dtype=np.float32)
    finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(torque)))
    moved = bool(np.max(np.abs(qpos)) > 1e-3)
    within_ctrlrange = bool(np.max(np.abs(torque)) <= float(metadata.torque_limit[0]) + 1e-6)
    passed = finite and moved and within_ctrlrange

    summary = {
        "status": "passed" if passed else "failed",
        "validation": "mujoco_normalized_action_pd_contract_smoke",
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "steps": int(steps),
        "control_dt": metadata.control_dt,
        "simulation_dt": metadata.simulation_dt,
        "substeps_per_control_step": int(substeps),
        "metadata": {
            **asdict(metadata),
            "joint_names": list(metadata.joint_names),
            "default_joint_pos": metadata.default_joint_pos.tolist(),
            "action_scale": metadata.action_scale.tolist(),
            "stiffness": metadata.stiffness.tolist(),
            "damping": metadata.damping.tolist(),
            "torque_limit": metadata.torque_limit.tolist(),
        },
        "initial_action": float(action),
        "target_before_switch": float(metadata.default_joint_pos[0] + metadata.action_scale[0] * action),
        "target_after_switch": float(metadata.default_joint_pos[0] - metadata.action_scale[0] * action),
        "final_qpos": float(qpos[-1]),
        "max_abs_qpos": float(np.max(np.abs(qpos))),
        "max_abs_torque": float(np.max(np.abs(torque))),
        "finite": finite,
        "moved": moved,
        "within_ctrlrange": within_ctrlrange,
        "scope_note": "This gate validates MuJoCo installation plus shared action/PD semantics; full G1 sim-to-sim still requires a robot XML/URDF backend mapping.",
    }
    _write_json(Path(output), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate MuJoCo installation and shared controller contract.")
    parser.add_argument("--output", required=True, help="JSON summary path.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action", type=float, default=0.7)
    args = parser.parse_args(argv)
    summary = run_mujoco_sim_to_sim_smoke(output=args.output, steps=args.steps, action=args.action)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
