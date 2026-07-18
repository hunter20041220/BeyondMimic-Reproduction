"""MuJoCo G1 asset and PD-controller smoke gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_G1_SCENE_XML = Path(
    "/shared_disk/zzy/BeyondMimic/download/reference_code/unitree_rl_mjlab/"
    "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
)
DEFAULT_CONTROLLER_YAML = Path(
    "/shared_disk/zzy/BeyondMimic/reproduction/third_party/official/"
    "motion_tracking_controller/config/g1/controllers.yaml"
)
DEFAULT_TEACHER_MAP = Path("configs/local/teacher_map_13motion_50hz.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_controller_metadata(path: Path) -> dict[str, dict[str, float]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data["standby_controller"]["ros__parameters"]
    joint_names = [str(name) for name in params["joint_names"]]
    return {
        "default_position": {name: float(value) for name, value in zip(joint_names, params["default_position"], strict=True)},
        "kp": {name: float(value) for name, value in zip(joint_names, params["kp"], strict=True)},
        "kd": {name: float(value) for name, value in zip(joint_names, params["kd"], strict=True)},
    }


def _load_policy_joint_order(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    teachers = data.get("teachers", [])
    if not teachers:
        return []
    return [str(name) for name in teachers[0].get("joint_names", [])]


def run_mujoco_g1_pd_smoke(
    *,
    output: str | Path,
    scene_xml: str | Path = DEFAULT_G1_SCENE_XML,
    controller_yaml: str | Path = DEFAULT_CONTROLLER_YAML,
    teacher_map: str | Path = DEFAULT_TEACHER_MAP,
    steps: int = 500,
    root_height: float = 0.79,
) -> dict[str, Any]:
    """Load G1 MJCF and hold the official standby pose with PD torques."""

    import mujoco

    if steps <= 0:
        raise ValueError("--steps must be positive")

    scene_path = Path(scene_xml)
    controller_path = Path(controller_yaml)
    teacher_map_path = Path(teacher_map)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not controller_path.is_file():
        raise FileNotFoundError(controller_path)

    controller = _load_controller_metadata(controller_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    actuator_joint_names: list[str] = []
    qpos_addr: list[int] = []
    qvel_addr: list[int] = []
    ctrlrange: list[np.ndarray] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_joint_names.append(str(joint_name))
        qpos_addr.append(int(model.jnt_qposadr[joint_id]))
        qvel_addr.append(int(model.jnt_dofadr[joint_id]))
        ctrlrange.append(model.actuator_ctrlrange[actuator_id].copy())

    missing_controller_joints = [
        name for name in actuator_joint_names if name not in controller["default_position"]
    ]
    if missing_controller_joints:
        raise ValueError(f"controller YAML is missing actuator joints: {missing_controller_joints}")

    target = np.asarray([controller["default_position"][name] for name in actuator_joint_names], dtype=np.float64)
    kp = np.asarray([controller["kp"][name] for name in actuator_joint_names], dtype=np.float64)
    kd = np.asarray([controller["kd"][name] for name in actuator_joint_names], dtype=np.float64)
    limits = np.asarray(ctrlrange, dtype=np.float64)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.asarray([0.0, 0.0, root_height], dtype=np.float64)
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for value, address in zip(target, qpos_addr, strict=True):
        data.qpos[address] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    heights: list[float] = []
    torque_max = 0.0
    finite = True
    for _ in range(steps):
        q = np.asarray([data.qpos[address] for address in qpos_addr], dtype=np.float64)
        qd = np.asarray([data.qvel[address] for address in qvel_addr], dtype=np.float64)
        torque = kp * (target - q) - kd * qd
        torque = np.clip(torque, limits[:, 0], limits[:, 1])
        data.ctrl[:] = torque
        torque_max = max(torque_max, float(np.max(np.abs(torque))))
        mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
        finite = finite and bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))

    policy_joint_order = _load_policy_joint_order(teacher_map_path)
    policy_joint_set_matches = bool(policy_joint_order) and set(policy_joint_order) == set(actuator_joint_names)
    policy_to_mujoco_permutation = (
        [policy_joint_order.index(name) for name in actuator_joint_names] if policy_joint_set_matches else []
    )
    within_ctrlrange = bool(torque_max <= float(np.max(np.abs(limits))) + 1e-6)
    min_height = float(min(heights)) if heights else float("nan")
    passed = (
        model.nq == 36
        and model.nv == 35
        and model.nu == 29
        and finite
        and min_height > 0.5
        and within_ctrlrange
        and not missing_controller_joints
    )

    summary = {
        "status": "passed" if passed else "failed",
        "validation": "mujoco_g1_29dof_asset_pd_hold_smoke",
        "scene_xml": str(scene_path),
        "controller_yaml": str(controller_path),
        "teacher_map": str(teacher_map_path),
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "steps": int(steps),
        "simulation_dt": float(model.opt.timestep),
        "simulated_time_s": float(data.time),
        "actuator_joint_names": actuator_joint_names,
        "policy_joint_set_matches_mujoco": policy_joint_set_matches,
        "policy_to_mujoco_permutation": policy_to_mujoco_permutation,
        "root_height_initial": float(root_height),
        "root_height_final": float(heights[-1]) if heights else float("nan"),
        "root_height_min": min_height,
        "max_abs_torque": torque_max,
        "within_ctrlrange": within_ctrlrange,
        "contact_count_final": int(data.ncon),
        "finite": finite,
        "scope_note": (
            "This validates the real G1 MJCF asset, 29-DOF actuator/joint mapping, "
            "and PD torque backend. Full diffusion sim-to-sim still requires wiring "
            "the VAE/diffusion policy observation loop to this MuJoCo backend."
        ),
    }
    _write_json(Path(output), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate MuJoCo G1 MJCF asset and PD hold.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-xml", default=str(DEFAULT_G1_SCENE_XML))
    parser.add_argument("--controller-yaml", default=str(DEFAULT_CONTROLLER_YAML))
    parser.add_argument("--teacher-map", default=str(DEFAULT_TEACHER_MAP))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--root-height", type=float, default=0.79)
    args = parser.parse_args(argv)
    summary = run_mujoco_g1_pd_smoke(
        output=args.output,
        scene_xml=args.scene_xml,
        controller_yaml=args.controller_yaml,
        teacher_map=args.teacher_map,
        steps=args.steps,
        root_height=args.root_height,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
