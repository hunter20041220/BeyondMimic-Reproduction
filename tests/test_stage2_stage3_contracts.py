from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from beyondmimic_repro.adapters.mujoco.shared_controller_contract import JointControllerMetadata, normalized_action_to_pd_target
from beyondmimic_repro.contracts.dagger_dataset import DAggerDatasetMetadata, load_dagger_dataset, merge_dagger_rounds, save_dagger_dataset
from beyondmimic_repro.contracts.teacher_assets import load_teacher_map
from beyondmimic_repro.contracts.vae_rollout import VAERolloutMetadata, save_vae_rollout
from beyondmimic_repro.stage3.datasets.state_latent_builder import build_from_vae_rollout
from beyondmimic_repro.stage3.representation.character_frame import world_to_character_yaw
from beyondmimic_repro.stage3.representation.emphasis_projection import apply_inverse_projection, apply_projection, build_projection_matrix


torch = pytest.importorskip("torch")

from beyondmimic_repro.stage2.rollout.ou_noise import OrnsteinUhlenbeckNoise


def _dagger_payload(n: int = 3) -> dict[str, np.ndarray]:
    return {
        "encoder_reference_input": np.zeros((n, 67), dtype=np.float32),
        "decoder_proprio_input": np.zeros((n, 96), dtype=np.float32),
        "student_mu": np.zeros((n, 32), dtype=np.float32),
        "student_logvar": np.zeros((n, 32), dtype=np.float32),
        "student_latent": np.zeros((n, 32), dtype=np.float32),
        "student_action": np.zeros((n, 29), dtype=np.float32),
        "teacher_action": np.zeros((n, 29), dtype=np.float32),
        "policy_observation": np.zeros((n, 160), dtype=np.float32),
        "root_state": np.zeros((n, 13), dtype=np.float32),
        "joint_position": np.zeros((n, 29), dtype=np.float32),
        "joint_velocity": np.zeros((n, 29), dtype=np.float32),
        "previous_action": np.zeros((n, 29), dtype=np.float32),
        "reward": np.zeros((n,), dtype=np.float32),
        "done": np.zeros((n,), dtype=np.bool_),
        "body_position_error": np.zeros((n, 14, 3), dtype=np.float32),
        "body_orientation_error": np.zeros((n, 14, 3), dtype=np.float32),
        "joint_position_error": np.zeros((n, 29), dtype=np.float32),
        "joint_velocity_error": np.zeros((n, 29), dtype=np.float32),
    }


def test_teacher_map_parsing_with_relocation(tmp_path) -> None:
    teacher_map = tmp_path / "teacher_map.json"
    teacher_map.write_text(
        json.dumps(
            {
                "teachers": [
                    {
                        "motion_name": "walk1",
                        "checkpoint_path": "/old/root/model.pt",
                        "motion_file": "/old/root/motion.npz",
                        "teacher_rollout_path": "/old/root/rollout.npz",
                        "frequency_hz": 50,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assets = load_teacher_map(teacher_map, data_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")
    assert assets["walk1"].checkpoint_path == tmp_path / "ckpt" / "model.pt"
    assert assets["walk1"].motion_file == tmp_path / "data" / "motion.npz"


def test_dagger_dataset_round_trip_and_merge(tmp_path) -> None:
    a = tmp_path / "d0.npz"
    b = tmp_path / "d1.npz"
    save_dagger_dataset(a, _dagger_payload(2), DAggerDatasetMetadata(round_ids=("D0",)))
    save_dagger_dataset(b, _dagger_payload(3), DAggerDatasetMetadata(round_ids=("D1",)))
    payload, metadata = load_dagger_dataset(a)
    assert payload["teacher_action"].shape == (2, 29)
    assert metadata.schema_version == "stage2-dagger-v1"
    merged = tmp_path / "merged.npz"
    summary = merge_dagger_rounds([a, b], merged)
    assert summary["sample_count"] == 5


def test_ou_noise_reproducibility_and_partial_reset() -> None:
    n1 = OrnsteinUhlenbeckNoise(seed=7)
    n2 = OrnsteinUhlenbeckNoise(seed=7)
    n1.reset(2, 29)
    n2.reset(2, 29)
    assert torch.allclose(n1.step(), n2.step())
    before = n1.step()
    n1.step(reset_mask=torch.tensor([True, False]))
    assert not torch.allclose(before[0], n1.state[0])
    assert n1.state.shape == (2, 29)


def test_character_frame_translation_and_yaw_invariance() -> None:
    points = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    local = world_to_character_yaw(points, np.zeros(3), 0.0)
    yaw = np.pi / 2.0
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = points @ rot.T + np.array([3.0, -2.0, 0.5])
    shifted_local = world_to_character_yaw(shifted, np.array([3.0, -2.0, 0.5]), yaw)
    assert np.allclose(local, shifted_local, atol=1e-6)


def test_projection_inverse_round_trip() -> None:
    payload = build_projection_matrix(seed=11)
    state_dim = payload["P"].shape[1]
    states = np.ones((2, state_dim), dtype=np.float64)
    recovered = apply_inverse_projection(apply_projection(states, payload["P"]), payload["P_inv"])
    assert np.allclose(states, recovered, atol=1e-6)


def test_state_latent_from_vae_rollout_shape(tmp_path) -> None:
    rollout = tmp_path / "vae_rollout.npz"
    save_vae_rollout(
        rollout,
        {
            "actual_state": np.zeros((1, 24, 99), dtype=np.float32),
            "latent": np.zeros((1, 24, 32), dtype=np.float32),
            "clean_action": np.zeros((1, 24, 29), dtype=np.float32),
            "executed_action": np.zeros((1, 24, 29), dtype=np.float32),
            "accepted": np.ones((1, 24), dtype=np.bool_),
            "episode_id": np.zeros((1, 24), dtype=np.int32),
            "time_index": np.arange(24, dtype=np.int32)[None, :],
        },
        VAERolloutMetadata(),
    )
    summary = build_from_vae_rollout(rollout, tmp_path / "state_latent.npz")
    assert summary["window_count"] == 4


def test_controller_normalized_action_contract() -> None:
    meta = JointControllerMetadata(
        joint_names=tuple(f"j{i}" for i in range(29)),
        default_joint_pos=np.ones(29, dtype=np.float32),
        action_scale=np.full(29, 0.5, dtype=np.float32),
        stiffness=np.ones(29, dtype=np.float32),
        damping=np.ones(29, dtype=np.float32),
        torque_limit=np.ones(29, dtype=np.float32),
        control_dt=0.02,
        simulation_dt=0.005,
    )
    target = normalized_action_to_pd_target(np.ones(29, dtype=np.float32), meta)
    assert np.allclose(target, 1.5)


def test_stage_configs_load() -> None:
    for path in [
        "configs/stage2/vae_paper.yaml",
        "configs/stage3/diffusion_paper_25hz.yaml",
        "configs/stage3/diffusion_engineering_50hz.yaml",
    ]:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["schema_version"]
