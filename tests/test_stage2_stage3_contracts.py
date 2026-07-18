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


def test_state_latent_from_vae_rollout_filters_done_windows(tmp_path) -> None:
    rollout = tmp_path / "vae_rollout_done.npz"
    done = np.zeros((2, 24), dtype=np.bool_)
    done[0, 3] = True
    accepted = np.ones((2, 24), dtype=np.bool_)
    accepted[0, 4:] = False
    save_vae_rollout(
        rollout,
        {
            "actual_state": np.zeros((2, 24, 99), dtype=np.float32),
            "latent": np.zeros((2, 24, 32), dtype=np.float32),
            "clean_action": np.zeros((2, 24, 29), dtype=np.float32),
            "executed_action": np.zeros((2, 24, 29), dtype=np.float32),
            "accepted": accepted,
            "done": done,
            "episode_id": np.broadcast_to(np.arange(2, dtype=np.int32)[:, None], (2, 24)),
            "time_index": np.broadcast_to(np.arange(24, dtype=np.int32)[None, :], (2, 24)),
        },
        VAERolloutMetadata(),
    )
    output = tmp_path / "state_latent_done.npz"
    summary = build_from_vae_rollout(rollout, output, motion_id=7)
    assert summary["window_count"] == 4
    assert summary["discarded_window_count"] == 4
    with np.load(output, allow_pickle=False) as data:
        assert np.all(data["source_environment_id"] == 1)
        assert np.all(data["motion_id"] == 7)
        assert np.array_equal(data["time_index"], np.arange(4, dtype=np.int32))


def test_state_latent_from_vae_rollout_can_reject_failed_episode_prefixes(tmp_path) -> None:
    rollout = tmp_path / "vae_rollout_prefix_fail.npz"
    accepted = np.ones((2, 24), dtype=np.bool_)
    accepted[0, 22:] = False
    save_vae_rollout(
        rollout,
        {
            "actual_state": np.zeros((2, 24, 99), dtype=np.float32),
            "latent": np.zeros((2, 24, 32), dtype=np.float32),
            "clean_action": np.zeros((2, 24, 29), dtype=np.float32),
            "executed_action": np.zeros((2, 24, 29), dtype=np.float32),
            "accepted": accepted,
            "episode_id": np.broadcast_to(np.arange(2, dtype=np.int32)[:, None], (2, 24)),
            "time_index": np.broadcast_to(np.arange(24, dtype=np.int32)[None, :], (2, 24)),
        },
        VAERolloutMetadata(frequency_hz=50.0),
    )

    loose_summary = build_from_vae_rollout(rollout, tmp_path / "state_latent_prefix_loose.npz")
    strict_output = tmp_path / "state_latent_prefix_strict.npz"
    strict_summary = build_from_vae_rollout(
        rollout,
        strict_output,
        require_full_episode_accepted=True,
    )

    assert loose_summary["window_count"] == 6
    assert strict_summary["window_count"] == 4
    assert strict_summary["accepted_episode_count"] == 1
    assert strict_summary["rejected_episode_count"] == 1
    with np.load(strict_output, allow_pickle=False) as data:
        assert np.all(data["source_environment_id"] == 1)


def test_state_latent_from_vae_rollout_can_downsample_to_25hz(tmp_path) -> None:
    rollout = tmp_path / "vae_rollout_50hz.npz"
    save_vae_rollout(
        rollout,
        {
            "actual_state": np.zeros((1, 50, 99), dtype=np.float32),
            "latent": np.zeros((1, 50, 32), dtype=np.float32),
            "clean_action": np.zeros((1, 50, 29), dtype=np.float32),
            "executed_action": np.zeros((1, 50, 29), dtype=np.float32),
            "accepted": np.ones((1, 50), dtype=np.bool_),
            "episode_id": np.zeros((1, 50), dtype=np.int32),
            "time_index": np.arange(50, dtype=np.int32)[None, :],
        },
        VAERolloutMetadata(frequency_hz=50.0),
    )
    output = tmp_path / "state_latent_25hz.npz"
    summary = build_from_vae_rollout(rollout, output, target_frequency_hz=25.0)
    assert summary["source_stride"] == 2
    assert summary["target_frequency_hz"] == 25.0
    assert summary["window_count"] == 5
    with np.load(output, allow_pickle=False) as data:
        assert np.all(data["frequency_hz"] == 25.0)
        assert np.array_equal(data["time_index"], np.array([0, 2, 4, 6, 8], dtype=np.int32))


def test_state_latent_from_vae_rollout_builds_paper_projected_state(tmp_path) -> None:
    rollout = tmp_path / "vae_rollout_paper_raw.npz"
    steps = 24
    root_pos = np.zeros((1, steps, 3), dtype=np.float32)
    root_pos[0, :, 0] = np.linspace(0.0, 0.23, steps, dtype=np.float32)
    root_quat = np.zeros((1, steps, 4), dtype=np.float32)
    root_quat[..., 0] = 1.0
    root_vel = np.zeros((1, steps, 3), dtype=np.float32)
    root_vel[..., 0] = 0.5
    body_pos = np.zeros((1, steps, 14, 3), dtype=np.float32)
    body_pos[..., 0] = root_pos[:, :, None, 0]
    body_vel = np.zeros((1, steps, 14, 3), dtype=np.float32)
    save_vae_rollout(
        rollout,
        {
            "actual_state": np.zeros((1, steps, 79), dtype=np.float32),
            "root_pos_w": root_pos,
            "root_quat_w": root_quat,
            "root_lin_vel_w": root_vel,
            "root_ang_vel_w": np.zeros_like(root_vel),
            "body_pos_w": body_pos,
            "body_lin_vel_w": body_vel,
            "latent": np.zeros((1, steps, 32), dtype=np.float32),
            "clean_action": np.zeros((1, steps, 29), dtype=np.float32),
            "executed_action": np.zeros((1, steps, 29), dtype=np.float32),
            "accepted": np.ones((1, steps), dtype=np.bool_),
            "episode_id": np.zeros((1, steps), dtype=np.int32),
            "time_index": np.arange(steps, dtype=np.int32)[None, :],
        },
        VAERolloutMetadata(),
    )
    hybrid_summary = build_from_vae_rollout(rollout, tmp_path / "state_latent_hybrid.npz", state_representation="paper_hybrid")
    projected_summary = build_from_vae_rollout(
        rollout,
        tmp_path / "state_latent_projected.npz",
        state_representation="paper_projected",
    )
    assert hybrid_summary["state_dim"] == 99
    assert projected_summary["state_dim"] == 163
    with np.load(tmp_path / "state_latent_projected.npz", allow_pickle=False) as data:
        assert data["states"].shape[-1] == 163
        assert data["state_projection_matrix"].shape == (163, 99)
        assert data["state_projection_inverse"].shape == (99, 163)


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
