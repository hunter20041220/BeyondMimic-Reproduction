from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from beyondmimic_repro.contracts.state_latent import StateLatentMetadata, save_state_latent_dataset
from beyondmimic_repro.stage2.training_runtime import build_vae_arrays_from_teacher_rollout, train_vae_bc_warmstart_runtime
from beyondmimic_repro.stage3.diffusion.training_runtime import train_state_latent_diffusion_runtime


def test_structured_teacher_rollout_builds_vae_arrays_and_trains(tmp_path) -> None:
    rollout = tmp_path / "teacher_rollout.npz"
    shape = (2, 3)
    body_names = np.array(["pelvis"], dtype="<U16")
    np.savez_compressed(
        rollout,
        teacher_action=np.zeros((*shape, 29), dtype=np.float32),
        ref_joint_pos=np.zeros((*shape, 29), dtype=np.float32),
        ref_joint_vel=np.zeros((*shape, 29), dtype=np.float32),
        joint_pos=np.zeros((*shape, 29), dtype=np.float32),
        joint_vel=np.zeros((*shape, 29), dtype=np.float32),
        root_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (*shape, 1)),
        root_lin_vel_w=np.zeros((*shape, 3), dtype=np.float32),
        root_ang_vel_w=np.zeros((*shape, 3), dtype=np.float32),
        body_pos_w=np.zeros((*shape, 1, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (*shape, 1, 1)),
        ref_anchor_pos_w=np.zeros((*shape, 3), dtype=np.float32),
        ref_anchor_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (*shape, 1)),
        body_names=body_names,
    )
    arrays, metadata = build_vae_arrays_from_teacher_rollout(rollout)
    assert arrays["encoder_reference_input"].shape == (6, 67)
    assert arrays["decoder_proprio_input"].shape == (6, 96)
    assert metadata["source"] == "structured_rollout_tensors"

    config = tmp_path / "vae.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "encoder_input_dim": 67,
                    "decoder_proprio_dim": 96,
                    "latent_dim": 4,
                    "action_dim": 29,
                    "encoder_hidden_dims": [8],
                    "decoder_hidden_dims": [8],
                    "activation": "ELU",
                },
                "training": {
                    "learning_rate": 0.001,
                    "kl_coefficient": 0.01,
                    "gradient_accumulation_steps": 1,
                    "batch_size": 2,
                    "epochs": 1,
                    "mixed_precision": "fp32",
                },
            }
        ),
        encoding="utf-8",
    )
    summary = train_vae_bc_warmstart_runtime(
        teacher_rollout=rollout,
        config_path=config,
        output_dir=tmp_path / "vae_out",
        device="cpu",
        seed=1,
    )
    assert summary["status"] == "trained"
    assert (tmp_path / "vae_out" / "checkpoints" / "latest.pt").is_file()


def test_state_latent_diffusion_runtime_trains_tiny_model(tmp_path) -> None:
    dataset = tmp_path / "state_latent.npz"
    states = np.zeros((4, 21, 5), dtype=np.float32)
    latents = np.zeros((4, 21, 32), dtype=np.float32)
    tokens = np.concatenate([states, latents], axis=-1)
    save_state_latent_dataset(
        dataset,
        {
            "states": states,
            "latents": latents,
            "tokens": tokens,
            "valid_mask": np.ones((4, 21), dtype=np.bool_),
            "episode_id": np.arange(4, dtype=np.int32),
            "motion_id": np.zeros(4, dtype=np.int32),
            "time_index": np.arange(4, dtype=np.int32),
            "frequency_hz": np.full(4, 50.0, dtype=np.float32),
        },
        StateLatentMetadata(),
    )
    config = tmp_path / "diffusion.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sequence_length": 21,
                "prediction_type": "x0",
                "model": {"embedding_dim": 16, "attention_heads": 4, "transformer_layers": 1},
                "diffusion": {"denoising_steps": 4},
                "training": {
                    "per_device_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "mixed_precision": "fp32",
                    "warmup_steps": 1,
                    "epochs": 1,
                },
                "ema": {"power": 0.75, "max": 0.999},
            }
        ),
        encoding="utf-8",
    )
    summary = train_state_latent_diffusion_runtime(
        dataset_path=dataset,
        config_path=config,
        output_dir=tmp_path / "diff_out",
        prediction_type="x0",
        device="cpu",
        seed=1,
    )
    assert summary["status"] == "trained"
    assert (tmp_path / "diff_out" / "checkpoints" / "latest.pt").is_file()
    assert json.loads((tmp_path / "diff_out" / "summary.json").read_text(encoding="utf-8"))["status"] == "trained"
