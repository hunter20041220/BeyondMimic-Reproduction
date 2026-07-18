#!/usr/bin/env python3
"""Build an unlabeled transition-aware locomotion state-latent dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.contracts.state_latent import StateLatentMetadata, save_state_latent_dataset


TARGET_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _swap_lr(name: str) -> str:
    if name.startswith("left_"):
        return name.replace("left_", "right_", 1)
    if name.startswith("right_"):
        return name.replace("right_", "left_", 1)
    return name


def _body_mirror_source() -> np.ndarray:
    by_name = {name: i for i, name in enumerate(TARGET_BODY_NAMES)}
    return np.asarray([by_name[_swap_lr(name)] for name in TARGET_BODY_NAMES], dtype=np.int64)


def _mirror_polar_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    return out


def _mirror_axial_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    out[..., 2] *= -1.0
    return out


def _mirror_rot6d_column_major(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out *= np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    return out


def _mirror_body_vectors(flat: np.ndarray, source: np.ndarray) -> np.ndarray:
    vec = np.asarray(flat, dtype=np.float32).reshape(*flat.shape[:-1], len(source), 3)
    mirrored = vec[..., source, :].copy()
    mirrored[..., 1] *= -1.0
    return mirrored.reshape(flat.shape).astype(np.float32, copy=False)


def _mirror_hybrid_state(hybrid: np.ndarray) -> np.ndarray:
    if hybrid.shape[-1] != 99:
        raise ValueError(f"hybrid paper state must have dim 99, got {hybrid.shape}")
    source = _body_mirror_source()
    out = np.asarray(hybrid, dtype=np.float32).copy()
    out[..., 0:3] = _mirror_polar_vec3(out[..., 0:3])
    out[..., 3:9] = _mirror_rot6d_column_major(out[..., 3:9])
    out[..., 9:12] = _mirror_polar_vec3(out[..., 9:12])
    out[..., 12:15] = _mirror_axial_vec3(out[..., 12:15])
    out[..., 15:57] = _mirror_body_vectors(out[..., 15:57], source)
    out[..., 57:99] = _mirror_body_vectors(out[..., 57:99], source)
    return out


def _make_lookup(features: dict[str, np.ndarray]) -> dict[tuple[int, int, int], int]:
    return {
        (int(motion), int(env), int(time)): i
        for i, (motion, env, time) in enumerate(
            zip(features["motion_id"], features["source_environment_id"], features["time_index"], strict=True)
        )
    }


def _phase_bins(time_index: np.ndarray, *, bin_count: int = 10) -> np.ndarray:
    max_time = max(1, int(np.max(time_index)))
    return np.minimum((np.asarray(time_index, dtype=np.int64) * bin_count) // (max_time + 1), bin_count - 1)


def _compute_sample_weight(payload: dict[str, np.ndarray], features: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    motion_id = np.asarray(payload["motion_id"], dtype=np.int64)
    env_id = np.asarray(payload["source_environment_id"], dtype=np.int64)
    time_index = np.asarray(payload["time_index"], dtype=np.int64)
    lookup = _make_lookup(features)
    feature_index = np.asarray([lookup[(int(m), int(e), int(t))] for m, e, t in zip(motion_id, env_id, time_index, strict=True)])
    weight = np.ones(motion_id.shape[0], dtype=np.float32)

    motion_counts = dict(zip(*np.unique(motion_id, return_counts=True)))
    motion_target = float(np.mean(list(motion_counts.values())))
    for motion, count in motion_counts.items():
        weight[motion_id == motion] *= motion_target / float(count)

    bins = _phase_bins(time_index, bin_count=10)
    phase_counts: dict[str, int] = {}
    for motion in np.unique(motion_id):
        mask_motion = motion_id == motion
        counts = np.bincount(bins[mask_motion], minlength=10).astype(np.float64)
        target = float(np.mean(counts[counts > 0])) if np.any(counts > 0) else 1.0
        for phase in range(10):
            mask = mask_motion & (bins == phase)
            if counts[phase] > 0:
                weight[mask] *= target / counts[phase]
            phase_counts[f"{int(motion)}:{phase}"] = int(counts[phase])

    intermediate = np.asarray(features["intermediate_speed"], dtype=np.bool_)[feature_index]
    walk_to_jog = np.asarray(features["walk_to_jog"], dtype=np.bool_)[feature_index]
    jog_to_run = np.asarray(features["jog_to_run"], dtype=np.bool_)[feature_index]
    run_to_walk = np.asarray(features["run_to_walk"], dtype=np.bool_)[feature_index]
    contact_change = np.asarray(features["contact_change"], dtype=np.bool_)[feature_index]
    flight_fraction = np.asarray(features["flight_fraction"], dtype=np.float32)[feature_index]
    weight *= np.where(intermediate, 1.50, 1.0).astype(np.float32)
    bridge = walk_to_jog | jog_to_run | run_to_walk
    weight *= np.where(bridge, 4.00, 1.0).astype(np.float32)
    weight *= np.where(contact_change, 1.35, 1.0).astype(np.float32)
    weight *= np.where(flight_fraction > 0.0, 1.25, 1.0).astype(np.float32)
    weight = np.clip(weight, 0.25, 12.0)
    weight /= float(np.mean(weight))
    summary = {
        "motion_counts": {str(int(k)): int(v) for k, v in motion_counts.items()},
        "phase_bin_counts": phase_counts,
        "weight_min": float(weight.min()),
        "weight_mean": float(weight.mean()),
        "weight_max": float(weight.max()),
        "weight_p50": float(np.percentile(weight, 50)),
        "weight_p90": float(np.percentile(weight, 90)),
        "weight_p99": float(np.percentile(weight, 99)),
        "weighted_window_flags": {
            "intermediate_speed": int(intermediate.sum()),
            "bridge_any": int(bridge.sum()),
            "walk_to_jog": int(walk_to_jog.sum()),
            "jog_to_run": int(jog_to_run.sum()),
            "run_to_walk": int(run_to_walk.sum()),
            "contact_change": int(contact_change.sum()),
            "flight_fraction_positive": int((flight_fraction > 0.0).sum()),
        },
        "weight_formula": (
            "motion balance * 10-bin reference phase balance * "
            "1.5(intermediate) * 4.0(natural accel/decel bridge) * "
            "1.35(contact change) * 1.25(flight>0), clipped [0.25,12], normalized mean=1"
        ),
    }
    return weight.astype(np.float32), summary


def _load_metadata(raw: str) -> StateLatentMetadata:
    base = StateLatentMetadata()
    data = json.loads(raw) if raw else {}
    return StateLatentMetadata(**{**base.__dict__, **data})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--bridge-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    print(f"[transition-dataset] loading bridge features {args.bridge_features}", flush=True)
    with np.load(args.bridge_features, allow_pickle=False) as data:
        features = {key: data[key] for key in data.files}

    print(f"[transition-dataset] loading state-latent dataset {args.input_dataset}", flush=True)
    with np.load(args.input_dataset, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata_raw = str(data["metadata_json"]) if "metadata_json" in data.files else ""
    metadata = _load_metadata(metadata_raw)
    states = np.asarray(payload["states"], dtype=np.float32)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    tokens = np.asarray(payload["tokens"], dtype=np.float32)
    projection_matrix = np.asarray(payload["state_projection_matrix"], dtype=np.float32)
    projection_inverse = np.asarray(payload["state_projection_inverse"], dtype=np.float32)
    if states.shape[-1] != 163:
        raise ValueError(f"expected paper_projected 163D states, got {states.shape}")
    if latents.shape[-1] != 32:
        raise ValueError(f"expected 32D VAE latents, got {latents.shape}")
    weight, weight_summary = _compute_sample_weight(payload, features)

    print("[transition-dataset] unprojecting and mirroring paper states", flush=True)
    hybrid = np.matmul(states, projection_inverse.T).astype(np.float32)
    mirrored_hybrid = _mirror_hybrid_state(hybrid)
    mirrored_states = np.matmul(mirrored_hybrid, projection_matrix.T).astype(np.float32)
    mirrored_latents = latents.copy()
    mirrored_tokens = np.concatenate([mirrored_states, mirrored_latents], axis=-1).astype(np.float32)
    if not np.all(np.isfinite(mirrored_tokens)):
        raise ValueError("mirrored tokens contain NaN/Inf")
    roundtrip_error = float(np.max(np.abs(_mirror_hybrid_state(mirrored_hybrid) - hybrid)))

    print("[transition-dataset] materializing augmented payload", flush=True)
    out_payload: dict[str, np.ndarray] = {}
    n = int(tokens.shape[0])
    for key, value in payload.items():
        arr = np.asarray(value)
        if key in {"normalization_mean", "normalization_std"}:
            continue
        if key == "states":
            out_payload[key] = np.concatenate([states, mirrored_states], axis=0)
        elif key == "latents":
            out_payload[key] = np.concatenate([latents, mirrored_latents], axis=0)
        elif key == "tokens":
            out_payload[key] = np.concatenate([tokens, mirrored_tokens], axis=0)
        elif arr.ndim > 0 and arr.shape[0] == n:
            out_payload[key] = np.concatenate([arr, arr], axis=0)
        else:
            out_payload[key] = arr
    out_payload["sample_weight"] = np.concatenate([weight, weight], axis=0).astype(np.float32)
    out_payload["symmetry_augmented"] = np.concatenate([np.zeros(n, dtype=np.bool_), np.ones(n, dtype=np.bool_)], axis=0)
    out_payload["mirror_source_index"] = np.concatenate([np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32)], axis=0)
    out_tokens = np.asarray(out_payload["tokens"], dtype=np.float32)
    out_payload["normalization_mean"] = out_tokens.mean(axis=(0, 1)).astype(np.float32)
    std = out_tokens.std(axis=(0, 1)).astype(np.float32)
    out_payload["normalization_std"] = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    out_payload["state_dim"] = np.asarray(163, dtype=np.int32)
    out_payload["latent_dim"] = np.asarray(32, dtype=np.int32)
    out_payload["token_dim"] = np.asarray(195, dtype=np.int32)
    out_payload["state_representation"] = np.asarray("paper_projected")

    print(f"[transition-dataset] saving {out_tokens.shape[0]} windows to {args.output}", flush=True)
    saved = save_state_latent_dataset(args.output, out_payload, metadata, compressed=args.compressed)
    summary = {
        **saved,
        "input_dataset": str(args.input_dataset),
        "bridge_features": str(args.bridge_features),
        "original_window_count": n,
        "augmented_window_count": int(out_tokens.shape[0]),
        "symmetry_enabled": True,
        "sagittal_mirror": {
            "plane": "x-z sagittal plane, y sign flip",
            "root_position_velocity": "polar y-flip",
            "root_orientation_rot6d": "R' = diag(1,-1,1) R diag(1,-1,1), first-two-column Rot6D",
            "root_angular_velocity": "axial [-x,+y,-z]",
            "body_positions_velocities": "left/right body swap plus polar y-flip",
            "latents": "copied; VAE latent axes have no fixed sagittal physical basis",
        },
        "mirror_roundtrip_max_abs_error_99d": roundtrip_error,
        "sampling": weight_summary,
        "compressed": bool(args.compressed),
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
