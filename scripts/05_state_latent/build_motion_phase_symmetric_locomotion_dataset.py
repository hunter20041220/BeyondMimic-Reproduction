#!/usr/bin/env python3
"""Build motion/phase-balanced sagittal-symmetric locomotion state-latent data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - CPU numpy path remains available.
    torch = None


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


def _load_metadata(raw: str) -> StateLatentMetadata:
    base = StateLatentMetadata()
    data = json.loads(raw) if raw else {}
    return StateLatentMetadata(**{**base.__dict__, **data})


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


def _mirror_hybrid_state_torch(hybrid: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is required for CUDA mirror projection")
    if hybrid.shape[-1] != 99:
        raise ValueError(f"hybrid paper state must have dim 99, got {hybrid.shape}")
    device = hybrid.device
    source = torch.as_tensor(_body_mirror_source(), dtype=torch.long, device=device)
    out = hybrid.clone()
    out[..., 0:3] = out[..., 0:3] * torch.tensor([1.0, -1.0, 1.0], dtype=out.dtype, device=device)
    out[..., 3:9] = out[..., 3:9] * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=out.dtype, device=device)
    out[..., 9:12] = out[..., 9:12] * torch.tensor([1.0, -1.0, 1.0], dtype=out.dtype, device=device)
    out[..., 12:15] = out[..., 12:15] * torch.tensor([-1.0, 1.0, -1.0], dtype=out.dtype, device=device)
    for start in (15, 57):
        vec = out[..., start : start + 42].reshape(*out.shape[:-1], len(source), 3)
        vec = vec.index_select(-2, source).clone()
        vec[..., 1] *= -1.0
        out[..., start : start + 42] = vec.reshape(*out.shape[:-1], 42)
    return out


def _project_mirrored_states(
    states: np.ndarray,
    *,
    projection_matrix: np.ndarray,
    projection_inverse: np.ndarray,
    mirror_device: str,
    chunk_windows: int,
) -> tuple[np.ndarray, float]:
    if mirror_device.startswith("cuda") and torch is not None:
        device = torch.device(mirror_device)
        p = torch.as_tensor(projection_matrix, dtype=torch.float32, device=device)
        p_inv = torch.as_tensor(projection_inverse, dtype=torch.float32, device=device)
        mirrored_states = np.empty_like(states, dtype=np.float32)
        roundtrip_error = 0.0
        for start in range(0, states.shape[0], chunk_windows):
            stop = min(states.shape[0], start + chunk_windows)
            chunk = torch.as_tensor(states[start:stop], dtype=torch.float32, device=device)
            hybrid = torch.matmul(chunk, p_inv.T)
            mirrored_hybrid = _mirror_hybrid_state_torch(hybrid)
            projected = torch.matmul(mirrored_hybrid, p.T)
            mirrored_states[start:stop] = projected.detach().cpu().numpy()
            restored = _mirror_hybrid_state_torch(mirrored_hybrid)
            roundtrip_error = max(roundtrip_error, float(torch.max(torch.abs(restored - hybrid)).detach().cpu()))
            del chunk, hybrid, mirrored_hybrid, projected, restored
        return mirrored_states, roundtrip_error

    mirrored_states = np.empty_like(states, dtype=np.float32)
    roundtrip_error = 0.0
    for start in range(0, states.shape[0], chunk_windows):
        stop = min(states.shape[0], start + chunk_windows)
        hybrid = np.matmul(states[start:stop], projection_inverse.T).astype(np.float32)
        mirrored_hybrid = _mirror_hybrid_state(hybrid)
        mirrored_states[start:stop] = np.matmul(mirrored_hybrid, projection_matrix.T).astype(np.float32)
        roundtrip_error = max(roundtrip_error, float(np.max(np.abs(_mirror_hybrid_state(mirrored_hybrid) - hybrid))))
    return mirrored_states, roundtrip_error


def _motion_phase_weight(payload: dict[str, np.ndarray], *, phase_bins: int) -> tuple[np.ndarray, dict[str, Any]]:
    motion_id = np.asarray(payload["motion_id"], dtype=np.int64)
    phase_source_key = "reference_frame_index" if "reference_frame_index" in payload else "time_index"
    phase_source = np.asarray(payload[phase_source_key], dtype=np.int64)
    weight = np.ones(motion_id.shape[0], dtype=np.float32)
    unique_motion, motion_counts_arr = np.unique(motion_id, return_counts=True)
    motion_counts = {int(m): int(c) for m, c in zip(unique_motion, motion_counts_arr)}
    motion_target = float(np.mean(motion_counts_arr)) if motion_counts_arr.size else 1.0
    for motion, count in motion_counts.items():
        weight[motion_id == motion] *= motion_target / float(max(1, count))

    bins = np.zeros(motion_id.shape[0], dtype=np.int64)
    phase_ranges: dict[str, int] = {}
    for motion in unique_motion:
        mask_motion = motion_id == motion
        local_phase = phase_source[mask_motion]
        max_phase = max(1, int(np.max(local_phase)) if local_phase.size else 1)
        bins[mask_motion] = np.minimum((local_phase * phase_bins) // (max_phase + 1), phase_bins - 1)
        phase_ranges[str(int(motion))] = int(max_phase + 1)
    phase_counts: dict[str, int] = {}
    for motion in unique_motion:
        mask_motion = motion_id == motion
        counts = np.bincount(bins[mask_motion], minlength=phase_bins).astype(np.float64)
        nonzero = counts[counts > 0]
        target = float(np.mean(nonzero)) if nonzero.size else 1.0
        for phase in range(phase_bins):
            mask = mask_motion & (bins == phase)
            if counts[phase] > 0:
                weight[mask] *= target / counts[phase]
            phase_counts[f"{int(motion)}:{phase}"] = int(counts[phase])
    weight = np.clip(weight, 0.25, 8.0)
    weight /= float(np.mean(weight))
    return weight.astype(np.float32), {
        "motion_counts": {str(k): v for k, v in motion_counts.items()},
        "phase_bin_counts": phase_counts,
        "phase_bins": int(phase_bins),
        "phase_source": phase_source_key,
        "phase_source_range_by_motion": phase_ranges,
        "weight_min": float(weight.min()) if weight.size else None,
        "weight_mean": float(weight.mean()) if weight.size else None,
        "weight_max": float(weight.max()) if weight.size else None,
        "weight_formula": "motion inverse-frequency * per-motion phase-bin inverse-frequency, clipped [0.25,8], mean-normalized",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phase-bins", type=int, default=10)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--mirror-device", default="cpu", help="Use e.g. cuda:0 for projected-state mirror matmuls.")
    parser.add_argument("--chunk-windows", type=int, default=8192)
    args = parser.parse_args()

    with np.load(args.input_dataset, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata_raw = str(data["metadata_json"]) if "metadata_json" in data.files else ""
    metadata = _load_metadata(metadata_raw)
    states = np.asarray(payload["states"], dtype=np.float32)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    tokens = np.asarray(payload["tokens"], dtype=np.float32)
    projection_matrix = np.asarray(payload["state_projection_matrix"], dtype=np.float32)
    projection_inverse = np.asarray(payload["state_projection_inverse"], dtype=np.float32)
    if states.shape[-1] != 163 or latents.shape[-1] != 32:
        raise ValueError(f"expected states[...,163] and latents[...,32], got {states.shape}, {latents.shape}")
    if tokens.shape[-1] != 195:
        raise ValueError(f"expected token dim 195, got {tokens.shape[-1]}")

    sample_weight, sampling_summary = _motion_phase_weight(payload, phase_bins=args.phase_bins)
    mirrored_states, roundtrip_error = _project_mirrored_states(
        states,
        projection_matrix=projection_matrix,
        projection_inverse=projection_inverse,
        mirror_device=args.mirror_device,
        chunk_windows=max(1, int(args.chunk_windows)),
    )
    mirrored_latents = latents.copy()
    mirrored_tokens = np.concatenate([mirrored_states, mirrored_latents], axis=-1).astype(np.float32)

    n = int(tokens.shape[0])
    out_payload: dict[str, np.ndarray] = {}
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
    out_payload["sample_weight"] = np.concatenate([sample_weight, sample_weight], axis=0).astype(np.float32)
    out_payload["symmetry_augmented"] = np.concatenate([np.zeros(n, dtype=np.bool_), np.ones(n, dtype=np.bool_)], axis=0)
    out_payload["mirror_source_index"] = np.concatenate([np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32)], axis=0)
    out_payload["symmetry_manifest_json"] = np.asarray(
        json.dumps(
            {
                "plane": "sagittal x-z plane; polar vector y components are negated",
                "body_mapping": "left/right target body names are swapped before y reflection",
                "root_orientation": "rot6d mirrored as R' = diag(1,-1,1) R diag(1,-1,1)",
                "root_angular_velocity": "axial vector mirrored as [-x,+y,-z]",
                "latents": "copied because the VAE latent basis has no fixed hand-authored physical axis",
                "labels": "no speed, gait, command, or transition labels are introduced",
            },
            sort_keys=True,
        )
    )
    out_tokens = np.asarray(out_payload["tokens"], dtype=np.float32)
    out_payload["normalization_mean"] = out_tokens.mean(axis=(0, 1)).astype(np.float32)
    std = out_tokens.std(axis=(0, 1)).astype(np.float32)
    out_payload["normalization_std"] = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    out_payload["state_dim"] = np.asarray(163, dtype=np.int32)
    out_payload["latent_dim"] = np.asarray(32, dtype=np.int32)
    out_payload["token_dim"] = np.asarray(195, dtype=np.int32)
    out_payload["state_representation"] = np.asarray("paper_projected")

    saved = save_state_latent_dataset(args.output, out_payload, metadata, compressed=args.compressed)
    summary = {
        **saved,
        "input_dataset": str(args.input_dataset),
        "original_window_count": n,
        "augmented_window_count": int(out_tokens.shape[0]),
        "symmetry_enabled": True,
        "mirror_roundtrip_max_abs_error_99d": roundtrip_error,
        "sampling": sampling_summary,
        "compressed": bool(args.compressed),
        "mirror_device": args.mirror_device,
        "chunk_windows": int(args.chunk_windows),
        "symmetry_manifest": json.loads(str(out_payload["symmetry_manifest_json"])),
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
