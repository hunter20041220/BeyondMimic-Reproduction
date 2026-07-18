#!/usr/bin/env python3
"""Build all-motion motion/phase-balanced sagittal-symmetric state-latent data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.contracts.state_latent import save_state_latent_dataset
from build_motion_phase_symmetric_locomotion_dataset import (  # noqa: E402
    _load_metadata,
    _motion_phase_weight,
    _project_mirrored_states,
    _write_json,
)


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
                "scope": "all available 50Hz motions, balanced by motion_id and reference phase proxy",
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
