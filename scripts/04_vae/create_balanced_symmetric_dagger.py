#!/usr/bin/env python3
"""Build a motion-balanced DAgger dataset with sagittal mirror augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.contracts.dagger_dataset import DAggerDatasetMetadata, load_dagger_dataset, save_dagger_dataset


POLICY_OBS_DIM = 160
REF_DIM = 29
ACTION_DIM = 29
LATENT_DIM = 32


def _load_reference_names(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    teachers = data["teachers"] if isinstance(data, dict) and "teachers" in data else data
    if not teachers:
        raise ValueError(f"teacher map has no teachers: {path}")
    first = teachers[0]
    return [str(v) for v in first["joint_names"]], [str(v) for v in first.get("body_names", [])]


def _swap_lr(name: str) -> str:
    if "left_" in name:
        return name.replace("left_", "right_", 1)
    if "right_" in name:
        return name.replace("right_", "left_", 1)
    return name


def _joint_sign(name: str) -> float:
    if "_roll_" in name or name.endswith("_roll_joint"):
        return -1.0
    if "_yaw_" in name or name.endswith("_yaw_joint"):
        return -1.0
    return 1.0


def _mirror_index_and_sign(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index_by_name = {name: i for i, name in enumerate(names)}
    source = []
    signs = []
    for name in names:
        mirror_name = _swap_lr(name)
        if mirror_name not in index_by_name:
            raise ValueError(f"mirror counterpart {mirror_name!r} missing for {name!r}")
        source.append(index_by_name[mirror_name])
        signs.append(_joint_sign(name))
    return np.asarray(source, dtype=np.int64), np.asarray(signs, dtype=np.float32)


def _mirror_joint_like(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)[..., source].copy()
    out *= signs
    return out


def _mirror_polar_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    return out


def _mirror_axial_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    out[..., 2] *= -1.0
    return out


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float32)
    c1 = np.stack([arr[..., 0], arr[..., 2], arr[..., 4]], axis=-1)
    c2 = np.stack([arr[..., 1], arr[..., 3], arr[..., 5]], axis=-1)
    c1 = c1 / np.clip(np.linalg.norm(c1, axis=-1, keepdims=True), 1.0e-8, None)
    c2 = c2 - (c1 * c2).sum(axis=-1, keepdims=True) * c1
    c2 = c2 / np.clip(np.linalg.norm(c2, axis=-1, keepdims=True), 1.0e-8, None)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=-1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[..., :, :2], dtype=np.float32).reshape(*matrix.shape[:-2], 6)


def _mirror_rot6d(rot6d: np.ndarray) -> np.ndarray:
    out = np.asarray(rot6d, dtype=np.float32).copy()
    out *= np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    return out


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-8, None)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32, copy=False)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    out = np.empty(m.shape[:-2] + (4,), dtype=np.float32)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    mask = trace > 0.0
    s = np.sqrt(np.clip(trace[mask] + 1.0, 1.0e-8, None)) * 2.0
    out[mask, 0] = 0.25 * s
    out[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    out[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    out[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask_x = ~mask & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    s = np.sqrt(np.clip(1.0 + m[mask_x, 0, 0] - m[mask_x, 1, 1] - m[mask_x, 2, 2], 1.0e-8, None)) * 2.0
    out[mask_x, 0] = (m[mask_x, 2, 1] - m[mask_x, 1, 2]) / s
    out[mask_x, 1] = 0.25 * s
    out[mask_x, 2] = (m[mask_x, 0, 1] + m[mask_x, 1, 0]) / s
    out[mask_x, 3] = (m[mask_x, 0, 2] + m[mask_x, 2, 0]) / s

    mask_y = ~mask & ~mask_x & (m[..., 1, 1] > m[..., 2, 2])
    s = np.sqrt(np.clip(1.0 + m[mask_y, 1, 1] - m[mask_y, 0, 0] - m[mask_y, 2, 2], 1.0e-8, None)) * 2.0
    out[mask_y, 0] = (m[mask_y, 0, 2] - m[mask_y, 2, 0]) / s
    out[mask_y, 1] = (m[mask_y, 0, 1] + m[mask_y, 1, 0]) / s
    out[mask_y, 2] = 0.25 * s
    out[mask_y, 3] = (m[mask_y, 1, 2] + m[mask_y, 2, 1]) / s

    mask_z = ~mask & ~mask_x & ~mask_y
    s = np.sqrt(np.clip(1.0 + m[mask_z, 2, 2] - m[mask_z, 0, 0] - m[mask_z, 1, 1], 1.0e-8, None)) * 2.0
    out[mask_z, 0] = (m[mask_z, 1, 0] - m[mask_z, 0, 1]) / s
    out[mask_z, 1] = (m[mask_z, 0, 2] + m[mask_z, 2, 0]) / s
    out[mask_z, 2] = (m[mask_z, 1, 2] + m[mask_z, 2, 1]) / s
    out[mask_z, 3] = 0.25 * s

    out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1.0e-8, None)
    out[out[..., 0] < 0.0] *= -1.0
    return out


def _mirror_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    out[..., 3] *= -1.0
    return out


def _mirror_encoder(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[:, 0:29] = _mirror_joint_like(out[:, 0:29], source, signs)
    out[:, 29:58] = _mirror_joint_like(out[:, 29:58], source, signs)
    out[:, 58:61] = _mirror_polar_vec3(out[:, 58:61])
    out[:, 61:67] = _mirror_rot6d(out[:, 61:67])
    return out


def _mirror_proprio(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[:, 0:3] = _mirror_polar_vec3(out[:, 0:3])
    out[:, 3:6] = _mirror_polar_vec3(out[:, 3:6])
    out[:, 6:9] = _mirror_axial_vec3(out[:, 6:9])
    out[:, 9:38] = _mirror_joint_like(out[:, 9:38], source, signs)
    out[:, 38:67] = _mirror_joint_like(out[:, 38:67], source, signs)
    out[:, 67:96] = _mirror_joint_like(out[:, 67:96], source, signs)
    return out


def _mirror_policy_observation(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.shape[-1] != POLICY_OBS_DIM:
        return out
    out[:, 0:29] = _mirror_joint_like(out[:, 0:29], source, signs)
    out[:, 29:58] = _mirror_joint_like(out[:, 29:58], source, signs)
    out[:, 58:61] = _mirror_polar_vec3(out[:, 58:61])
    out[:, 61:67] = _mirror_rot6d(out[:, 61:67])
    out[:, 67:70] = _mirror_polar_vec3(out[:, 67:70])
    out[:, 70:73] = _mirror_axial_vec3(out[:, 70:73])
    out[:, 73:102] = _mirror_joint_like(out[:, 73:102], source, signs)
    out[:, 102:131] = _mirror_joint_like(out[:, 102:131], source, signs)
    out[:, 131:160] = _mirror_joint_like(out[:, 131:160], source, signs)
    return out


def _mirror_root_state(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.shape[-1] != 13:
        return out
    out[:, 0:3] = _mirror_polar_vec3(out[:, 0:3])
    out[:, 3:7] = _mirror_quat_wxyz(out[:, 3:7])
    out[:, 7:10] = _mirror_polar_vec3(out[:, 7:10])
    out[:, 10:13] = _mirror_axial_vec3(out[:, 10:13])
    return out


def _mirror_payload(payload: dict[str, np.ndarray], source: np.ndarray, signs: np.ndarray) -> dict[str, np.ndarray]:
    mirrored: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if key == "encoder_reference_input":
            mirrored[key] = _mirror_encoder(arr, source, signs)
        elif key == "decoder_proprio_input":
            mirrored[key] = _mirror_proprio(arr, source, signs)
        elif key in {"student_action", "teacher_action", "joint_position", "joint_velocity", "previous_action"}:
            mirrored[key] = _mirror_joint_like(arr, source, signs)
        elif key == "policy_observation":
            mirrored[key] = _mirror_policy_observation(arr, source, signs)
        elif key == "root_state":
            mirrored[key] = _mirror_root_state(arr)
        elif key in {"student_mu", "student_logvar", "student_latent"} and arr.shape[-1] == LATENT_DIM:
            mirrored[key] = arr.copy()
        else:
            mirrored[key] = arr.copy()
    return mirrored


def _stable_unique(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _select_balanced_indices(
    motion_name: np.ndarray,
    *,
    max_samples_per_motion: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    rng = np.random.default_rng(seed)
    motion_as_str = np.asarray(motion_name).astype(str, copy=False)
    unique_motion, first_index, inverse = np.unique(
        motion_as_str,
        return_index=True,
        return_inverse=True,
    )
    stable_order = np.argsort(first_index, kind="stable")
    selected_parts: list[np.ndarray] = []
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    for unique_id in stable_order:
        motion = str(unique_motion[unique_id])
        indices = np.flatnonzero(inverse == unique_id)
        before[motion] = int(indices.size)
        if indices.size > max_samples_per_motion:
            indices = np.sort(rng.choice(indices, size=max_samples_per_motion, replace=False))
        after[motion] = int(indices.size)
        selected_parts.append(indices)
    if not selected_parts:
        raise ValueError("no samples available for balanced selection")
    return np.concatenate(selected_parts, axis=0), before, after


def _load_payloads(paths: list[Path]) -> tuple[list[dict[str, np.ndarray]], list[str], list[dict[str, Any]], float]:
    loaded = []
    round_ids: list[str] = []
    input_meta: list[dict[str, Any]] = []
    frequency_hz: float | None = None
    for index, path in enumerate(paths, start=1):
        print(f"[balanced-dagger] loading {index}/{len(paths)}: {path}", flush=True)
        payload, meta = load_dagger_dataset(path)
        if frequency_hz is None:
            frequency_hz = float(meta.frequency_hz)
        elif abs(float(meta.frequency_hz) - frequency_hz) > 1.0e-6:
            raise ValueError(
                "all DAgger inputs must have the same frequency_hz: "
                f"expected {frequency_hz}, got {meta.frequency_hz} from {path}"
            )
        print(
            f"[balanced-dagger] loaded {path.name}: {int(payload['teacher_action'].shape[0])} samples",
            flush=True,
        )
        loaded.append(payload)
        round_ids.extend(meta.round_ids or (path.stem,))
        input_meta.append({"path": str(path), "sample_count": int(payload["teacher_action"].shape[0]), **meta.to_dict()})
    return loaded, round_ids, input_meta, float(frequency_hz or 50.0)


def _payload_motion_name(payload: dict[str, np.ndarray]) -> np.ndarray:
    sample_count = int(payload["teacher_action"].shape[0])
    motion_name = np.asarray(payload.get("motion_name"))
    if motion_name.shape != (sample_count,):
        return np.full((sample_count,), "motion", dtype="<U128")
    return motion_name.astype(str, copy=False)


def _select_balanced_entries(
    loaded: list[dict[str, np.ndarray]],
    *,
    max_samples_per_motion: int,
    seed: int,
    oversample_to_max: bool,
) -> tuple[list[str], dict[str, list[tuple[int, np.ndarray]]], dict[str, int], dict[str, int]]:
    rng = np.random.default_rng(seed)
    motions: list[str] = []
    motion_parts: dict[str, list[tuple[int, np.ndarray]]] = {}
    before: dict[str, int] = {}

    for dataset_index, payload in enumerate(loaded):
        names = _payload_motion_name(payload)
        for motion in _stable_unique(names):
            indices = np.flatnonzero(names == motion)
            if motion not in motion_parts:
                motions.append(motion)
                motion_parts[motion] = []
                before[motion] = 0
            motion_parts[motion].append((dataset_index, indices))
            before[motion] += int(indices.size)

    selected: dict[str, list[tuple[int, np.ndarray]]] = {}
    after: dict[str, int] = {}
    for motion in motions:
        parts = motion_parts[motion]
        total = sum(int(indices.size) for _, indices in parts)
        target = max_samples_per_motion if oversample_to_max else min(total, max_samples_per_motion)
        if total > target:
            chosen = np.sort(rng.choice(total, size=target, replace=False))
        elif total < target:
            chosen = np.sort(rng.choice(total, size=target, replace=True))
        else:
            chosen = np.arange(total, dtype=np.int64)

        selected_parts: list[tuple[int, np.ndarray]] = []
        offset = 0
        for dataset_index, indices in parts:
            stop = offset + int(indices.size)
            mask = (chosen >= offset) & (chosen < stop)
            if np.any(mask):
                selected_parts.append((dataset_index, indices[chosen[mask] - offset]))
            offset = stop
        selected[motion] = selected_parts
        after[motion] = int(target)
    return motions, selected, before, after


def _materialize_selected_payload(
    loaded: list[dict[str, np.ndarray]],
    motions: list[str],
    selected: dict[str, list[tuple[int, np.ndarray]]],
) -> dict[str, np.ndarray]:
    keys = list(loaded[0].keys())
    output: dict[str, np.ndarray] = {}
    entries = [
        (motion, dataset_index, indices)
        for motion in motions
        for dataset_index, indices in selected[motion]
        if indices.size
    ]
    sample_count = sum(int(indices.size) for _, _, indices in entries)
    for key in keys:
        first = np.asarray(loaded[0][key])
        out = np.empty((sample_count, *first.shape[1:]), dtype=first.dtype)
        cursor = 0
        print(f"[balanced-dagger] materializing key: {key} -> {out.shape}", flush=True)
        for chunk_index, (motion, dataset_index, indices) in enumerate(entries, start=1):
            source = np.asarray(loaded[dataset_index][key])
            next_cursor = cursor + int(indices.size)
            np.take(source, indices, axis=0, out=out[cursor:next_cursor])
            cursor = next_cursor
            if key == "policy_observation" and (chunk_index % 4 == 0 or chunk_index == len(entries)):
                print(
                    f"[balanced-dagger] materializing key: {key} progress {cursor}/{sample_count}",
                    flush=True,
                )
        output[key] = out
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input DAgger NPZ; can be repeated.")
    parser.add_argument("--teacher-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples-per-motion", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument(
        "--oversample-to-max",
        action="store_true",
        help="Sample with replacement so every motion contributes exactly --max-samples-per-motion samples.",
    )
    args = parser.parse_args()

    input_paths = [Path(path).expanduser() for path in args.input]
    loaded, round_ids, input_meta, frequency_hz = _load_payloads(input_paths)
    total_raw = sum(int(payload["teacher_action"].shape[0]) for payload in loaded)
    print(f"[balanced-dagger] total raw samples: {total_raw}", flush=True)
    joint_names, body_names = _load_reference_names(Path(args.teacher_map).expanduser())
    if len(joint_names) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} joints, got {len(joint_names)}")
    source, signs = _mirror_index_and_sign(joint_names)

    motions, selected_entries, before_counts, selected_counts = _select_balanced_entries(
        loaded,
        max_samples_per_motion=args.max_samples_per_motion,
        seed=args.seed,
        oversample_to_max=args.oversample_to_max,
    )
    print(
        f"[balanced-dagger] selected {sum(selected_counts.values())} samples before symmetry across {len(selected_counts)} motions",
        flush=True,
    )
    selected_payload = _materialize_selected_payload(loaded, motions, selected_entries)
    output_payload = selected_payload
    if not args.no_symmetry:
        print("[balanced-dagger] applying sagittal symmetry augmentation", flush=True)
        mirrored = _mirror_payload(selected_payload, source, signs)
        output_payload = {
            key: np.concatenate([np.asarray(selected_payload[key]), np.asarray(mirrored[key])], axis=0)
            for key in selected_payload
        }
    print(f"[balanced-dagger] saving {int(output_payload['teacher_action'].shape[0])} samples to {args.output}", flush=True)

    metadata = DAggerDatasetMetadata(
        frequency_hz=frequency_hz,
        joint_position_semantics="relative_to_default",
        source="balanced student-state DAgger with sagittal symmetry augmentation",
        round_ids=tuple(round_ids + ["balanced_sagittal_symmetry" if not args.no_symmetry else "balanced"]),
    )
    summary = save_dagger_dataset(args.output, output_payload, metadata)
    summary.update(
        {
            "input_datasets": input_meta,
            "joint_names": joint_names,
            "body_names": body_names,
            "mirror_joint_source_indices": source.tolist(),
            "mirror_joint_signs": signs.tolist(),
            "max_samples_per_motion": args.max_samples_per_motion,
            "oversample_to_max": args.oversample_to_max,
            "symmetry_enabled": not args.no_symmetry,
            "counts_before_balance": before_counts,
            "counts_after_balance_before_symmetry": selected_counts,
            "counts_after_symmetry": {
                key: int(value * (1 if args.no_symmetry else 2)) for key, value in selected_counts.items()
            },
            "policy_observation_mirror_layout": (
                "command[0:58], anchor_pos[58:61], anchor_rot6d[61:67], "
                "base_lin_vel[67:70], base_ang_vel[70:73], joint_pos[73:102], "
                "joint_vel[102:131], action[131:160]"
            ),
            "latent_fields": "student_mu/student_logvar/student_latent are copied because latent axes have no fixed sagittal semantics",
        }
    )
    sidecar = Path(args.output).with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
