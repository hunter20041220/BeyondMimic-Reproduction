#!/usr/bin/env python3
"""Analyze natural gait-speed bridge windows in locomotion VAE+OU rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROLLOUT_DIR = Path("outputs/rtx4090_stage23/vae_D1D5_stable_walkrun_ou_state_latent_gpu7/rollouts")
MOTION_FILES = {
    0: ("walk1_subject1", "vae_D1D5_stable_ou_walk1_subject1_128env_250steps_sigma0.10.npz"),
    1: ("walk2_subject4", "vae_D1D5_stable_ou_walk2_subject4_128env_250steps_sigma0.10.npz"),
    2: ("run2_subject1", "vae_D1D5_stable_ou_run2_subject1_128env_250steps_sigma0.10.npz"),
    3: ("sprint1_subject2", "vae_D1D5_stable_ou_sprint1_subject2_128env_250steps_sigma0.10.npz"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_percentiles(values: np.ndarray, percentiles: list[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"p{int(p):02d}": None for p in percentiles}
    return {f"p{int(p):02d}": float(np.percentile(finite, p)) for p in percentiles}


def _body_index(names: np.ndarray, target: str) -> int:
    values = [str(v) for v in names.tolist()]
    if target not in values:
        raise ValueError(f"body {target!r} missing from rollout body_names={values}")
    return values.index(target)


def _stable_episode_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    accepted = np.asarray(data["accepted"], dtype=np.bool_)
    physical = np.asarray(data["physical_accepted"], dtype=np.bool_) if "physical_accepted" in data.files else accepted
    done = np.asarray(data["done"], dtype=np.bool_) if "done" in data.files else np.zeros_like(accepted)
    fall = np.asarray(data["physical_fall"], dtype=np.bool_) if "physical_fall" in data.files else np.zeros_like(accepted)
    return accepted.all(axis=1) & physical.all(axis=1) & (~done).all(axis=1) & (~fall).all(axis=1)


def _contact_proxy(data: np.lib.npyio.NpzFile, stable_env: np.ndarray) -> dict[str, np.ndarray]:
    body_names = np.asarray(data["body_names"])
    left_idx = _body_index(body_names, "left_ankle_roll_link")
    right_idx = _body_index(body_names, "right_ankle_roll_link")
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
    left_z = body_pos[:, :, left_idx, 2]
    right_z = body_pos[:, :, right_idx, 2]
    left_vz = body_vel[:, :, left_idx, 2]
    right_vz = body_vel[:, :, right_idx, 2]
    stable_min_z = np.minimum(left_z[stable_env], right_z[stable_env])
    ground = float(np.percentile(stable_min_z, 3.0)) if stable_min_z.size else float(np.percentile(np.minimum(left_z, right_z), 3.0))
    z_eps = 0.045
    vz_eps = 0.75
    left_contact = (left_z <= ground + z_eps) & (np.abs(left_vz) <= vz_eps)
    right_contact = (right_z <= ground + z_eps) & (np.abs(right_vz) <= vz_eps)
    foot_flight = ~(left_contact | right_contact)
    if "contact_force_max" in data.files:
        force_flight = np.asarray(data["contact_force_max"], dtype=np.float32) < 1.0
        flight_proxy = force_flight
    else:
        force_flight = foot_flight
        flight_proxy = foot_flight
    pattern = np.zeros(left_contact.shape, dtype=np.int8)
    pattern[left_contact & ~right_contact] = 1
    pattern[right_contact & ~left_contact] = 2
    pattern[left_contact & right_contact] = 3
    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "foot_flight": foot_flight,
        "force_flight": force_flight,
        "flight_proxy": flight_proxy,
        "contact_pattern": pattern,
        "ground_proxy_z": np.asarray(ground, dtype=np.float32),
    }


def _motion_window_features(
    *,
    motion_id: int,
    motion_name: str,
    path: Path,
    sequence_length: int,
    frequency_hz: float,
    walk_speed: float,
    run_speed: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = np.load(path)
    stable_env = _stable_episode_mask(data)
    root_xy = np.asarray(data["root_pos_w"], dtype=np.float32)[..., :2]
    step_delta = np.diff(root_xy, axis=1, prepend=root_xy[:, :1])
    speed = np.linalg.norm(step_delta, axis=-1) * float(frequency_hz)
    speed_derivative = np.diff(speed, axis=1, prepend=speed[:, :1]) * float(frequency_hz)
    contact = _contact_proxy(data, stable_env)
    accepted = np.asarray(data["accepted"], dtype=np.bool_)
    physical = np.asarray(data["physical_accepted"], dtype=np.bool_) if "physical_accepted" in data.files else accepted
    valid_token = accepted & physical
    step_count = speed.shape[1]
    last_start = step_count - sequence_length + 1
    if last_start <= 0:
        raise ValueError(f"{path} too short for sequence_length={sequence_length}")

    env_ids: list[int] = []
    starts: list[int] = []
    start_speed: list[float] = []
    end_speed: list[float] = []
    mean_speed: list[float] = []
    max_speed: list[float] = []
    mean_derivative: list[float] = []
    flight_fraction: list[float] = []
    contact_change: list[bool] = []
    walk_to_jog: list[bool] = []
    jog_to_run: list[bool] = []
    run_to_walk: list[bool] = []
    intermediate: list[bool] = []

    jog_low = walk_speed * 1.15
    run_entry = run_speed * 0.80
    walk_band_high = walk_speed * 1.25
    min_delta = 0.22
    for env_id in np.flatnonzero(stable_env):
        for start in range(last_start):
            idx = slice(start, start + sequence_length)
            if not bool(valid_token[env_id, idx].all()):
                continue
            window_speed = speed[env_id, idx]
            s0 = float(np.mean(window_speed[:4]))
            s1 = float(np.mean(window_speed[-4:]))
            sm = float(np.mean(window_speed))
            sx = float(np.max(window_speed))
            deriv = float((s1 - s0) / ((sequence_length - 1) / frequency_hz))
            pattern = contact["contact_pattern"][env_id, idx]
            flight = contact["flight_proxy"][env_id, idx]
            is_intermediate = bool(np.any((window_speed >= walk_speed) & (window_speed <= run_speed)))
            is_walk_to_jog = bool(s0 <= walk_band_high and s1 >= jog_low and s1 < run_entry and (s1 - s0) >= min_delta)
            is_jog_to_run = bool(s0 >= walk_speed and s0 < run_entry and s1 >= run_entry and (s1 - s0) >= min_delta)
            is_run_to_walk = bool(s0 >= run_entry and s1 <= walk_band_high and (s0 - s1) >= min_delta)
            env_ids.append(int(env_id))
            starts.append(int(start))
            start_speed.append(s0)
            end_speed.append(s1)
            mean_speed.append(sm)
            max_speed.append(sx)
            mean_derivative.append(deriv)
            flight_fraction.append(float(np.mean(flight.astype(np.float32))))
            contact_change.append(bool(np.any(pattern[1:] != pattern[:-1]) or np.any(flight[1:] != flight[:-1])))
            intermediate.append(is_intermediate)
            walk_to_jog.append(is_walk_to_jog)
            jog_to_run.append(is_jog_to_run)
            run_to_walk.append(is_run_to_walk)

    features = {
        "motion_id": np.full(len(starts), motion_id, dtype=np.int16),
        "motion_name": np.full(len(starts), motion_name),
        "source_environment_id": np.asarray(env_ids, dtype=np.int16),
        "time_index": np.asarray(starts, dtype=np.int16),
        "start_speed": np.asarray(start_speed, dtype=np.float32),
        "end_speed": np.asarray(end_speed, dtype=np.float32),
        "mean_speed": np.asarray(mean_speed, dtype=np.float32),
        "max_speed": np.asarray(max_speed, dtype=np.float32),
        "mean_speed_derivative": np.asarray(mean_derivative, dtype=np.float32),
        "flight_fraction": np.asarray(flight_fraction, dtype=np.float32),
        "contact_change": np.asarray(contact_change, dtype=np.bool_),
        "intermediate_speed": np.asarray(intermediate, dtype=np.bool_),
        "walk_to_jog": np.asarray(walk_to_jog, dtype=np.bool_),
        "jog_to_run": np.asarray(jog_to_run, dtype=np.bool_),
        "run_to_walk": np.asarray(run_to_walk, dtype=np.bool_),
    }
    stable_speed = speed[stable_env]
    token_flight = contact["flight_proxy"][stable_env]
    pattern = contact["contact_pattern"][stable_env]
    summary = {
        "motion_id": int(motion_id),
        "motion_name": motion_name,
        "rollout": str(path),
        "stable_episode_count": int(stable_env.sum()),
        "total_episode_count": int(stable_env.size),
        "window_count": int(len(starts)),
        "speed": {
            "median": float(np.median(stable_speed)) if stable_speed.size else None,
            **_safe_percentiles(stable_speed, [10, 25, 50, 75, 90]),
        },
        "speed_derivative": _safe_percentiles(speed_derivative[stable_env], [10, 25, 50, 75, 90]),
        "token_intermediate_fraction": float(np.mean((stable_speed >= walk_speed) & (stable_speed <= run_speed)))
        if stable_speed.size
        else 0.0,
        "token_flight_proxy_fraction": float(np.mean(token_flight)) if token_flight.size else 0.0,
        "contact_pattern_fraction": {
            "flight": float(np.mean(pattern == 0)) if pattern.size else 0.0,
            "left_only": float(np.mean(pattern == 1)) if pattern.size else 0.0,
            "right_only": float(np.mean(pattern == 2)) if pattern.size else 0.0,
            "double": float(np.mean(pattern == 3)) if pattern.size else 0.0,
        },
        "window_counts": {
            "intermediate_speed": int(features["intermediate_speed"].sum()),
            "walk_to_jog": int(features["walk_to_jog"].sum()),
            "jog_to_run": int(features["jog_to_run"].sum()),
            "run_to_walk": int(features["run_to_walk"].sum()),
            "contact_change": int(features["contact_change"].sum()),
            "flight_fraction_positive": int((features["flight_fraction"] > 0.0).sum()),
        },
    }
    return summary, features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--walk-speed", type=float, default=0.468394304424024)
    parser.add_argument("--run-speed", type=float, default=1.3162681813025874)
    parser.add_argument("--sequence-length", type=int, default=21)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    args = parser.parse_args()

    summaries: dict[str, Any] = {
        "walk_speed_target": float(args.walk_speed),
        "run_speed_target": float(args.run_speed),
        "run80_speed_threshold": float(args.run_speed * 0.80),
        "sequence_length": int(args.sequence_length),
        "frequency_hz": float(args.frequency_hz),
        "motions": {},
    }
    feature_chunks: dict[str, list[np.ndarray]] = {}
    for motion_id, (motion_name, filename) in MOTION_FILES.items():
        summary, features = _motion_window_features(
            motion_id=motion_id,
            motion_name=motion_name,
            path=args.rollout_dir / filename,
            sequence_length=args.sequence_length,
            frequency_hz=args.frequency_hz,
            walk_speed=args.walk_speed,
            run_speed=args.run_speed,
        )
        summaries["motions"][motion_name] = summary
        for key, value in features.items():
            feature_chunks.setdefault(key, []).append(value)
    merged = {key: np.concatenate(value, axis=0) for key, value in feature_chunks.items()}
    total = int(merged["time_index"].shape[0])
    bridge_any = merged["walk_to_jog"] | merged["jog_to_run"] | merged["run_to_walk"]
    summaries["total_window_count"] = total
    summaries["total_counts"] = {
        "intermediate_speed": int(merged["intermediate_speed"].sum()),
        "walk_to_jog": int(merged["walk_to_jog"].sum()),
        "jog_to_run": int(merged["jog_to_run"].sum()),
        "run_to_walk": int(merged["run_to_walk"].sum()),
        "bridge_any": int(bridge_any.sum()),
        "contact_change": int(merged["contact_change"].sum()),
        "flight_fraction_positive": int((merged["flight_fraction"] > 0.0).sum()),
    }
    summaries["bridge_windows_exist"] = bool(
        summaries["total_counts"]["intermediate_speed"] > 0
        and summaries["total_counts"]["walk_to_jog"] > 0
        and summaries["total_counts"]["jog_to_run"] > 0
        and summaries["total_counts"]["run_to_walk"] > 0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "locomotion_bridge_analysis.json", summaries)
    np.savez_compressed(args.output_dir / "locomotion_bridge_window_features.npz", **merged)
    print(json.dumps(summaries, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
