"""Build Stage-3 state-latent data from VAE closed-loop rollout."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from beyondmimic_repro.contracts.state_latent import StateLatentMetadata, save_state_latent_dataset
from beyondmimic_repro.contracts.state_latent import load_state_latent_dataset
from beyondmimic_repro.contracts.vae_rollout import load_vae_rollout
from beyondmimic_repro.state import build_paper_hybrid_state_window, hybrid_state_schema, project_hybrid_state


LEGACY_TEACHER_WARNING = "This path is not the paper-faithful Stage-3 dataset path."


def _bool_array(payload: dict[str, np.ndarray], key: str, shape: tuple[int, int], default: bool) -> np.ndarray:
    if key not in payload:
        return np.full(shape, default, dtype=np.bool_)
    arr = np.asarray(payload[key], dtype=np.bool_)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    return arr


def _int_array(payload: dict[str, np.ndarray], key: str, shape: tuple[int, int], fallback: np.ndarray) -> np.ndarray:
    if key not in payload:
        return fallback.astype(np.int32, copy=False)
    arr = np.asarray(payload[key], dtype=np.int32)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    return arr


def _empty_windows(sequence_length: int, state_dim: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.zeros((0, sequence_length, state_dim), dtype=np.float32)
    latents = np.zeros((0, sequence_length, latent_dim), dtype=np.float32)
    tokens = np.zeros((0, sequence_length, state_dim + latent_dim), dtype=np.float32)
    return states, latents, tokens


def _require_array(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"paper state representation requires VAE rollout array {key!r}")
    return np.asarray(payload[key], dtype=np.float32)


def build_from_vae_rollout(
    vae_rollout_path: str | Path,
    output_path: str | Path,
    *,
    metadata: StateLatentMetadata | None = None,
    motion_id: int = 0,
    target_frequency_hz: float | None = None,
    compressed: bool = True,
    state_representation: str = "actual_state",
    projection_seed: int = 7,
    acceptance_key: str = "accepted",
    require_full_episode_accepted: bool = False,
    episode_acceptance_seconds: float | None = None,
) -> dict[str, object]:
    """Build actual VAE rollout state + latent windows."""
    if state_representation not in {"actual_state", "paper_hybrid", "paper_projected"}:
        raise ValueError(
            "state_representation must be one of "
            f"actual_state, paper_hybrid, paper_projected; got {state_representation!r}"
        )
    payload, rollout_meta = load_vae_rollout(vae_rollout_path)
    source_frequency_hz = float(rollout_meta.frequency_hz)
    if metadata is not None:
        meta = metadata
    else:
        meta = StateLatentMetadata(frequency_hz=float(target_frequency_hz or source_frequency_hz))
    if state_representation == "paper_hybrid":
        meta = replace(meta, state_schema="paper_hybrid_99d_yaw_centric")
    elif state_representation == "paper_projected":
        meta = replace(meta, state_schema="paper_projected_163d_yaw_centric")
    target_hz = float(meta.frequency_hz)
    if target_hz <= 0.0:
        raise ValueError("target frequency must be positive")
    frequency_ratio = source_frequency_hz / target_hz
    source_stride = int(round(frequency_ratio))
    if source_stride < 1 or abs(frequency_ratio - source_stride) > 1.0e-5:
        raise ValueError(
            f"target frequency must evenly divide source frequency: source={source_frequency_hz} target={target_hz}"
        )
    source_states = np.asarray(payload["actual_state"], dtype=np.float32)
    source_latents = np.asarray(payload["latent"], dtype=np.float32)
    output_state_dim = int(source_states.shape[-1])
    paper_raw: dict[str, np.ndarray] = {}
    projection_matrix: np.ndarray | None = None
    projection_inverse: np.ndarray | None = None
    state_slices: dict[str, list[int]] | None = None
    if state_representation != "actual_state":
        paper_raw = {
            key: _require_array(payload, key)
            for key in ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pos_w", "body_lin_vel_w"]
        }
        schema = hybrid_state_schema()
        if paper_raw["body_pos_w"].shape[-2:] != (schema.target_body_count, 3):
            raise ValueError(
                "paper state requires body_pos_w/body_lin_vel_w shape [E,T,14,3], "
                f"got {paper_raw['body_pos_w'].shape} and {paper_raw['body_lin_vel_w'].shape}"
            )
        if state_representation == "paper_projected":
            _, projection_matrix, projection_inverse = project_hybrid_state(
                np.zeros((1, schema.state_dim), dtype=np.float32),
                seed=projection_seed,
                schema=schema,
            )
            output_state_dim = schema.projected_dim
        else:
            output_state_dim = schema.state_dim
    episode_count, step_count = source_states.shape[:2]
    shape = (episode_count, step_count)
    if acceptance_key != "accepted" and acceptance_key not in payload:
        raise ValueError(f"acceptance_key {acceptance_key!r} is not present in VAE rollout")
    accepted = _bool_array(payload, acceptance_key, shape, True)
    done = _bool_array(payload, "done", shape, False)
    fallback_episode = np.broadcast_to(np.arange(episode_count, dtype=np.int32)[:, None], shape)
    fallback_time = np.broadcast_to(np.arange(step_count, dtype=np.int32)[None, :], shape)
    source_episode = _int_array(payload, "episode_id", shape, fallback_episode)
    source_time = _int_array(payload, "time_index", shape, fallback_time)
    source_reference_frame = _int_array(payload, "reference_frame_index", shape, fallback_time)

    state_windows: list[np.ndarray] = []
    latent_windows: list[np.ndarray] = []
    source_episode_ids: list[int] = []
    source_environment_ids: list[int] = []
    start_time_indices: list[int] = []
    reference_frame_indices: list[int] = []
    sequence_length = meta.sequence_length
    valid_token = np.logical_and(accepted, np.logical_not(done))
    episode_valid = np.ones(episode_count, dtype=np.bool_)
    episode_acceptance_steps = 0
    if episode_acceptance_seconds is not None:
        if episode_acceptance_seconds <= 0.0:
            raise ValueError("episode_acceptance_seconds must be positive when provided")
        episode_acceptance_steps = int(np.ceil(float(episode_acceptance_seconds) * source_frequency_hz))
        if episode_acceptance_steps > step_count:
            raise ValueError(
                "rollout is shorter than requested episode acceptance window: "
                f"required_steps={episode_acceptance_steps} source_steps={step_count}"
            )
        require_full_episode_accepted = True
    elif require_full_episode_accepted:
        episode_acceptance_steps = step_count
    if require_full_episode_accepted:
        episode_valid = valid_token[:, :episode_acceptance_steps].all(axis=1)
    window_start_stride = source_stride
    last_window_start_exclusive = step_count - (sequence_length - 1) * source_stride
    if last_window_start_exclusive > 0:
        for env_idx in range(episode_count):
            if not bool(episode_valid[env_idx]):
                continue
            for start in range(0, last_window_start_exclusive, window_start_stride):
                sample_indices = start + np.arange(sequence_length) * source_stride
                source_stop = int(sample_indices[-1]) + 1
                if not bool(valid_token[env_idx, start:source_stop].all()):
                    continue
                if state_representation == "actual_state":
                    state_window = source_states[env_idx, sample_indices]
                else:
                    state_window, state_slices = build_paper_hybrid_state_window(
                        paper_raw["root_pos_w"][env_idx, sample_indices],
                        paper_raw["root_quat_w"][env_idx, sample_indices],
                        paper_raw["root_lin_vel_w"][env_idx, sample_indices],
                        paper_raw["root_ang_vel_w"][env_idx, sample_indices],
                        paper_raw["body_pos_w"][env_idx, sample_indices],
                        paper_raw["body_lin_vel_w"][env_idx, sample_indices],
                        current_index=meta.past_steps,
                        quat_format="wxyz",
                    )
                    if state_representation == "paper_projected":
                        if projection_matrix is None:
                            raise RuntimeError("projection matrix was not initialized")
                        state_window = (state_window @ projection_matrix.T).astype(np.float32)
                state_windows.append(state_window.astype(np.float32, copy=False))
                latent_windows.append(source_latents[env_idx, sample_indices])
                source_episode_ids.append(int(source_episode[env_idx, start]))
                source_environment_ids.append(env_idx)
                start_time_indices.append(int(source_time[env_idx, start]))
                current_source_index = int(sample_indices[min(meta.past_steps, len(sample_indices) - 1)])
                reference_frame_indices.append(int(source_reference_frame[env_idx, current_source_index]))

    if state_windows:
        states = np.stack(state_windows, axis=0).astype(np.float32)
        latents = np.stack(latent_windows, axis=0).astype(np.float32)
        tokens = np.concatenate([states, latents], axis=-1).astype(np.float32)
    else:
        states, latents, tokens = _empty_windows(sequence_length, output_state_dim, source_latents.shape[-1])
    window_count = tokens.shape[0]
    if window_count:
        normalization_mean = tokens.mean(axis=(0, 1)).astype(np.float32)
        normalization_std = tokens.std(axis=(0, 1)).astype(np.float32)
        normalization_std = np.where(normalization_std < 1.0e-6, 1.0, normalization_std).astype(np.float32)
    else:
        normalization_mean = np.zeros(tokens.shape[-1], dtype=np.float32)
        normalization_std = np.ones(tokens.shape[-1], dtype=np.float32)
    out_payload = {
        "states": states,
        "latents": latents,
        "tokens": tokens,
        "valid_mask": np.ones((window_count, meta.sequence_length), dtype=np.bool_),
        "episode_id": np.asarray(source_episode_ids, dtype=np.int32),
        "motion_id": np.full(window_count, int(motion_id), dtype=np.int32),
        "time_index": np.asarray(start_time_indices, dtype=np.int32),
        "reference_frame_index": np.asarray(reference_frame_indices, dtype=np.int32),
        "source_environment_id": np.asarray(source_environment_ids, dtype=np.int32),
        "frequency_hz": np.full(window_count, meta.frequency_hz, dtype=np.float32),
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "state_representation": np.asarray(state_representation),
        "state_schema": np.asarray(meta.state_schema),
        "state_dim": np.asarray(states.shape[-1], dtype=np.int32),
        "latent_dim": np.asarray(latents.shape[-1], dtype=np.int32),
        "token_dim": np.asarray(tokens.shape[-1], dtype=np.int32),
        "causal_manifest_json": np.asarray(
            json.dumps(
                {
                    "token_semantics": (
                        "token[k] = concat(state_{t+k}, latent_{t+k}) from one continuous real "
                        f"{source_frequency_hz:g}Hz rollout."
                    ),
                    "state_t": "Isaac state recorded before executing action at time t.",
                    "latent_t": "Frozen VAE encoder mean for the same pre-action state/reference at time t.",
                    "clean_action_t": "Frozen VAE decoder action from latent_t and proprioception_t.",
                    "executed_action_t": "clean_action_t plus OU noise when OU collection is enabled.",
                    "next_state_t_plus_1": "The next row in the rollout after env.step(executed_action_t).",
                    "window_sampling": "No temporal offset is inserted inside a token; source_stride records any frequency downsampling.",
                },
                sort_keys=True,
            )
        ),
    }
    if projection_matrix is not None and projection_inverse is not None:
        out_payload["state_projection_matrix"] = projection_matrix.astype(np.float32)
        out_payload["state_projection_inverse"] = projection_inverse.astype(np.float32)
    if state_slices is not None:
        out_payload["state_slices_json"] = np.asarray(json.dumps(state_slices, sort_keys=True))
    summary = save_state_latent_dataset(output_path, out_payload, meta, compressed=compressed)
    possible_windows_per_episode = len(range(0, max(0, last_window_start_exclusive), window_start_stride))
    possible_windows = possible_windows_per_episode * episode_count
    summary.update(
        {
            "source_rollout": str(vae_rollout_path),
            "source_episode_count": int(episode_count),
            "source_steps": int(step_count),
            "source_frequency_hz": source_frequency_hz,
            "source_stride": int(source_stride),
            "target_frequency_hz": target_hz,
            "state_representation": state_representation,
            "state_schema": meta.state_schema,
            "state_dim": int(states.shape[-1]),
            "projection_seed": int(projection_seed) if state_representation == "paper_projected" else None,
            "acceptance_key": acceptance_key,
            "require_full_episode_accepted": bool(require_full_episode_accepted),
            "episode_acceptance_seconds": episode_acceptance_seconds,
            "episode_acceptance_steps": int(episode_acceptance_steps),
            "accepted_episode_count": int(episode_valid.sum()),
            "rejected_episode_count": int(episode_count - episode_valid.sum()),
            "possible_window_count": int(possible_windows),
            "discarded_window_count": int(possible_windows - window_count),
            "motion_id": int(motion_id),
            "causal_manifest": json.loads(str(out_payload["causal_manifest_json"])),
        }
    )
    return summary


def merge_state_latent_datasets(
    dataset_paths: list[str | Path],
    output_path: str | Path,
    *,
    compressed: bool = True,
) -> dict[str, object]:
    """Merge compatible state-latent shards and recompute token normalization."""
    if not dataset_paths:
        raise ValueError("at least one state-latent shard is required")
    loaded = [load_state_latent_dataset(path) for path in dataset_paths]
    first_payload, first_meta = loaded[0]
    first_tokens = np.asarray(first_payload["tokens"], dtype=np.float32)
    if first_tokens.ndim != 3:
        raise ValueError(f"tokens must be [N,T,D], got {first_tokens.shape}")
    concat_keys = [
        key
        for key, value in first_payload.items()
        if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == first_tokens.shape[0]
    ]
    concat_keys = [key for key in concat_keys if key not in {"normalization_mean", "normalization_std"}]

    merged: dict[str, np.ndarray] = {}
    source_summaries: list[dict[str, object]] = []
    for shard_idx, (path, (payload, meta)) in enumerate(zip(dataset_paths, loaded)):
        if meta != first_meta:
            raise ValueError(f"metadata mismatch in shard {path}: {meta} != {first_meta}")
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        if tokens.shape[1:] != first_tokens.shape[1:]:
            raise ValueError(f"token shape mismatch in shard {path}: {tokens.shape[1:]} != {first_tokens.shape[1:]}")
        source_summaries.append({"path": str(path), "window_count": int(tokens.shape[0])})
        for key in concat_keys:
            arr = np.asarray(payload[key])
            if arr.shape[0] != tokens.shape[0]:
                raise ValueError(f"{key} first dim mismatch in shard {path}: {arr.shape[0]} != {tokens.shape[0]}")
            merged.setdefault(key, [])
            merged[key].append(arr)
        for key, value in payload.items():
            if key in concat_keys or key in {"normalization_mean", "normalization_std"}:
                continue
            arr = np.asarray(value)
            if key not in merged:
                merged[key] = arr
            elif not np.array_equal(np.asarray(merged[key]), arr):
                raise ValueError(f"non-window array {key!r} differs in shard {path}")
        if shard_idx == 0:
            continue

    out_payload: dict[str, np.ndarray] = {}
    for key, value in merged.items():
        if key in concat_keys:
            out_payload[key] = np.concatenate(value, axis=0)
        else:
            out_payload[key] = np.asarray(value)
    tokens = np.asarray(out_payload["tokens"], dtype=np.float32)
    if tokens.shape[0]:
        normalization_mean = tokens.mean(axis=(0, 1)).astype(np.float32)
        normalization_std = tokens.std(axis=(0, 1)).astype(np.float32)
        normalization_std = np.where(normalization_std < 1.0e-6, 1.0, normalization_std).astype(np.float32)
    else:
        normalization_mean = np.zeros(tokens.shape[-1], dtype=np.float32)
        normalization_std = np.ones(tokens.shape[-1], dtype=np.float32)
    out_payload["normalization_mean"] = normalization_mean
    out_payload["normalization_std"] = normalization_std
    out_payload["state_schema"] = np.asarray(first_meta.state_schema)
    out_payload["state_dim"] = np.asarray(tokens.shape[-1] - out_payload["latents"].shape[-1], dtype=np.int32)
    out_payload["latent_dim"] = np.asarray(out_payload["latents"].shape[-1], dtype=np.int32)
    out_payload["token_dim"] = np.asarray(tokens.shape[-1], dtype=np.int32)
    if "state_representation" not in out_payload:
        if first_meta.state_schema == "paper_projected_163d_yaw_centric":
            state_representation = "paper_projected"
        elif first_meta.state_schema == "paper_hybrid_99d_yaw_centric":
            state_representation = "paper_hybrid"
        else:
            state_representation = "actual_state"
        out_payload["state_representation"] = np.asarray(state_representation)

    summary = save_state_latent_dataset(output_path, out_payload, first_meta, compressed=compressed)
    summary.update(
        {
            "source_shards": source_summaries,
            "state_dim": int(out_payload["states"].shape[-1]),
            "token_dim": int(tokens.shape[-1]),
            "total_window_count": int(tokens.shape[0]),
        }
    )
    return summary


def build_state_latent_from_teacher_legacy(*_: object, **__: object) -> None:
    """Legacy guard for teacher-rollout direct encodings."""
    warnings.warn(LEGACY_TEACHER_WARNING, stacklevel=2)
    print(LEGACY_TEACHER_WARNING)
    raise RuntimeError(LEGACY_TEACHER_WARNING)
