"""Sagittal mirror augmentation for paper VAE tensors."""

from __future__ import annotations

import numpy as np


def swap_lr(name: str) -> str:
    if "left_" in name:
        return name.replace("left_", "right_", 1)
    if "right_" in name:
        return name.replace("right_", "left_", 1)
    return name


def joint_sign(name: str) -> float:
    if "_roll_" in name or name.endswith("_roll_joint"):
        return -1.0
    if "_yaw_" in name or name.endswith("_yaw_joint"):
        return -1.0
    return 1.0


def mirror_index_and_sign(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index_by_name = {name: i for i, name in enumerate(joint_names)}
    source: list[int] = []
    signs: list[float] = []
    for name in joint_names:
        mirror_name = swap_lr(name)
        if mirror_name not in index_by_name:
            raise ValueError(f"mirror counterpart {mirror_name!r} missing for {name!r}")
        source.append(index_by_name[mirror_name])
        signs.append(joint_sign(name))
    return np.asarray(source, dtype=np.int64), np.asarray(signs, dtype=np.float32)


def mirror_joint_like(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)[..., source].copy()
    out *= signs
    return out


def mirror_polar_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    return out


def mirror_axial_vec3(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    out[..., 2] *= -1.0
    return out


def mirror_rot6d(rot6d: np.ndarray) -> np.ndarray:
    out = np.asarray(rot6d, dtype=np.float32).copy()
    out *= np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    return out


def mirror_encoder_reference(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 0:29] = mirror_joint_like(out[..., 0:29], source, signs)
    out[..., 29:58] = mirror_joint_like(out[..., 29:58], source, signs)
    out[..., 58:61] = mirror_polar_vec3(out[..., 58:61])
    out[..., 61:67] = mirror_rot6d(out[..., 61:67])
    return out


def mirror_decoder_proprio(arr: np.ndarray, source: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[..., 0:3] = mirror_polar_vec3(out[..., 0:3])
    out[..., 3:6] = mirror_polar_vec3(out[..., 3:6])
    out[..., 6:9] = mirror_axial_vec3(out[..., 6:9])
    out[..., 9:38] = mirror_joint_like(out[..., 9:38], source, signs)
    out[..., 38:67] = mirror_joint_like(out[..., 38:67], source, signs)
    out[..., 67:96] = mirror_joint_like(out[..., 67:96], source, signs)
    return out


def mirror_vae_arrays(
    encoder: np.ndarray,
    proprio: np.ndarray,
    action: np.ndarray,
    *,
    source: np.ndarray,
    signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        mirror_encoder_reference(encoder, source, signs),
        mirror_decoder_proprio(proprio, source, signs),
        mirror_joint_like(action, source, signs),
    )
