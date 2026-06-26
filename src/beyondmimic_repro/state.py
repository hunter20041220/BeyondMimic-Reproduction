"""State projection helpers for paper-formula debug artifacts."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from beyondmimic_repro.validation import ensure_finite


ROOT_STATE_DIM = 15
DEFAULT_TARGET_BODY_COUNT = 14
TARGET_BODY_FEATURE_DIM = DEFAULT_TARGET_BODY_COUNT * 6
HYBRID_STATE_DIM = ROOT_STATE_DIM + TARGET_BODY_FEATURE_DIM
EMPHASIS_COEFFICIENT = 6
DEFAULT_GAUSSIAN_ROWS = 64


@dataclass(frozen=True)
class HybridStateSchema:
    """Paper S3 hybrid state layout for one trajectory timestep.

    The default 99-D layout is root pose/twist in the current yaw-centric
    character frame plus target-body position/velocity in each local root
    yaw frame.  It intentionally omits reference-motion commands.
    """

    target_body_count: int = DEFAULT_TARGET_BODY_COUNT
    root_dim: int = ROOT_STATE_DIM
    coefficient: int = EMPHASIS_COEFFICIENT
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS

    @property
    def body_feature_dim(self) -> int:
        return self.target_body_count * 6

    @property
    def state_dim(self) -> int:
        return self.root_dim + self.body_feature_dim

    @property
    def projected_dim(self) -> int:
        return self.state_dim + self.gaussian_rows

    @property
    def slices(self) -> dict[str, list[int]]:
        body_pos_start = self.root_dim
        body_vel_start = body_pos_start + 3 * self.target_body_count
        return {
            "root_pos_rel_current_frame": [0, 3],
            "root_rot6d_rel_current_frame": [3, 9],
            "root_lin_vel_rel_current_frame": [9, 12],
            "root_ang_vel_rel_current_frame": [12, 15],
            "body_pos_local_root_frame": [body_pos_start, body_vel_start],
            "body_lin_vel_local_root_frame": [body_vel_start, self.state_dim],
        }

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.update(
            {
                "body_feature_dim": self.body_feature_dim,
                "state_dim": self.state_dim,
                "projected_dim": self.projected_dim,
                "slices": self.slices,
            }
        )
        return data


def hybrid_state_schema(
    target_body_count: int = DEFAULT_TARGET_BODY_COUNT,
    coefficient: int = EMPHASIS_COEFFICIENT,
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS,
) -> HybridStateSchema:
    """Return the default paper S3 99-D hybrid state schema."""
    if target_body_count <= 0 or coefficient <= 0 or gaussian_rows <= 0:
        raise ValueError("target_body_count, coefficient, and gaussian_rows must be positive")
    return HybridStateSchema(target_body_count=target_body_count, coefficient=coefficient, gaussian_rows=gaussian_rows)


def validate_hybrid_state(states: np.ndarray, schema: HybridStateSchema | None = None) -> np.ndarray:
    """Validate finite paper hybrid state tensors with last dimension 99."""
    schema = schema or hybrid_state_schema()
    arr = ensure_finite("hybrid_states", states)
    if arr.shape[-1] != schema.state_dim:
        raise ValueError(f"hybrid state last dim must be {schema.state_dim}, got {arr.shape}")
    return arr


def valid_contiguous_window_mask(
    dones: np.ndarray,
    motion_time_steps: np.ndarray,
    sequence_length: int,
    *,
    timeouts: np.ndarray | None = None,
    reject_timeouts: bool = False,
) -> np.ndarray:
    """Return valid ``[start, env]`` windows for continuous rollout training.

    Teacher rollout shards store one row per simulator step and environment.
    For state-latent diffusion training, a window must not cross a reset and
    the reference-motion timestep must advance by exactly one frame across the
    whole sequence.  This helper is intentionally strict: ambiguous windows are
    rejected instead of being silently spliced into training data.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    done_arr = np.asarray(dones, dtype=np.bool_)
    time_arr = ensure_finite("motion_time_steps", motion_time_steps).astype(np.int64, copy=False)
    if done_arr.ndim == 1:
        done_arr = done_arr[:, None]
    if time_arr.ndim == 1:
        time_arr = time_arr[:, None]
    if done_arr.shape != time_arr.shape:
        raise ValueError(f"dones and motion_time_steps must have the same shape, got {done_arr.shape} and {time_arr.shape}")
    if done_arr.ndim != 2:
        raise ValueError(f"dones and motion_time_steps must have shape [T,E] or [T], got {done_arr.shape}")

    timeout_arr = None
    if timeouts is not None:
        timeout_arr = np.asarray(timeouts, dtype=np.bool_)
        if timeout_arr.ndim == 1:
            timeout_arr = timeout_arr[:, None]
        if timeout_arr.shape != done_arr.shape:
            raise ValueError(f"timeouts must match dones shape {done_arr.shape}, got {timeout_arr.shape}")

    step_count, env_count = done_arr.shape
    if sequence_length > step_count:
        return np.zeros((0, env_count), dtype=np.bool_)

    mask = np.zeros((step_count - sequence_length + 1, env_count), dtype=np.bool_)
    for start in range(mask.shape[0]):
        end = start + sequence_length
        window_done = np.any(done_arr[start:end], axis=0)
        window_timeout = np.zeros(env_count, dtype=np.bool_)
        if reject_timeouts and timeout_arr is not None:
            window_timeout = np.any(timeout_arr[start:end], axis=0)
        continuous_time = np.all(np.diff(time_arr[start:end], axis=0) == 1, axis=0)
        mask[start] = (~window_done) & (~window_timeout) & continuous_time
    return mask


def _yaw_matrix_batch(yaws: np.ndarray) -> np.ndarray:
    yaws = ensure_finite("yaws", yaws)
    c = np.cos(yaws)
    s = np.sin(yaws)
    mats = np.zeros(yaws.shape + (3, 3), dtype=np.float64)
    mats[..., 0, 0] = c
    mats[..., 0, 1] = -s
    mats[..., 1, 0] = s
    mats[..., 1, 1] = c
    mats[..., 2, 2] = 1.0
    return mats


def quat_to_matrix_array(quaternions: np.ndarray, quat_format: str = "xyzw") -> np.ndarray:
    """Convert finite quaternion arrays to rotation matrices.

    The paper-state fixture stores quaternions as ``xyzw``.  IsaacLab runtime
    tensors commonly use ``wxyz``.  The format is therefore explicit so rollout
    datasets cannot silently mix conventions before VAE/diffusion training.
    """
    q = ensure_finite("quaternions", quaternions)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion last dimension must be 4, got {q.shape}")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("quaternions contain a zero-norm entry")
    q = q / norms
    if quat_format == "xyzw":
        x, y, z, w = np.moveaxis(q, -1, 0)
    elif quat_format == "wxyz":
        w, x, y, z = np.moveaxis(q, -1, 0)
    else:
        raise ValueError(f"quat_format must be 'xyzw' or 'wxyz', got {quat_format!r}")

    rot = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[..., 0, 1] = 2.0 * (x * y - z * w)
    rot[..., 0, 2] = 2.0 * (x * z + y * w)
    rot[..., 1, 0] = 2.0 * (x * y + z * w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[..., 1, 2] = 2.0 * (y * z - x * w)
    rot[..., 2, 0] = 2.0 * (x * z - y * w)
    rot[..., 2, 1] = 2.0 * (y * z + x * w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def matrix_to_rot6d_array(rotations: np.ndarray) -> np.ndarray:
    """Flatten the first two rotation-matrix columns as paper Rot6D features."""
    rot = ensure_finite("rotations", rotations)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix trailing shape must be (3, 3), got {rot.shape}")
    return np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)


def yaw_from_matrix_array(rotations: np.ndarray) -> np.ndarray:
    """Return z-yaw angles from rotation matrices using ``atan2(R[1,0], R[0,0])``."""
    rot = ensure_finite("rotations", rotations)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix trailing shape must be (3, 3), got {rot.shape}")
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def build_paper_hybrid_state_window(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    root_lin_vel_w: np.ndarray,
    root_ang_vel_w: np.ndarray,
    body_pos_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    *,
    current_index: int,
    quat_format: str = "xyzw",
    schema: HybridStateSchema | None = None,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Build the paper's 99-D yaw-centric hybrid state for one trajectory window.

    Inputs are raw simulator/root/body world-frame arrays for a contiguous
    window.  The output terms match the paper-state debug artifact:
    root pose/twist expressed in the current frame, and target-body
    position/linear velocity expressed in each timestep's local root yaw frame.
    """
    schema = schema or hybrid_state_schema()
    root_pos = ensure_finite("root_pos_w", root_pos_w)
    root_quat = ensure_finite("root_quat_w", root_quat_w)
    root_lin_vel = ensure_finite("root_lin_vel_w", root_lin_vel_w)
    root_ang_vel = ensure_finite("root_ang_vel_w", root_ang_vel_w)
    body_pos = ensure_finite("body_pos_w", body_pos_w)
    body_lin_vel = ensure_finite("body_lin_vel_w", body_lin_vel_w)

    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"root_pos_w must have shape [T,3], got {root_pos.shape}")
    if root_quat.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_quat_w must have shape [T,4], got {root_quat.shape}")
    if root_lin_vel.shape != root_pos.shape or root_ang_vel.shape != root_pos.shape:
        raise ValueError(
            "root_lin_vel_w/root_ang_vel_w must match root_pos_w shape "
            f"{root_pos.shape}, got {root_lin_vel.shape}, {root_ang_vel.shape}"
        )
    if body_pos.ndim != 3 or body_pos.shape[0] != root_pos.shape[0] or body_pos.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T,B,3] with T={root_pos.shape[0]}, got {body_pos.shape}")
    if body_lin_vel.shape != body_pos.shape:
        raise ValueError(f"body_lin_vel_w must match body_pos_w shape {body_pos.shape}, got {body_lin_vel.shape}")
    if body_pos.shape[1] != schema.target_body_count:
        raise ValueError(f"schema target body count {schema.target_body_count} does not match body_pos_w {body_pos.shape}")
    if not 0 <= current_index < root_pos.shape[0]:
        raise ValueError(f"current_index must be in [0,{root_pos.shape[0] - 1}], got {current_index}")

    root_rot = quat_to_matrix_array(root_quat, quat_format=quat_format)
    root_yaw = yaw_from_matrix_array(root_rot)
    current_yaw_inv = _yaw_matrix_batch(np.array([-root_yaw[current_index]], dtype=np.float64))[0]

    root_pos_rel_current = (root_pos - root_pos[current_index]) @ current_yaw_inv.T
    root_rot6d_rel_current = matrix_to_rot6d_array(current_yaw_inv @ root_rot)
    root_lin_vel_rel_current = (root_lin_vel - root_lin_vel[current_index]) @ current_yaw_inv.T
    root_ang_vel_rel_current = root_ang_vel @ current_yaw_inv.T

    per_step_yaw_inv = _yaw_matrix_batch(-root_yaw)
    body_pos_local = np.einsum("tij,tbj->tbi", per_step_yaw_inv, body_pos - root_pos[:, None, :])
    body_lin_vel_local = np.einsum(
        "tij,tbj->tbi",
        per_step_yaw_inv,
        body_lin_vel - root_lin_vel[:, None, :],
    )

    slices = schema.slices
    state = np.concatenate(
        [
            root_pos_rel_current,
            root_rot6d_rel_current,
            root_lin_vel_rel_current,
            root_ang_vel_rel_current,
            body_pos_local.reshape(root_pos.shape[0], -1),
            body_lin_vel_local.reshape(root_pos.shape[0], -1),
        ],
        axis=-1,
    )
    return validate_hybrid_state(state, schema), slices


def emphasis_projection(
    seed: int = 7,
    state_dim: int = HYBRID_STATE_DIM,
    root_dim: int = ROOT_STATE_DIM,
    coefficient: int = EMPHASIS_COEFFICIENT,
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct paper yaw-centric state projection ``P`` and pseudoinverse.

    Shapes are ``P[(gaussian_rows+state_dim), state_dim]`` and
    ``P_inv[state_dim, (gaussian_rows+state_dim)]`` for 99-D yaw-centric
    trajectory state tokens.  The default root dimension is 15: 3-D relative
    root position, 6-D relative root orientation, 3-D relative linear velocity,
    and 3-D relative angular velocity.  The paper coefficient ``c=6`` scales
    the diagonal root-feature matrix ``B``; it is not the number of Gaussian
    rows in ``A``.
    """
    if min(state_dim, root_dim, coefficient, gaussian_rows) <= 0 or root_dim > state_dim:
        raise ValueError("state_dim/root_dim/coefficient/gaussian_rows must be positive with root_dim <= state_dim")
    rng = np.random.default_rng(seed)
    b = np.zeros((root_dim, state_dim), dtype=np.float64)
    b[:, :root_dim] = coefficient * np.eye(root_dim)
    a = rng.normal(size=(gaussian_rows, root_dim))
    p = np.vstack([a @ b, np.eye(state_dim)])
    return p, np.linalg.pinv(p)


def project_hybrid_state(
    states: np.ndarray,
    seed: int = 7,
    schema: HybridStateSchema | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project paper hybrid states with the emphasis projection matrix."""
    schema = schema or hybrid_state_schema()
    arr = validate_hybrid_state(states, schema)
    p, p_inv = emphasis_projection(
        seed=seed,
        state_dim=schema.state_dim,
        root_dim=schema.root_dim,
        coefficient=schema.coefficient,
        gaussian_rows=schema.gaussian_rows,
    )
    return arr @ p.T, p, p_inv


def unproject_hybrid_state(
    projected_states: np.ndarray,
    projection_inverse: np.ndarray,
    schema: HybridStateSchema | None = None,
) -> np.ndarray:
    """Recover 99-D hybrid states from projected-space predictions."""
    schema = schema or hybrid_state_schema()
    projected = ensure_finite("projected_states", projected_states)
    p_inv = ensure_finite("projection_inverse", projection_inverse)
    expected = schema.projected_dim
    if projected.shape[-1] != expected:
        raise ValueError(f"projected state last dim must be {expected}, got {projected.shape}")
    if p_inv.shape != (schema.state_dim, schema.projected_dim):
        raise ValueError(f"projection_inverse must have shape {(schema.state_dim, schema.projected_dim)}, got {p_inv.shape}")
    recovered = projected @ p_inv.T
    return validate_hybrid_state(recovered, schema)


def smoothness_penalty(path: np.ndarray) -> float:
    """Mean squared second-difference penalty for trajectory path ``[T,D]``."""
    path = ensure_finite("path", path)
    if path.ndim < 2 or path.shape[0] < 3:
        raise ValueError(f"path must have shape [T,D] with T>=3, got {path.shape}")
    second = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.mean(second**2))


__all__ = [
    "DEFAULT_TARGET_BODY_COUNT",
    "DEFAULT_GAUSSIAN_ROWS",
    "EMPHASIS_COEFFICIENT",
    "HYBRID_STATE_DIM",
    "HybridStateSchema",
    "ROOT_STATE_DIM",
    "TARGET_BODY_FEATURE_DIM",
    "build_paper_hybrid_state_window",
    "emphasis_projection",
    "hybrid_state_schema",
    "matrix_to_rot6d_array",
    "project_hybrid_state",
    "quat_to_matrix_array",
    "smoothness_penalty",
    "unproject_hybrid_state",
    "valid_contiguous_window_mask",
    "validate_hybrid_state",
    "yaw_from_matrix_array",
]
