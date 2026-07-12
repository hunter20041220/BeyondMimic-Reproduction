"""Emphasis projection persistence for Stage-3 state tokens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from beyondmimic_repro.state import HybridStateSchema, emphasis_projection, hybrid_state_schema
from beyondmimic_repro.validation import ensure_finite


def _schema_hash(schema: HybridStateSchema) -> str:
    raw = json.dumps(schema.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_projection_matrix(seed: int = 7, schema: HybridStateSchema | None = None) -> dict[str, np.ndarray | dict[str, object] | int | str]:
    """Build and return A, B, P, P_inv plus schema metadata."""
    schema = schema or hybrid_state_schema()
    p, p_inv = emphasis_projection(
        seed=seed,
        state_dim=schema.state_dim,
        root_dim=schema.root_dim,
        coefficient=schema.coefficient,
        gaussian_rows=schema.gaussian_rows,
    )
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(schema.gaussian_rows, schema.root_dim))
    b = np.zeros((schema.root_dim, schema.state_dim), dtype=np.float64)
    b[:, : schema.root_dim] = schema.coefficient * np.eye(schema.root_dim)
    return {
        "seed": seed,
        "A": a,
        "B": b,
        "P": p,
        "P_inv": p_inv,
        "schema": schema.to_dict(),
        "state_schema_hash": _schema_hash(schema),
    }


def apply_projection(states: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Apply P to states with last dimension D."""
    arr = ensure_finite("states", states)
    p = ensure_finite("projection", projection)
    if p.shape[1] != arr.shape[-1]:
        raise ValueError(f"projection input dim {p.shape[1]} != state dim {arr.shape[-1]}")
    return arr @ p.T


def apply_inverse_projection(projected: np.ndarray, projection_inverse: np.ndarray) -> np.ndarray:
    """Apply saved pseudoinverse P_inv."""
    arr = ensure_finite("projected", projected)
    p_inv = ensure_finite("projection_inverse", projection_inverse)
    if p_inv.shape[1] != arr.shape[-1]:
        raise ValueError(f"projection_inverse input dim {p_inv.shape[1]} != projected dim {arr.shape[-1]}")
    return arr @ p_inv.T


def save_projection(path: str | Path, projection_payload: dict[str, object]) -> None:
    """Persist a projection once; training/inference should reload it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in projection_payload.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in projection_payload.items() if not isinstance(value, np.ndarray)}
    np.savez_compressed(output, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))


def load_projection(path: str | Path) -> dict[str, object]:
    """Load a saved projection payload."""
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, object] = {key: data[key] for key in data.files if key != "metadata_json"}
        payload.update(json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {})
    return payload
