from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from beyondmimic_repro.adapters.mujoco.sim_to_sim_smoke import run_mujoco_sim_to_sim_smoke


def test_mujoco_sim_to_sim_smoke(tmp_path) -> None:
    summary = run_mujoco_sim_to_sim_smoke(output=tmp_path / "mujoco_smoke.json", steps=8)
    assert summary["status"] == "passed"
    assert summary["mujoco_version"] == mujoco.__version__
