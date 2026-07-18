from __future__ import annotations

from pathlib import Path

from beyondmimic_repro.adapters.mujoco.g1_pd_smoke import run_mujoco_g1_pd_smoke


def test_mujoco_g1_pd_smoke(tmp_path: Path) -> None:
    output = tmp_path / "g1_pd_smoke.json"
    summary = run_mujoco_g1_pd_smoke(output=output, steps=20)
    assert summary["status"] == "passed"
    assert summary["nu"] == 29
    assert summary["policy_joint_set_matches_mujoco"]
    assert len(summary["policy_to_mujoco_permutation"]) == 29
    assert output.is_file()
