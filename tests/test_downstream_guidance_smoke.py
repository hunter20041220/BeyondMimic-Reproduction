from __future__ import annotations

import pytest

pytest.importorskip("torch")

from beyondmimic_repro.stage3.guidance.downstream_task_smoke import run_downstream_guidance_smoke


def test_downstream_guidance_smoke(tmp_path) -> None:
    summary = run_downstream_guidance_smoke(output=tmp_path / "downstream_guidance.json")
    assert summary["status"] == "passed"
    assert all(summary["checks"].values())
