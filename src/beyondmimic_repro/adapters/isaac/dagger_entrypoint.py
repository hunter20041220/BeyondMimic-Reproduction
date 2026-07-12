"""Isaac DAgger runtime boundary.

The student controls the robot; the teacher labels student states.
"""

from __future__ import annotations

from beyondmimic_repro.adapters.isaac.contracts import IsaacRunConfig, launch_isaac_app


def run_collect_dagger_round(config: IsaacRunConfig) -> dict[str, object]:
    """Launch Isaac, then hand control to the runtime-specific collection code."""
    app = launch_isaac_app(config)
    try:
        # Isaac task imports must happen here on the 4090 host after AppLauncher.
        __import__("whole_body_tracking")
    finally:
        if hasattr(app, "close"):
            app.close()
    return {
        "status": "isaac_runtime_entrypoint_ready",
        "task_name": config.task_name,
        "validation": "not validated on H20; requires RTX 4090 + Isaac Sim runtime",
    }
