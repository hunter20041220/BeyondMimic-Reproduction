#!/usr/bin/env python3
"""Check whether tracking teachers are homogeneous enough to distill together.

Standalone teacher-homogeneity gate for multi-teacher distillation: parses each
teacher run's ``params/env.yaml`` snapshot (no IsaacSim needed, runs in
seconds) and compares the constants that must match across teachers in one
distillation:

    action scale            per-joint action scaling
    default joint angles    robot init_state joint_pos
    actuator gains          stiffness / damping per joint group
    policy obs terms        ordered observation term names

Any mismatch prints a per-key diff table and exits 1 (do NOT mix these
teachers — retrain the outlier under the unified config, or ensure every
collect/eval run uses --override_from_run with mutually homogeneous teachers).

Example:
    python scripts/00_setup/check_teacher_compat.py \
        --runs checkpoints/tracking/action1 checkpoints/tracking/action2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOL = 1e-9


def parse_float_block(block: str) -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(r"(\S+): (-?[\d.eE+-]+)", block)}


def parse_env_yaml(run_dir: Path) -> dict:
    """Extract comparable constants from a run's params/env.yaml via regex.

    The dump contains python-specific tags (!!python/object/apply), so it is
    not safe_load-able; the regexes target the IsaacLab env.yaml dump layout so
    the standalone check agrees with the in-loop distillation homogeneity gate.
    """
    path = run_dir / "params" / "env.yaml"
    if not path.exists():
        sys.exit(f"[FATAL] {path} not found — pass teacher run directories "
                 f"(each must contain params/env.yaml)")
    text = path.read_text()

    scale_m = re.search(
        r"\n  joint_pos:\n(?:.*\n)*?    scale:\n((?:      \S+: -?[\d.eE+-]+\n)+)", text)
    init_m = re.search(
        r"\n    init_state:\n(?:.*\n)*?      joint_pos:\n((?:        \S+: -?[\d.eE+-]+\n)+)", text)

    gains: dict[str, float] = {}
    for kind in ("stiffness", "damping"):
        for m in re.finditer(
                rf"\n        {kind}:\n((?:          \S+: -?[\d.eE+-]+\n)+)", text):
            for joint, value in parse_float_block(m.group(1)).items():
                gains[f"{kind}/{joint}"] = value

    obs_terms: list[str] = []
    policy_m = re.search(r"\nobservations:\n  policy:\n((?:    .*\n|      .*\n|        .*\n)+)", text)
    if policy_m:
        obs_terms = re.findall(r"^    ([a-z_]+):$", policy_m.group(1), re.MULTILINE)

    return {
        "action_scale": parse_float_block(scale_m.group(1)) if scale_m else {},
        "default_joint_pos": parse_float_block(init_m.group(1)) if init_m else {},
        "actuator_gains": gains,
        "policy_obs_terms": obs_terms,
    }


def diff_dicts(ref: dict[str, float], other: dict[str, float]) -> list[tuple[str, object, object]]:
    rows = []
    for key in sorted(set(ref) | set(other)):
        a, b = ref.get(key), other.get(key)
        if a is None or b is None or abs(a - b) > TOL:
            rows.append((key, a, b))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Teacher run directories (each containing params/env.yaml), "
                             "first one is the reference.")
    args = parser.parse_args()

    runs = [Path(r).expanduser() for r in args.runs]
    if len(runs) < 2:
        sys.exit("[FATAL] need at least two --runs to compare")

    parsed = {run.name or str(run): parse_env_yaml(run) for run in runs}
    names = list(parsed)
    ref_name, ref = names[0], parsed[names[0]]

    for field in ("action_scale", "default_joint_pos", "actuator_gains", "policy_obs_terms"):
        if not ref[field]:
            print(f"[WARN] could not parse {field} from reference run '{ref_name}'")

    compatible = True
    for name in names[1:]:
        other = parsed[name]

        if other["policy_obs_terms"] != ref["policy_obs_terms"]:
            compatible = False
            print(f"[DIFF] policy obs terms: {ref_name} vs {name}")
            print(f"       {ref_name}: {ref['policy_obs_terms']}")
            print(f"       {name}: {other['policy_obs_terms']}")

        for field in ("action_scale", "default_joint_pos", "actuator_gains"):
            rows = diff_dicts(ref[field], other[field])
            if rows:
                compatible = False
                print(f"[DIFF] {field}: {ref_name} vs {name}")
                for key, a, b in rows:
                    print(f"       {key:45s} {a!s:>22s}  {b!s:>22s}")

    n_scale = len(ref["action_scale"])
    n_joints = len(ref["default_joint_pos"])
    n_gains = len(ref["actuator_gains"])
    n_obs = len(ref["policy_obs_terms"])
    print(f"[INFO] compared {len(names)} runs against '{ref_name}' "
          f"({n_scale} scale keys, {n_joints} default-joint keys, "
          f"{n_gains} gain keys, {n_obs} policy obs terms)")

    if compatible:
        print(f"[OK] teachers are homogeneous — safe to distill together: {', '.join(names)}")
    else:
        print("[FAIL] teachers are NOT homogeneous — do not mix in one distillation. "
              "Retrain the outlier under the unified robot config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
