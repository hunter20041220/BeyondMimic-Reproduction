#!/usr/bin/env python3
"""Summarize generated metrics files without committing heavy outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()
    root = Path(args.metrics_dir)
    files = sorted(root.glob("*.json"))
    summary = {}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"status": "invalid_json"}
        summary[path.name] = {
            "status": payload.get("status", "unknown") if isinstance(payload, dict) else "unknown",
            "keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
