#!/usr/bin/env python3
"""Check that the lightweight reproduction environment is usable."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys

import numpy as np

from beyondmimic_repro import __version__
from beyondmimic_repro.utils.io import write_json


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/metrics/environment_check.json")
    args = parser.parse_args()
    summary = {
        "status": "ok",
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch_available": module_available("torch"),
        "yaml_available": module_available("yaml"),
    }
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
