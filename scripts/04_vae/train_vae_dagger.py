#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from beyondmimic_repro.cli.stage23 import main_train_vae_dagger

if __name__ == "__main__":
    raise SystemExit(main_train_vae_dagger())
