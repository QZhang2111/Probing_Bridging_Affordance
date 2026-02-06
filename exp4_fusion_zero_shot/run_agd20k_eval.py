#!/usr/bin/env python3
"""One-click AGD20K evaluation for Geometry × Interaction fusion.

This is a thin wrapper around:
  FLUX/AGD20K_Flux_unseen/run_flux_kontext_eval.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
SCRIPT = EXP_ROOT / "src" / "agd20k_eval" / "run_flux_kontext_eval.py"


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
