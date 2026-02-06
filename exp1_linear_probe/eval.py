#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
SCRIPT = EXP_ROOT / "umd_linear_probing" / "scripts" / "eval.py"


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
