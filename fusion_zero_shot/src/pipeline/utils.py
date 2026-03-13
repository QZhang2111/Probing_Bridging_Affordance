from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io_vis import (
    ensure_dir,
    sanitize_token,
    save_colormap,
    save_colormap_overlay,
    save_overlay,
)


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    """Run a subprocess and stream stdout/stderr."""

    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
