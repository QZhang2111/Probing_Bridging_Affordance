from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path


def warp_heatmap_cli(
    script_path: Path,
    original: Path,
    edited: Path,
    heatmap: Path,
    out_dir: Path,
    alpha: float = 0.5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    cmd = [
        "python",
        str(script_path),
        "--original",
        str(original),
        "--edited",
        str(edited),
        "--heatmap",
        str(heatmap),
        "--out_dir",
        str(out_dir),
        "--alpha",
        str(alpha),
    ]
    logging.info("Warp heatmap: %s", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=script_path.parent.resolve())
    warped = out_dir / (heatmap.stem + "_on_original.png")
    if not warped.exists():
        raise FileNotFoundError(f"Warped heatmap not found: {warped}")
    return warped
