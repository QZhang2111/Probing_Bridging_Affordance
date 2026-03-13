from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def run_kontext_generation(
    script_path: Path,
    model_dir: Path,
    image_path: Path,
    prompt: str,
    output_root: Path,
    num_steps: int,
    guidance: float,
    seed: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run `visualize_flux_kontext_cross_attention.py` and return key output paths.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(script_path),
        "--model_dir",
        str(model_dir),
        "--image_path",
        str(image_path),
        "--output_root",
        str(output_root),
        "--prompt",
        prompt,
        "--num_steps",
        str(num_steps),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
    ]
    if height is not None:
        cmd += ["--height", str(height)]
    if width is not None:
        cmd += ["--width", str(width)]
    if negative_prompt:
        cmd += ["--negative_prompt", negative_prompt]

    logging.info("Running Kontext generation: %s", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=script_path.parent.resolve())

    # After execution, locate directories
    latest_dirs = sorted(output_root.glob("*"))
    if not latest_dirs:
        raise RuntimeError(f"No output directory created in {output_root}")
    exp_dir = latest_dirs[-1]

    tokens_path = exp_dir / "tokens_t5.json"
    if not tokens_path.exists():
        raise FileNotFoundError(f"tokens_t5.json not found in {exp_dir}")
    with tokens_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)

    return {
        "exp_dir": exp_dir,
        "tokens_json": tokens_path,
        "tokens": tokens,
        "per_token_dir": exp_dir / "per_token",
        "generated_image": exp_dir / "gen.png",
    }
