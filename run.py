#!/usr/bin/env python3
"""Unified launcher for Probing & Bridging affordance experiments.

Examples:
  python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
  python run.py geometry-eval -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
  python run.py interaction-probe -- --model-id /path/to/FLUX.1-Kontext-dev --image /path/to/img.png --prompt "hold toothbrush" --affordance hold
  python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "geometry-train": ROOT / "geometry_probing" / "train.py",
    "geometry-eval": ROOT / "geometry_probing" / "eval.py",
    "interaction-probe": ROOT / "interaction_probing" / "cross_attention_probe" / "cross_attention_probe.py",
    "fusion-eval": ROOT / "fusion_zero_shot" / "run_agd20k_eval.py",
    "aux-knife-sim": ROOT / "auxiliary_analysis" / "scripts" / "run_knife_patch_similarity.py",
    "aux-cross-sim": ROOT / "auxiliary_analysis" / "scripts" / "run_cross_domain_similarity.py",
    "aux-pca": ROOT / "auxiliary_analysis" / "scripts" / "run_pca_analysis.py",
    "aux-clip-probe": ROOT / "auxiliary_analysis" / "scripts" / "run_clip_patch_probe.py",
    # Backward-compatible aliases
    "exp1-train": ROOT / "geometry_probing" / "train.py",
    "exp1-eval": ROOT / "geometry_probing" / "eval.py",
    "exp3-probe": ROOT / "interaction_probing" / "cross_attention_probe" / "cross_attention_probe.py",
    "exp4-eval": ROOT / "fusion_zero_shot" / "run_agd20k_eval.py",
    "exp2-knife-sim": ROOT / "auxiliary_analysis" / "scripts" / "run_knife_patch_similarity.py",
    "exp2-cross-sim": ROOT / "auxiliary_analysis" / "scripts" / "run_cross_domain_similarity.py",
    "exp2-pca": ROOT / "auxiliary_analysis" / "scripts" / "run_pca_analysis.py",
    "exp2-clip-probe": ROOT / "auxiliary_analysis" / "scripts" / "run_clip_patch_probe.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the target script. Prefix with '--' to avoid argparse capture.",
    )
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    script = COMMANDS[parsed.command]
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    forward = parsed.args
    if forward and forward[0] == "--":
        forward = forward[1:]

    cmd = [sys.executable, str(script), *forward]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
