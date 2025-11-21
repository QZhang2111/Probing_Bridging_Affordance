#!/usr/bin/env python
"""Entry-point for running the linear probing experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.trainer import LinearProbeExperiment
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Single configuration file containing the full experiment setup.",
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="[Deprecated] Base config file to be merged with --local.",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=PROJECT_ROOT / "configs" / "local.yaml",
        help="[Optional] Local override config, merged into --defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None and args.config.exists():
        # Single-file configuration mode
        config = load_config(args.config, None)
    else:
        local_path = args.local if args.local.exists() else None
        config = load_config(args.defaults, local_path)
    experiment = LinearProbeExperiment(config)
    experiment.train()


if __name__ == "__main__":  # pragma: no cover
    main()
