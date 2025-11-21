#!/usr/bin/env python3
"""Render quick overlays for the first UMD mask of each class."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino.pipeline.roi import save_mask_overlays_for_first_instances
from dino.src.settings import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    cache_root = settings.paths.get("cache_root")
    if cache_root is None:
        cache_root = ROOT / "outputs" / "cache"
    mask_root = cache_root / "masks" / "umd"
    out_root = cache_root / "roi" / "previews" / "umd"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, default=mask_root, help="Directory containing class-wise mask npz files")
    parser.add_argument("--out-root", type=Path, default=out_root, help="Directory to save overlay previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_mask_overlays_for_first_instances(
        args.mask_root.expanduser().resolve(),
        args.out_root.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
