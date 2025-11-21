#!/usr/bin/env python3
"""
Precompute and cache DINO token features for a folder of images.

This script mirrors the caching strategy used in `pipeline/pca_stage.py`
but keeps a single `FeatureExtractor` instance alive, so running it once
avoids repeated model initialisation during downstream pipelines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1] / "dino"))

from pipeline.pca_stage import _cache_filename  # reuse naming convention
from dino.pipeline.features import FeatureExtractor


def find_images(root: Path, recursive: bool, exts: Iterable[str]) -> List[Path]:
    """
    Collect image files under `root`.
    """
    patterns = [f"*{ext}" for ext in exts]
    if recursive:
        images = sorted(
            {
                path
                for pattern in patterns
                for path in root.rglob(pattern)
                if path.is_file()
            }
        )
    else:
        images = sorted(
            {
                path
                for pattern in patterns
                for path in root.glob(pattern)
                if path.is_file()
            }
        )
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DINO feature cache for images.")
    parser.add_argument("image_root", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(__file__).resolve().parent / "cache",
        help="Directory to store cached npz files (default: zero_shot/cache).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp",
        help="Comma separated list of file extensions to include.",
    )
    parser.add_argument("--target-width", type=int, default=1280, help="Target width for DINO extractor.")
    parser.add_argument("--target-height", type=int, default=960, help="Target height for DINO extractor.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for DINO extractor.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if cached file exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process (useful for dry runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_root: Path = args.image_root.expanduser().resolve()
    cache_root: Path = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    extensions = [ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}" for ext in args.exts.split(",")]
    images = find_images(image_root, args.recursive, extensions)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print(f"[info] No images found under {image_root} matching extensions {extensions}")
        return

    target_wh: Tuple[int, int] = (args.target_width, args.target_height)
    extractor = FeatureExtractor()

    processed = 0
    skipped = 0

    try:
        iterator = tqdm(images, desc="Caching DINO tokens")
    except Exception:
        iterator = images  # fallback if tqdm unavailable

    for image_path in iterator:
        cache_path = _cache_filename(cache_root, image_path, target_wh, args.patch_size)

        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            tokens, Hp, Wp, meta = extractor.extract_image(image_path, target_wh, args.patch_size)
            np.savez_compressed(
                cache_path,
                tokens=tokens.astype(np.float32),
                Hp=int(Hp),
                Wp=int(Wp),
                meta=meta.as_dict(),
            )
            processed += 1
        except Exception as exc:
            print(f"[warn] Failed to cache {image_path}: {exc}")
            continue

    print(f"[done] Processed {processed} images, skipped {skipped} (cache hit). Cache stored in {cache_root}")


if __name__ == "__main__":
    main()
