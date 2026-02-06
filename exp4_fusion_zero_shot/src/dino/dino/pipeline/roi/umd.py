"""UMD-specific ROI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from dino.src.settings import get_settings

from .utils import dilate_patch_mask

CACHE_SUBDIR = Path("masks") / "umd"
DEFAULT_PATCH = 16
GRID_W = 80
GRID_H = 60


def _mask_cache_root() -> Path:
    settings = get_settings()
    cache_root = settings.paths.get("cache_root")
    if cache_root is None:
        raise RuntimeError("cache_root not configured; set paths.cache_root in configs/local.yaml")
    return cache_root / CACHE_SUBDIR


def load_umd_affordance_mask(
    cls_name: str,
    stem: str,
    *,
    cache_root: Path | None = None,
    dilate_iters: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Load cached UMD foreground mask tokens and metadata.

    Returns (mask_tokens_hw, metadata_dict).
    """

    cache_dir = (cache_root if cache_root is not None else _mask_cache_root()) / cls_name
    mask_path = cache_dir / f"{stem}.fgmask.{GRID_W}x{GRID_H}.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask npz not found: {mask_path}")
    with np.load(mask_path, allow_pickle=True) as data:
        mask_tokens = data["mask_tokens"].astype(np.uint8)
        meta = {
            "H_patches": int(data.get("H_patches", GRID_H)),
            "W_patches": int(data.get("W_patches", GRID_W)),
            "target_w": int(data.get("target_w", GRID_W * DEFAULT_PATCH)),
            "target_h": int(data.get("target_h", GRID_H * DEFAULT_PATCH)),
            "rgb_path": str(data.get("rgb_path", "")),
            "label_path": str(data.get("label_path", "")),
        }
    if dilate_iters > 0:
        mask_tokens = dilate_patch_mask(mask_tokens, iters=dilate_iters)
    return mask_tokens.astype(np.uint8), meta
