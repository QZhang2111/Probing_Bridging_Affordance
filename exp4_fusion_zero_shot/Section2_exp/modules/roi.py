"""ROI utilities tailored for the Section2 knife experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .feature import ResizeMeta


@dataclass
class ROISelection:
    indices: np.ndarray
    mask_hw: np.ndarray
    weights: Optional[np.ndarray] = None

    def fraction(self) -> float:
        return float(self.indices.size) / float(self.mask_hw.size)


def letterbox_mask(mask_path: Image.Image | str | bytes | "PathLike[str]", meta: ResizeMeta) -> np.ndarray:
    """Apply the same letterbox transform used for the RGB image to the mask."""

    with Image.open(mask_path) as mask_img:
        mask = mask_img.convert("L")
    resized = mask.resize((meta.resized_w, meta.resized_h), Image.NEAREST)
    canvas = Image.new("L", (meta.target_w, meta.target_h), 0)
    canvas.paste(resized, (meta.pad_left, meta.pad_top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def mask_from_alpha(image_path: Image.Image | str | bytes | "PathLike[str]", meta: ResizeMeta) -> np.ndarray:
    """Construct a mask from an image's alpha channel aligned with the resize metadata."""

    with Image.open(image_path) as img:
        if "A" not in img.getbands():
            return np.ones((meta.target_h, meta.target_w), dtype=np.float32)
        alpha = img.getchannel("A")

    resized = alpha.resize((meta.inner_w, meta.inner_h), Image.NEAREST)
    canvas = Image.new("L", (meta.target_w, meta.target_h), 0)
    canvas.paste(resized, (meta.pad_left, meta.pad_top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def mask_to_roi(
    mask_hw: np.ndarray,
    *,
    patch_size: int,
    threshold: float = 0.5,
    dilate_iters: int = 0,
) -> ROISelection:
    """Aggregate a binary mask into token indices and optional weights."""

    if mask_hw.ndim != 2:
        raise ValueError("mask must be HxW")
    H, W = mask_hw.shape
    if H % patch_size or W % patch_size:
        raise ValueError("mask dimensions must be divisible by patch size")

    grid_h = H // patch_size
    grid_w = W // patch_size
    mask_used = (mask_hw > 0.5).astype(np.float32)
    blocks = mask_used.reshape(grid_h, patch_size, grid_w, patch_size)
    ratios = blocks.mean(axis=(1, 3))

    token_mask = ratios >= threshold
    if dilate_iters > 0:
        token_mask = _dilate(token_mask.astype(np.uint8), iters=dilate_iters).astype(bool)

    indices = np.where(token_mask.reshape(-1))[0].astype(np.int64)
    weights = None
    token_mask_hw = token_mask.astype(np.uint8)
    return ROISelection(indices=indices, mask_hw=token_mask_hw, weights=weights)


def _dilate(mask_hw: np.ndarray, iters: int) -> np.ndarray:
    m = mask_hw.astype(np.uint8)
    for _ in range(max(0, iters)):
        padded = np.pad(m, ((1, 1), (1, 1)), mode="edge")
        neighbourhood = np.stack(
            [
                padded[:-2, :-2],
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, :-2],
                padded[1:-1, 1:-1],
                padded[1:-1, 2:],
                padded[2:, :-2],
                padded[2:, 1:-1],
                padded[2:, 2:],
            ],
            axis=0,
        )
        m = (neighbourhood.max(axis=0) > 0).astype(np.uint8)
    return m
