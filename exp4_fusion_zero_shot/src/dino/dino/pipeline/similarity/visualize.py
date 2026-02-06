"""Heatmap visualization helpers for similarity maps."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from matplotlib import cm


def normalize_heatmap(
    data: np.ndarray,
    mode: str = "percentile",
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    if mode == "minmax":
        mn = float(np.min(data))
        mx = float(np.max(data))
    else:
        mn = float(np.percentile(data, low))
        mx = float(np.percentile(data, high))
    if mx <= mn:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def save_overlay_heatmap(
    heatmap: np.ndarray,
    image: Image.Image,
    out_prefix: Path,
    *,
    cmap: str = "viridis",
    alpha: float = 0.6,
    normalize_mode: str = "percentile",
    low: float = 1.0,
    high: float = 99.0,
) -> Tuple[Path, Path]:
    npy_path = out_prefix.with_suffix(".npy")
    np.save(npy_path, heatmap.astype(np.float32))

    norm = normalize_heatmap(heatmap, mode=normalize_mode, low=low, high=high)
    cmap_obj = cm.get_cmap(cmap)
    colored = (cmap_obj(norm)[..., :3] * 255.0).astype(np.float32)
    base = np.asarray(image.resize(colored.shape[1::-1], Image.BILINEAR), dtype=np.float32)
    blended = (alpha * colored + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    out_path = out_prefix.with_suffix(".png")
    Image.fromarray(blended).save(out_path)
    return out_path, npy_path
