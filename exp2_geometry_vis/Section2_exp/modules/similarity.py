"""Cosine similarity helpers and visualisation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import cm

from .config import OverlayConfig
from .io import ensure_dir


def cosine_similarity(anchor: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    anchor = anchor.astype(np.float32)
    targets = targets.astype(np.float32)
    anchor_norm = float(np.linalg.norm(anchor) + eps)
    targets_norm = np.linalg.norm(targets, axis=1) + eps
    sims = (targets @ anchor) / (targets_norm * anchor_norm)
    return sims.astype(np.float32)


def save_similarity_overlay(
    heatmap_hw: np.ndarray,
    image_path: Path,
    out_prefix: Path,
    overlay: OverlayConfig,
) -> Path:
    ensure_dir(out_prefix.parent)
    np.save(out_prefix.with_suffix(".npy"), heatmap_hw.astype(np.float32))

    with Image.open(image_path) as img:
        base_rgb = img.convert("RGB")

    norm = _normalize_heatmap(heatmap_hw, overlay.low_pct, overlay.high_pct)
    cmap = cm.get_cmap(overlay.cmap)
    colored = (cmap(norm)[..., :3] * 255.0).astype(np.float32)

    base_resized = np.asarray(base_rgb.resize(colored.shape[1::-1], Image.BILINEAR), dtype=np.float32)
    blended = (overlay.alpha * colored + (1.0 - overlay.alpha) * base_resized).clip(0, 255).astype(np.uint8)
    out_path = out_prefix.with_suffix(".png")
    Image.fromarray(blended).save(out_path)
    return out_path


def _normalize_heatmap(data: np.ndarray, low: float, high: float) -> np.ndarray:
    lo = float(np.percentile(data, low))
    hi = float(np.percentile(data, high))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
