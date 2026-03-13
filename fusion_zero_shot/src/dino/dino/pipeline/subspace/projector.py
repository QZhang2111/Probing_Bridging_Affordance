"""Utilities for projecting tokens into PCA subspaces and visualising them."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .trainer import SubspaceModel


def project_tokens(tokens_hwk: np.ndarray, model: SubspaceModel) -> np.ndarray:
    """Project tokens [H,W,C] onto subspace components -> [H,W,K]."""

    H, W, C = tokens_hwk.shape
    if model.components.shape[0] != C:
        raise ValueError("token dimension mismatch with subspace")
    centered = tokens_hwk.reshape(-1, C) - model.mean.reshape(1, C)
    proj = centered @ model.components
    return proj.reshape(H, W, -1)


def scale_by_percentiles(
    projections: np.ndarray,
    roi_indices: Optional[np.ndarray],
    low: float = 1.0,
    high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize projections using percentile stretch computed from ROI tokens."""

    H, W, K = projections.shape
    flat = projections.reshape(-1, K)
    if roi_indices is None or roi_indices.size == 0:
        source = flat
    else:
        source = flat[roi_indices]
    lows = np.zeros(K, dtype=np.float32)
    highs = np.zeros(K, dtype=np.float32)
    norm = np.zeros_like(projections, dtype=np.float32)
    for idx in range(K):
        vals = source[:, idx]
        lo = float(np.percentile(vals, low))
        hi = float(np.percentile(vals, high))
        lows[idx] = lo
        highs[idx] = hi
        if hi <= lo:
            continue
        norm[..., idx] = np.clip((projections[..., idx] - lo) / (hi - lo), 0.0, 1.0)
    return norm, lows, highs


def embed_roi_tokens(norm_full: np.ndarray, roi_indices: np.ndarray) -> np.ndarray:
    """Keep normalized values only on ROI indices, zero elsewhere."""

    out = np.zeros_like(norm_full)
    if roi_indices.size == 0:
        return out
    flat_full = norm_full.reshape(-1, norm_full.shape[2])
    flat_out = out.reshape(-1, norm_full.shape[2])
    flat_out[roi_indices] = flat_full[roi_indices]
    return out


def apply_percentile_bounds(
    projections: np.ndarray,
    lows: Sequence[float],
    highs: Sequence[float],
) -> np.ndarray:
    """Apply precomputed percentile bounds to new projections."""

    H, W, K = projections.shape
    norm = np.zeros_like(projections, dtype=np.float32)
    for idx in range(min(K, len(lows))):
        lo = lows[idx]
        hi = highs[idx]
        if hi <= lo:
            continue
        norm[..., idx] = np.clip((projections[..., idx] - lo) / (hi - lo), 0.0, 1.0)
    return norm


def rgb_from_components(
    norm_hwk: np.ndarray,
    *,
    output_size: Optional[Tuple[int, int]] = None,
    channel_order: Sequence[int] = (0, 1, 2),
) -> Image.Image:
    """Convert normalized components to an RGB PIL image."""

    H, W, K = norm_hwk.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for out_ch, comp_idx in enumerate(channel_order):
        if comp_idx >= K:
            continue
        rgb[..., out_ch] = norm_hwk[..., comp_idx]
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(rgb_u8, mode="RGB")
    if output_size is not None and img.size != output_size:
        img = img.resize(output_size, Image.BILINEAR)
    return img
