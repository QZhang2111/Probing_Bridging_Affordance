"""Utilities for mapping pixel masks to token-level selections."""

from __future__ import annotations

import numpy as np


def mask_to_token_ratios(mask_up: np.ndarray, patch_size: int) -> np.ndarray:
    """Aggregate a high-res binary mask into per-token coverage ratios."""

    H, W = mask_up.shape
    if H % patch_size or W % patch_size:
        raise ValueError("mask resolution must be divisible by patch size")
    grid_h = H // patch_size
    grid_w = W // patch_size
    blocks = mask_up.reshape(grid_h, patch_size, grid_w, patch_size)
    ratios = blocks.sum(axis=(1, 3)).astype(np.float32) / float(patch_size * patch_size)
    return ratios


def ratios_to_indices(ratios: np.ndarray, threshold: float) -> np.ndarray:
    """Return indices of tokens whose coverage ratio exceeds the threshold."""

    mask = ratios >= threshold
    return np.where(mask.reshape(-1))[0]


def compute_soft_weights(ratios: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert ratio map to flattened soft weights normalized to sum=1."""

    flat = ratios.reshape(-1).astype(np.float32)
    total = flat.sum()
    if total <= eps:
        return np.zeros_like(flat)
    return flat / total


def dilate_patch_mask(mask_hw: np.ndarray, iters: int = 1) -> np.ndarray:
    """Apply binary dilation on a token-level mask."""

    m = mask_hw.astype(np.uint8)
    H, W = m.shape
    for _ in range(max(0, iters)):
        padded = np.pad(m, ((1, 1), (1, 1)), mode="edge")
        neigh = np.stack(
            [
                padded[0:H, 0:W],
                padded[0:H, 1:W + 1],
                padded[0:H, 2:W + 2],
                padded[1:H + 1, 0:W],
                padded[1:H + 1, 1:W + 1],
                padded[1:H + 1, 2:W + 2],
                padded[2:H + 2, 0:W],
                padded[2:H + 2, 1:W + 1],
                padded[2:H + 2, 2:W + 2],
            ],
            axis=0,
        )
        m = (neigh.max(axis=0) > 0).astype(np.uint8)
    return m
