"""Helpers for building ROI selections from masks/ratio maps."""

from __future__ import annotations

import numpy as np

from .structures import ROI, ROISelection, SelectionStrategy
from .utils import compute_soft_weights, dilate_patch_mask


def selection_from_ratios(
    ratios: np.ndarray,
    *,
    name: str = "roi",
    threshold: float = 0.5,
    strategy: SelectionStrategy = SelectionStrategy.STRICT,
) -> ROISelection:
    """Produce an ROISelection given per-token coverage ratios."""

    ratios = ratios.astype(np.float32)
    token_mask = ratios >= threshold if strategy != SelectionStrategy.SOFT else ratios > 0
    weights_flat = None
    if strategy == SelectionStrategy.SOFT:
        weights_flat = compute_soft_weights(ratios)
    indices = np.where(token_mask.reshape(-1))[0]
    roi = ROI(name=name, token_indices=indices, weights=None if weights_flat is None else weights_flat[indices])
    selection = ROISelection(
        strategy=strategy,
        rois=[roi],
        token_mask=token_mask.astype(np.uint8),
        token_weights=None if weights_flat is None else weights_flat.reshape(ratios.shape),
    )
    return selection


def selection_from_token_mask(mask: np.ndarray, *, name: str = "roi") -> ROISelection:
    mask_u8 = mask.astype(np.uint8)
    indices = np.where(mask_u8.reshape(-1) > 0)[0]
    roi = ROI(name=name, token_indices=indices)
    return ROISelection(
        strategy=SelectionStrategy.STRICT,
        rois=[roi],
        token_mask=mask_u8,
    )


def selection_from_mask_tokens(
    mask_tokens: np.ndarray,
    *,
    name: str = "roi",
    dilate_iters: int = 0,
) -> ROISelection:
    mask = mask_tokens.astype(np.uint8)
    if dilate_iters > 0:
        mask = dilate_patch_mask(mask, iters=dilate_iters)
    return selection_from_token_mask(mask, name=name)
