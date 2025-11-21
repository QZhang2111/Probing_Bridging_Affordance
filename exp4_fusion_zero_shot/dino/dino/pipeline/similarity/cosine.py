"""Cosine similarity helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity(anchor: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity between one anchor [C] and targets [N,C]."""

    anchor = anchor.astype(np.float32)
    targets = targets.astype(np.float32)
    anchor_norm = np.linalg.norm(anchor) + eps
    targets_norm = np.linalg.norm(targets, axis=1, keepdims=True) + eps
    sims = (targets @ anchor) / (targets_norm[:, 0] * anchor_norm)
    return sims.astype(np.float32)


def cosine_similarity_matrix(anchors: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity between anchors [M,C] and targets [N,C] -> [M,N]."""

    anchors = anchors.astype(np.float32)
    targets = targets.astype(np.float32)
    anchor_norms = np.linalg.norm(anchors, axis=1, keepdims=True) + eps
    target_norms = np.linalg.norm(targets, axis=1, keepdims=True) + eps
    sims = anchors @ targets.T
    sims = sims / (anchor_norms * target_norms.T)
    return sims.astype(np.float32)
