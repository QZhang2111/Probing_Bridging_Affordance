"""Shared PCA helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SubspaceModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def load(cls, path) -> "SubspaceModel":
        with np.load(path, allow_pickle=True) as data:
            mean = data["mean"].astype(np.float32)
            components = data["components"].astype(np.float32)
            eigenvalues = data["eigenvalues"].astype(np.float32)
        return cls(mean=mean, components=components, eigenvalues=eigenvalues)


def project_tokens(tokens_hwk: np.ndarray, model: SubspaceModel) -> np.ndarray:
    if tokens_hwk.ndim != 3:
        raise ValueError("tokens must be [H, W, C]")
    _, _, c = tokens_hwk.shape
    if model.components.shape[0] != c:
        raise ValueError("dimension mismatch")
    centered = tokens_hwk.reshape(-1, c) - model.mean.reshape(1, c)
    proj = centered @ model.components
    return proj.reshape(tokens_hwk.shape[0], tokens_hwk.shape[1], -1)


def apply_percentile_bounds(
    projections: np.ndarray,
    lows: Sequence[float],
    highs: Sequence[float],
) -> np.ndarray:
    if projections.ndim != 3:
        raise ValueError("projections must be [H, W, K]")
    norm = np.zeros_like(projections, dtype=np.float32)
    for idx in range(min(projections.shape[2], len(lows))):
        lo = float(lows[idx])
        hi = float(highs[idx])
        if hi <= lo:
            continue
        norm[..., idx] = np.clip((projections[..., idx] - lo) / (hi - lo), 0.0, 1.0)
    return norm


def upsample_components(
    norm_hwk: np.ndarray,
    *,
    output_size: Tuple[int, int],
) -> np.ndarray:
    tensor = torch.from_numpy(norm_hwk.transpose(2, 0, 1)).unsqueeze(0)
    up = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return up.squeeze(0).permute(1, 2, 0).numpy()
