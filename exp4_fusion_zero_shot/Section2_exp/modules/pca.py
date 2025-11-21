"""Weighted PCA utilities used by the Section2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SubspaceModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            components=self.components.astype(np.float32),
            eigenvalues=self.eigenvalues.astype(np.float32),
        )

    @classmethod
    def load(cls, path) -> "SubspaceModel":
        with np.load(path, allow_pickle=True) as data:
            mean = data["mean"].astype(np.float32)
            components = data["components"].astype(np.float32)
            eigenvalues = data["eigenvalues"].astype(np.float32)
        return cls(mean=mean, components=components, eigenvalues=eigenvalues)


def fit_weighted_pca(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None,
    k: int = 3,
    eps: float = 1e-8,
) -> SubspaceModel:
    if features.ndim != 2:
        raise ValueError("features must be [N, C]")
    if features.shape[0] == 0:
        raise ValueError("cannot fit PCA on empty features")

    X = features.astype(np.float32)
    n_samples, dim = X.shape

    if weights is not None:
        w = weights.reshape(-1).astype(np.float32)
        if w.shape[0] != n_samples:
            raise ValueError("weights length mismatch")
        weight_sum = float(w.sum())
        if weight_sum <= eps:
            raise ValueError("non-positive weights")
        w_norm = w / weight_sum
        mu = (X * w_norm[:, None]).sum(axis=0)
        centered = X - mu
        sqrt_w = np.sqrt(w_norm + eps, dtype=np.float32)[:, None]
        xw = centered * sqrt_w
    else:
        mu = X.mean(axis=0)
        centered = X - mu
        xw = centered

    cov = xw @ xw.T
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    comps = []
    sel_vals = []
    for idx in range(min(k, eigvals.shape[0])):
        lam = float(eigvals[idx])
        if lam <= eps:
            continue
        vec = eigvecs[:, idx]
        comp = (xw.T @ vec) / np.sqrt(lam)
        norm = float(np.linalg.norm(comp))
        if norm <= eps:
            continue
        comps.append((comp / norm).astype(np.float32))
        sel_vals.append(lam)

    if comps:
        components = np.stack(comps, axis=1).astype(np.float32)
        eigenvalues = np.asarray(sel_vals, dtype=np.float32)
    else:
        components = np.zeros((dim, 0), dtype=np.float32)
        eigenvalues = np.zeros((0,), dtype=np.float32)

    return SubspaceModel(mean=mu.astype(np.float32), components=components, eigenvalues=eigenvalues)


def project_tokens(tokens_hwk: np.ndarray, model: SubspaceModel) -> np.ndarray:
    if tokens_hwk.ndim != 3:
        raise ValueError("tokens must be [H, W, C]")
    H, W, C = tokens_hwk.shape
    if model.components.shape[0] != C:
        raise ValueError("dimension mismatch")
    centered = tokens_hwk.reshape(-1, C) - model.mean.reshape(1, C)
    proj = centered @ model.components
    return proj.reshape(H, W, -1)


def scale_by_percentiles(
    projections: np.ndarray,
    roi_indices: Optional[np.ndarray],
    low: float,
    high: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if projections.ndim != 3:
        raise ValueError("projections must be [H, W, K]")
    flat = projections.reshape(-1, projections.shape[2])
    if roi_indices is None or roi_indices.size == 0:
        source = flat
    else:
        source = flat[roi_indices]
    lows = np.zeros(projections.shape[2], dtype=np.float32)
    highs = np.zeros_like(lows)
    norm = np.zeros_like(projections, dtype=np.float32)
    for idx in range(projections.shape[2]):
        vals = source[:, idx]
        lo = float(np.percentile(vals, low))
        hi = float(np.percentile(vals, high))
        lows[idx] = lo
        highs[idx] = hi
        if hi <= lo:
            continue
        norm[..., idx] = np.clip((projections[..., idx] - lo) / (hi - lo), 0.0, 1.0)
    return norm, lows, highs


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


def embed_roi_tokens(norm_hwk: np.ndarray, roi_indices: np.ndarray) -> np.ndarray:
    out = np.zeros_like(norm_hwk, dtype=np.float32)
    if roi_indices.size == 0:
        return out
    flat_in = norm_hwk.reshape(-1, norm_hwk.shape[2])
    flat_out = out.reshape(-1, norm_hwk.shape[2])
    flat_out[roi_indices] = flat_in[roi_indices]
    return out


def upsample_components(
    norm_hwk: np.ndarray,
    *,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Upsample component maps to the RGB resolution."""

    tensor = torch.from_numpy(norm_hwk.transpose(2, 0, 1)).unsqueeze(0)
    up = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return up.squeeze(0).permute(1, 2, 0).numpy()
