"""Subspace training utilities (weighted PCA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SubspaceModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    variance_ratio: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            mu=self.mean.astype(np.float32),
            U=self.components.astype(np.float32),
            evals=self.eigenvalues.astype(np.float32),
            var_ratio=self.variance_ratio.astype(np.float32),
        )

    @classmethod
    def load(cls, path) -> "SubspaceModel":
        data = np.load(path, allow_pickle=True)
        return cls(
            mean=data["mu"].astype(np.float32),
            components=data["U"].astype(np.float32),
            eigenvalues=data.get("evals", data.get("eigenvalues", np.array([]))).astype(np.float32),
            variance_ratio=data.get("var_ratio", np.array([])).astype(np.float32),
        )


def fit_weighted_pca(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None,
    k: int = 3,
    eps: float = 1e-8,
) -> SubspaceModel:
    """Compute a weighted PCA basis (k components) for the feature set."""

    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    if features.shape[0] == 0:
        raise ValueError("Cannot run PCA on empty feature set")

    X = features.astype(np.float32)
    n_samples, dim = X.shape

    if weights is not None:
        w = weights.astype(np.float32).reshape(-1)
        if w.shape[0] != n_samples:
            raise ValueError("weights length mismatch")
        weight_sum = float(w.sum())
        if weight_sum <= eps:
            raise ValueError("Non-positive weight sum")
        w_norm = w / weight_sum
        mu = (X * w_norm[:, None]).sum(axis=0)
        centered = X - mu
        sqrt_w = np.sqrt(w_norm, dtype=np.float32)[:, None]
        xw = centered * sqrt_w
    else:
        mu = X.mean(axis=0)
        centered = X - mu
        xw = centered

    cov = xw @ xw.T
    cov = (cov + cov.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
    positive = eigvals > eps
    if not np.any(positive):
        components = np.zeros((dim, 0), dtype=np.float32)
        eigvals_sel = np.zeros((0,), dtype=np.float32)
    else:
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k_eff = min(k, eigvals.shape[0])
        comps = []
        evals_sel = []
        for i in range(k_eff):
            lam = float(eigvals[i])
            if lam <= eps:
                continue
            vec = eigvecs[:, i]
            comp = (xw.T @ vec) / np.sqrt(lam)
            norm = float(np.linalg.norm(comp))
            if norm <= eps:
                continue
            comps.append(comp / norm)
            evals_sel.append(lam)
        if comps:
            components = np.stack(comps, axis=1).astype(np.float32)
            eigvals_sel = np.asarray(evals_sel, dtype=np.float32)
        else:
            components = np.zeros((dim, 0), dtype=np.float32)
            eigvals_sel = np.zeros((0,), dtype=np.float32)

    total_var = float(eigvals_sel.sum())
    if total_var > 0.0 and eigvals_sel.size:
        var_ratio = eigvals_sel / total_var
    else:
        var_ratio = np.zeros_like(eigvals_sel, dtype=np.float32)

    return SubspaceModel(
        mean=mu.astype(np.float32),
        components=components,
        eigenvalues=eigvals_sel,
        variance_ratio=var_ratio,
    )
