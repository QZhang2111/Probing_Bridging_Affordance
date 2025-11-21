from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from dino.pipeline.features import FeatureExtractor

from .roi_stage import ResizeMeta, restore_to_original


@dataclass
class DINOArtifacts:
    tokens: np.ndarray
    Hp: int
    Wp: int
    patch: int
    meta: ResizeMeta
    cache_hit: bool = False


def _cache_filename(cache_root: Path, image_path: Path, target_wh: tuple[int, int], patch_size: int) -> Path:
    name = f"{image_path.stem}_{target_wh[0]}x{target_wh[1]}_p{patch_size}.npz"
    return cache_root / name


def extract_dino_tokens(
    image_path: Path,
    target_wh=(1280, 960),
    patch_size: int = 16,
    cache_root: Optional[Path] = None,
) -> DINOArtifacts:
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = _cache_filename(cache_root, image_path, target_wh, patch_size)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            meta_dict = data["meta"].item()
            meta = ResizeMeta(**meta_dict)
            return DINOArtifacts(
                tokens=data["tokens"].astype(np.float32),
                Hp=int(data["Hp"]),
                Wp=int(data["Wp"]),
                patch=patch_size,
                meta=meta,
                cache_hit=True,
            )

    extractor = FeatureExtractor()
    tokens, Hp, Wp, meta = extractor.extract_image(image_path, target_wh, patch_size)

    if cache_root is not None:
        cache_path = _cache_filename(cache_root, image_path, target_wh, patch_size)
        np.savez_compressed(
            cache_path,
            tokens=tokens.astype(np.float32),
            Hp=int(Hp),
            Wp=int(Wp),
            meta=meta.as_dict(),
        )

    return DINOArtifacts(tokens=tokens, Hp=Hp, Wp=Wp, patch=patch_size, meta=meta, cache_hit=False)


def run_pca(
    artifacts: DINOArtifacts,
    roi_indices: np.ndarray,
    *,
    num_components: int = 3,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    tokens = artifacts.tokens.astype(np.float32)
    roi_tokens = tokens[roi_indices]
    if roi_tokens.shape[0] < 10:
        raise RuntimeError("ROI tokens too few for PCA.")

    mu = roi_tokens.mean(axis=0, keepdims=True)
    centered = tokens - mu
    roi_centered = roi_tokens - mu

    max_components = min(roi_tokens.shape[0], tokens.shape[1])
    n_components = int(max(1, min(num_components, max_components)))

    pca = PCA(n_components=n_components, svd_solver="randomized", iterated_power=5, random_state=random_state)
    scores_roi = pca.fit_transform(roi_centered)
    scores_all = centered @ pca.components_.T

    norm_all = np.zeros_like(scores_all, dtype=np.float32)
    bounds = []
    for idx in range(pca.components_.shape[0]):
        roi_vals = scores_roi[:, idx]
        lo = float(np.percentile(roi_vals, low_pct))
        hi = float(np.percentile(roi_vals, high_pct))
        bounds.append((lo, hi))
        if hi <= lo:
            continue
        vals = scores_all[:, idx]
        norm_all[:, idx] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

    hw_full = norm_all.reshape(artifacts.Hp, artifacts.Wp, -1)

    H_up = artifacts.Hp * artifacts.patch
    W_up = artifacts.Wp * artifacts.patch
    letterbox_full = np.zeros((H_up, W_up, n_components), dtype=np.float32)
    for ch in range(n_components):
        plane = hw_full[..., ch]
        plane_up = cv2.resize(plane, (W_up, H_up), interpolation=cv2.INTER_LINEAR)
        letterbox_full[..., ch] = plane_up

    orig_full = np.zeros((artifacts.meta.orig_h, artifacts.meta.orig_w, n_components), dtype=np.float32)
    for ch in range(n_components):
        orig_full[..., ch] = restore_to_original(letterbox_full[..., ch], artifacts.meta)

    letterbox_rgb = np.zeros((H_up, W_up, 3), dtype=np.float32)
    orig_rgb = np.zeros((artifacts.meta.orig_h, artifacts.meta.orig_w, 3), dtype=np.float32)
    for ch in range(min(3, n_components)):
        letterbox_rgb[..., ch] = letterbox_full[..., ch]
        orig_rgb[..., ch] = orig_full[..., ch]

    return {
        "hw_full": hw_full,
        "letterbox_full": letterbox_full,
        "orig_full": orig_full,
        "letterbox_rgb": letterbox_rgb,
        "orig_rgb": orig_rgb,
        "bounds": bounds,
        "pca_components": pca.components_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "num_components": n_components,
    }
