#!/usr/bin/env python3
"""Generate ROI selections and PCA subspaces following run_single_roi_pca logic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Constants taken from run_single_roi_pca.py
LOW_PCT, HIGH_PCT = 1.0, 99.0
RAND_STATE = 0
ITER_POWER = 5
# Fit a 10-D PCA subspace, export the leading 3 dimensions for compatibility.
PCA_COMPONENTS_FULL = 10
PCA_COMPONENTS_EXPORT = 3
DILATE_ITERS = 1
ENABLE_DILATE = True


@dataclass
class SelectionSummary:
    class_name: str
    stem: str
    selection_path: Path
    pca_path: Path
    roi_tokens: int
    total_tokens: int
    explained_variance_ratio: Iterable[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "class": self.class_name,
            "stem": self.stem,
            "selection": str(self.selection_path),
            "pca_path": str(self.pca_path),
            "roi_tokens": self.roi_tokens,
            "total_tokens": self.total_tokens,
            "explained_variance_ratio": [float(x) for x in self.explained_variance_ratio],
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_tokens_npz(path: Path) -> tuple[np.ndarray, int, int, Dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        tokens = data["tokens_last"].astype(np.float32)
        meta = data["grid_meta"].item()
    Hp = int(meta["H_patches"])
    Wp = int(meta["W_patches"])
    return tokens, Hp, Wp, meta


def load_mask_npz(path: Path) -> tuple[np.ndarray, Dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        mask_tokens = data["mask_tokens"].astype(np.uint8)
        meta = {
            "H_patches": int(data.get("H_patches", mask_tokens.shape[0])),
            "W_patches": int(data.get("W_patches", mask_tokens.shape[1])),
            "target_w": int(data.get("target_w", 0)),
            "target_h": int(data.get("target_h", 0)),
            "rgb_path": str(data.get("rgb_path", "")),
            "label_path": str(data.get("label_path", "")),
        }
    return mask_tokens, meta


def dilate_token_mask(mask_hw: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask_hw
    m = mask_hw.astype(np.uint8)
    H, W = m.shape
    for _ in range(iters):
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


def find_first_mask(class_dir: Path) -> Optional[Path]:
    candidates = sorted(class_dir.glob("*_00000001*.fgmask.*.npz"))
    if candidates:
        return candidates[0]
    all_masks = sorted(class_dir.glob("*.fgmask.*.npz"))
    return all_masks[0] if all_masks else None


def find_tokens_path(tokens_root: Path, class_name: str, instance_id: str) -> Optional[Path]:
    pattern = f"{class_name}_{instance_id}.vit7b16.*.last.npz"
    matches = sorted(tokens_root.glob(pattern))
    return matches[0] if matches else None


def compute_percentiles(scores_roi: np.ndarray) -> list[Dict[str, float]]:
    percs = []
    for k in range(scores_roi.shape[1]):
        vals = scores_roi[:, k]
        lo = float(np.percentile(vals, LOW_PCT))
        hi = float(np.percentile(vals, HIGH_PCT))
        percs.append({"low": lo, "high": hi})
    return percs


def build_selection(
    class_name: str,
    mask_path: Path,
    mask_tokens: np.ndarray,
    tokens_path: Path,
    tokens_flat: np.ndarray,
    Hp: int,
    Wp: int,
    feat_meta: Dict[str, object],
    mask_meta: Dict[str, object],
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mask_flat = mask_tokens.reshape(-1).astype(bool)
    roi_indices_flat = np.where(mask_flat)[0]
    roi_indices_hw = np.column_stack(np.unravel_index(roi_indices_flat, (Hp, Wp)))
    roi_tokens = tokens_flat[roi_indices_flat]

    selection_npz = out_dir / "selection.npz"
    np.savez_compressed(
        selection_npz,
        token_paths=np.array([str(tokens_path)], dtype=object),
        token_indices=roi_indices_hw.astype(np.int16),
        token_indices_flat=roi_indices_flat.astype(np.int32),
        mask_tokens=mask_tokens.astype(np.uint8),
        roi_tokens=roi_tokens.astype(np.float32),
        feat_meta=np.array(feat_meta, dtype=object),
        mask_meta=np.array(
            {
                "source": str(mask_path),
                "rgb_path": mask_meta.get("rgb_path", ""),
                "target_w": mask_meta.get("target_w"),
                "target_h": mask_meta.get("target_h"),
            },
            dtype=object,
        ),
    )

    selection_meta = {
        "class": class_name,
        "mask_path": str(mask_path),
        "tokens_path": str(tokens_path),
        "roi_tokens": int(roi_tokens.shape[0]),
        "total_tokens": int(tokens_flat.shape[0]),
    }
    with (out_dir / "selection_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(selection_meta, fh, indent=2)

    return roi_tokens, roi_indices_flat, selection_meta


def run_pca(tokens_flat: np.ndarray, roi_mask_flat: np.ndarray) -> tuple[PCA, np.ndarray]:
    mu = tokens_flat[roi_mask_flat].mean(axis=0, keepdims=True)
    Xc_all = tokens_flat - mu
    Xc_roi = Xc_all[roi_mask_flat]

    pca = PCA(
        n_components=PCA_COMPONENTS_FULL,
        svd_solver="randomized",
        iterated_power=ITER_POWER,
        random_state=RAND_STATE,
    )
    scores_roi = pca.fit_transform(Xc_roi)
    return pca, scores_roi


def process_sample(
    class_name: str,
    mask_path: Path,
    tokens_path: Path,
    out_root: Path,
) -> Optional[SelectionSummary]:
    mask_tokens, mask_meta = load_mask_npz(mask_path)
    if ENABLE_DILATE and DILATE_ITERS > 0:
        mask_tokens = dilate_token_mask(mask_tokens, DILATE_ITERS)

    tokens_raw, Hp, Wp, feat_meta = load_tokens_npz(tokens_path)
    tokens_flat = tokens_raw.reshape(Hp * Wp, -1)

    out_dir = out_root / class_name / mask_path.stem.split(".fgmask")[0]
    ensure_dir(out_dir)

    roi_tokens, roi_indices_flat, _ = build_selection(
        class_name,
        mask_path,
        mask_tokens,
        tokens_path,
        tokens_flat,
        Hp,
        Wp,
        feat_meta,
        mask_meta,
        out_dir,
    )

    roi_mask_flat = np.zeros(tokens_flat.shape[0], dtype=bool)
    roi_mask_flat[roi_indices_flat] = True

    pca, scores_roi = run_pca(tokens_flat, roi_mask_flat)
    scores_export = scores_roi[:, :PCA_COMPONENTS_EXPORT]
    percentiles = compute_percentiles(scores_export)

    pca_full_path = out_dir / "pca_k10.npz"
    np.savez_compressed(
        pca_full_path,
        mu=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
    )

    pca_path = out_dir / "pca_k3.npz"
    np.savez_compressed(
        pca_path,
        mu=pca.mean_.astype(np.float32),
        components=pca.components_[:PCA_COMPONENTS_EXPORT].astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_[:PCA_COMPONENTS_EXPORT].astype(np.float32),
        singular_values=pca.singular_values_[:PCA_COMPONENTS_EXPORT].astype(np.float32),
        percentiles=np.array(percentiles, dtype=object),
    )

    meta = {
        "class": class_name,
        "stem": mask_path.stem.split(".fgmask")[0],
        "tokens_path": str(tokens_path),
        "mask_path": str(mask_path),
        "roi": {
            "n_roi": int(roi_tokens.shape[0]),
            "n_total": int(tokens_flat.shape[0]),
            "ratio": float(roi_tokens.shape[0] / float(tokens_flat.shape[0])),
            "dilate_iters": DILATE_ITERS,
        },
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "percentiles": {
            f"pc{i+1}": percentiles[i] for i in range(len(percentiles))
        },
        "solver": "randomized",
        "iterated_power": ITER_POWER,
        "random_state": RAND_STATE,
        "stretch_pct": [LOW_PCT, HIGH_PCT],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return SelectionSummary(
        class_name=class_name,
        stem=mask_path.stem.split(".fgmask")[0],
        selection_path=out_dir / "selection.npz",
        pca_path=pca_path,
        roi_tokens=roi_tokens.shape[0],
        total_tokens=tokens_flat.shape[0],
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def main() -> None:
    cache_root = ROOT / "outputs" / "cache"
    mask_root = cache_root / "masks" / "umd"
    tokens_root = cache_root / "tokens" / "umd"
    out_root = cache_root / "roi" / "selections"

    summaries: list[SelectionSummary] = []

    for class_dir in sorted(mask_root.iterdir()):
        if not class_dir.is_dir():
            continue
        mask_path = find_first_mask(class_dir)
        if mask_path is None:
            continue
        class_name = class_dir.name
        stem = mask_path.stem.split(".fgmask")[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        instance_id = parts[1]
        tokens_path = find_tokens_path(tokens_root, class_name, instance_id)
        if tokens_path is None:
            print(f"[skip] tokens not found for {class_name}_{instance_id}")
            continue
        summary = process_sample(class_name, mask_path, tokens_path, out_root)
        if summary:
            summaries.append(summary)

    if summaries:
        summary_out = out_root / "summary.json"
        ensure_dir(summary_out.parent)
        with summary_out.open("w", encoding="utf-8") as fh:
            json.dump({"count": len(summaries), "entries": [s.to_dict() for s in summaries]}, fh, indent=2)


if __name__ == "__main__":
    main()
