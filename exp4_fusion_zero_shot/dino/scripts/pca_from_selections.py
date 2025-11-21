#!/usr/bin/env python3
"""Fit PCA subspaces from cached ROI selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino.pipeline.common.fs import ensure_dir
from dino.pipeline.subspace import fit_weighted_pca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    cache_root = ROOT / "outputs" / "cache"
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=cache_root / "roi" / "selections" / "umd",
        help="Root directory containing ROI selection folders",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=cache_root / "roi" / "subspaces" / "umd",
        help="Destination directory for PCA outputs",
    )
    parser.add_argument("--components", type=int, default=10, help="Number of PCA components")
    return parser.parse_args()


def load_selection(selection_npz: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    with np.load(selection_npz, allow_pickle=True) as data:
        roi_tokens = data["roi_tokens"].astype(np.float32)
        weights = data.get("weights")
        if weights is not None:
            weights = weights.astype(np.float32)

        def _maybe_item(value):
            if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
                return value.item()
            return value

        meta = {
            "token_paths": [str(p) for p in data["token_paths"]],
            "mask_meta": _maybe_item(data.get("mask_meta")),
            "feat_meta": _maybe_item(data.get("feat_meta")),
            "image_path": str(data.get("image_path", "")),
        }
    return roi_tokens, weights, meta


def process_selection_dir(sel_dir: Path, selection_root: Path, out_root: Path, components: int) -> dict | None:
    selection_npz = sel_dir / "selection.npz"
    if not selection_npz.exists():
        return None

    roi_tokens, weights, meta = load_selection(selection_npz)
    if roi_tokens.shape[0] == 0:
        return None

    model = fit_weighted_pca(roi_tokens, weights, k=components)

    relative = sel_dir.relative_to(selection_root)
    out_dir = out_root / relative
    ensure_dir(out_dir)
    model_path = out_dir / f"pca_k{components}.npz"
    model.save(model_path)

    summary = {
        "selection": str(selection_npz),
        "pca_path": str(model_path),
        "roi_tokens": int(roi_tokens.shape[0]),
        "dimension": int(roi_tokens.shape[1]),
        "components": int(model.components.shape[1]),
        "explained_variance_ratio": model.variance_ratio.tolist(),
        "meta": meta,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    summaries = []
    for selection_npz in selection_root.rglob("selection.npz"):
        sel_dir = selection_npz.parent
        summary = process_selection_dir(sel_dir, selection_root, out_root, args.components)
        if summary:
            summaries.append(summary)

    if summaries:
        ensure_dir(out_root)
        summary_path = out_root / "summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump({"count": len(summaries), "entries": summaries}, fh, indent=2)


if __name__ == "__main__":
    main()
