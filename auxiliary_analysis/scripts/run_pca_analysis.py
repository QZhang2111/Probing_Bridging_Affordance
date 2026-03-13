#!/usr/bin/env python3
"""Project secondary scenes into each model's knife ROI PCA subspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "defaults.yaml"

from src.config import ExperimentConfig
from src.resize import ResizeMeta, restore_original_resolution
from src.io import ensure_dir, load_tokens_npz, save_image_rgb
from src.pca import SubspaceModel, apply_percentile_bounds, project_tokens, upsample_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys")
    parser.add_argument("--model", type=str, default=None, help="Run one model (same as --models <key>)")
    parser.add_argument("model_keys", nargs="*", help="Positional shorthand for model keys")
    return parser.parse_args()


def iter_models(cfg: ExperimentConfig, selected: Iterable[str] | None):
    if selected is None:
        return cfg.models.values()
    keys = list(selected)
    if not keys:
        return cfg.models.values()
    return [cfg.get_model(k) for k in keys]


def make_resize_meta(meta: dict) -> ResizeMeta:
    inner_w = int(meta.get("inner_w", meta.get("resized_w", meta.get("final_w", meta["target_w"]))))
    inner_h = int(meta.get("inner_h", meta.get("resized_h", meta.get("final_h", meta["target_h"]))))
    final_w = int(meta.get("final_w", meta.get("resized_w", meta["target_w"])))
    final_h = int(meta.get("final_h", meta.get("resized_h", meta["target_h"])))
    return ResizeMeta(
        orig_w=int(meta["orig_w"]),
        orig_h=int(meta["orig_h"]),
        target_w=int(meta["target_w"]),
        target_h=int(meta["target_h"]),
        inner_w=inner_w,
        inner_h=inner_h,
        final_w=final_w,
        final_h=final_h,
        scale=float(meta.get("scale", 1.0)),
        pad_left=int(meta.get("pad_left", 0)),
        pad_top=int(meta.get("pad_top", 0)),
        pad_right=int(meta.get("pad_right", 0)),
        pad_bottom=int(meta.get("pad_bottom", 0)),
    )


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_file(args.config)
    selected_keys: Iterable[str] | None = None
    if args.model or args.models or args.model_keys:
        queue: list[str] = []
        if args.model:
            queue.append(args.model)
        if args.models:
            queue.extend(args.models)
        if args.model_keys:
            queue.extend(args.model_keys)
        selected_keys = queue
    specs = iter_models(cfg, selected_keys)

    for spec in specs:
        model_root = cfg.output_root / spec.key
        pca_dir = model_root / "pca" / "knife"
        subspace_path = pca_dir / "knife_subspace.npz"
        percentiles_path = pca_dir / "knife_percentiles.npy"

        if not subspace_path.exists() or not percentiles_path.exists():
            raise FileNotFoundError(f"Missing PCA artefacts for '{spec.key}'.")

        subspace = SubspaceModel.load(subspace_path)
        lows, highs = np.load(percentiles_path)

        for target in ("kitchen", "office"):
            meta_file = model_root / "meta" / f"{target}.json"
            if not meta_file.exists():
                raise FileNotFoundError(f"Missing {target} meta for '{spec.key}': {meta_file}")
            with meta_file.open("r", encoding="utf-8") as fh:
                meta_info = json.load(fh)

            tokens_path = Path(meta_info["tokens_path"]).resolve()
            tokens, meta = load_tokens_npz(tokens_path)
            hp = int(meta["H_patches"])
            wp = int(meta["W_patches"])
            tokens_hw = tokens.reshape(hp, wp, -1)
            projections = project_tokens(tokens_hw, subspace)
            norm = apply_percentile_bounds(projections, lows=lows, highs=highs)
            resize_meta = make_resize_meta(meta)
            rgb_lbox = upsample_components(norm[..., :3], output_size=(resize_meta.target_h, resize_meta.target_w))
            rgb = restore_original_resolution(rgb_lbox, resize_meta)

            out_path = pca_dir / f"{target}_in_knife_space.png"
            ensure_dir(out_path.parent)
            save_image_rgb(rgb, out_path)
            print(f"[pca:{spec.key}] {target} projection -> {out_path}")


if __name__ == "__main__":
    main()
