#!/usr/bin/env python3
"""Compute cosine similarity overlays between knife anchor and other scenes."""

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
from src.io import ensure_dir, load_tokens_npz, save_image_gray
from src.pca import upsample_components
from src.similarity import cosine_similarity, save_similarity_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys")
    parser.add_argument("--model", type=str, default=None, help="Run one model (same as --models <key>)")
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
    if args.model or args.models:
        queue: list[str] = []
        if args.model:
            queue.append(args.model)
        if args.models:
            queue.extend(args.models)
        selected_keys = queue

    for spec in iter_models(cfg, selected_keys):
        model_root = cfg.output_root / spec.key
        anchor_path = model_root / "anchors" / "knife_patch.npy"
        if not anchor_path.exists():
            raise FileNotFoundError(f"Missing anchor for '{spec.key}'. Run run_knife_patch_similarity.py first.")
        anchor = np.load(anchor_path).astype(np.float32)

        for name in ("kitchen", "office"):
            meta_file = model_root / "meta" / f"{name}.json"
            if not meta_file.exists():
                raise FileNotFoundError(f"Missing {name} meta for '{spec.key}': {meta_file}")

            with meta_file.open("r", encoding="utf-8") as fh:
                meta_info = json.load(fh)

            tokens_path = Path(meta_info["tokens_path"]).resolve()
            tokens, meta = load_tokens_npz(tokens_path)
            hp = int(meta["H_patches"])
            wp = int(meta["W_patches"])
            sims = cosine_similarity(anchor, tokens)
            sim_hw = sims.reshape(hp, wp)

            sim_dir = model_root / "similarity"
            ensure_dir(sim_dir)
            np.save(sim_dir / f"{name}_cosine_heatmap.npy", sim_hw.astype(np.float32))

            resize_meta = make_resize_meta(meta)
            sim_lbox = upsample_components(sim_hw[..., None], output_size=(resize_meta.target_h, resize_meta.target_w))[..., 0]
            sim_orig = restore_original_resolution(sim_lbox, resize_meta)
            save_image_gray(sim_orig, sim_dir / f"{name}_cosine_gray.png")

            overlay_prefix = sim_dir / Path(f"{name}_cosine_overlay")
            image_path = cfg.images[name]
            save_similarity_overlay(sim_orig, image_path, overlay_prefix, cfg.overlay)
            print(f"[similarity:{spec.key}] {name} overlays saved")


if __name__ == "__main__":
    main()
