#!/usr/bin/env python3
"""Visualise cosine similarity between the knife anchor patch and each model's feature grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"

from modules.config import ExperimentConfig
from modules.feature import ResizeMeta, restore_original_resolution
from modules.io import ensure_dir, load_tokens_npz, save_image_gray
from modules.pca import upsample_components
from modules.similarity import cosine_similarity, save_similarity_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pixel", type=int, nargs=2, metavar=("X", "Y"), default=None, help="Anchor pixel in original RGB space")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys to process")
    parser.add_argument("--model", type=str, default=None, help="快捷指定单模运行，等效于 --models <key>")
    parser.add_argument("--image", type=Path, default=None, help="Override knife RGB path for overlays")
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


def pixel_to_token_index(pixel_xy: Tuple[int, int], meta: ResizeMeta, patch_size: int, grid_hw: Tuple[int, int]) -> int:
    x, y = pixel_xy
    x_scaled = x * float(meta.scale) + meta.pad_left
    y_scaled = y * float(meta.scale) + meta.pad_top
    patch_x = int(np.clip(x_scaled / patch_size, 0, grid_hw[1] - 1))
    patch_y = int(np.clip(y_scaled / patch_size, 0, grid_hw[0] - 1))
    return patch_y * grid_hw[1] + patch_x


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_file(args.config)
    anchor_pixel = tuple(args.pixel) if args.pixel is not None else cfg.anchor_pixel
    image_path = Path(args.image) if args.image else cfg.images["knife"]

    selected_keys: Iterable[str] | None = None
    if args.model or args.models:
        queue: list[str] = []
        if args.model:
            queue.append(args.model)
        if args.models:
            queue.extend(args.models)
        selected_keys = queue
    specs = iter_models(cfg, selected_keys)
    for spec in specs:
        model_root = cfg.output_root / spec.key
        meta_file = model_root / "meta" / "knife.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Missing knife meta for model '{spec.key}': {meta_file}")
        with meta_file.open("r", encoding="utf-8") as fh:
            meta_info = json.load(fh)

        tokens_path = Path(meta_info["tokens_path"]).resolve()
        tokens, meta = load_tokens_npz(tokens_path)
        Hp = int(meta["H_patches"])
        Wp = int(meta["W_patches"])
        patch = int(meta["patch_size"])

        resize_meta = make_resize_meta(meta)
        anchor_idx = pixel_to_token_index(anchor_pixel, resize_meta, patch, (Hp, Wp))
        anchor = tokens[anchor_idx]

        anchors_dir = model_root / "anchors"
        ensure_dir(anchors_dir)
        np.save(anchors_dir / "knife_patch.npy", anchor.astype(np.float32))

        sims = cosine_similarity(anchor, tokens)
        sim_hw = sims.reshape(Hp, Wp)

        similarity_dir = model_root / "similarity"
        ensure_dir(similarity_dir)
        np.save(similarity_dir / "knife_patch_heatmap.npy", sim_hw.astype(np.float32))

        sim_lbox = upsample_components(sim_hw[..., None], output_size=(resize_meta.target_h, resize_meta.target_w))[..., 0]
        sim_final = restore_original_resolution(sim_lbox, resize_meta)
        save_image_gray(sim_final, similarity_dir / "knife_patch_gray.png")
        save_similarity_overlay(sim_final, image_path, similarity_dir / Path("knife_patch_overlay"), cfg.overlay)

        print(f"[similarity:{spec.key}] anchor {anchor_idx} processed")


if __name__ == "__main__":
    main()
