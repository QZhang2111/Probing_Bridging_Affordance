#!/usr/bin/env python3
"""Run F1-style ROI PCA for the knife scene across all configured models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"

from modules.config import ExperimentConfig, ModelSpec
from modules.feature import ResizeMeta
from modules.io import ensure_dir, load_tokens_npz, save_image_colormap, save_image_rgb
from modules.pca import SubspaceModel, project_tokens
from modules.roi import ROISelection, letterbox_mask, mask_from_alpha, mask_to_roi


LOW_PCT = 1.0
HIGH_PCT = 99.0
RAND_STATE = 0
ITER_POWER = 5
MIN_ROI_TOKENS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mask", type=Path, default=None, help="Override knife mask path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Token coverage threshold for ROI selection")
    parser.add_argument("--dilate", type=int, default=1, help="Token-level dilation iterations")
    parser.add_argument("--models", nargs="*", default=None, help="Optional list of model keys to process")
    parser.add_argument("--model", type=str, default=None, help="快捷指定单模型，等价于 --models <key>")
    return parser.parse_args()


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


def iter_models(cfg: ExperimentConfig, selected: Iterable[str] | None):
    if selected is None:
        return cfg.models.values()
    keys = list(selected)
    if not keys:
        return cfg.models.values()
    return [cfg.get_model(k) for k in keys]


def percentile_stretch(values: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float, float]:
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        stretched = np.zeros_like(values, dtype=np.float32)
    else:
        stretched = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, lo, hi


def upsample_hwk(hw3: np.ndarray, H: int, W: int) -> np.ndarray:
    tensor = torch.from_numpy(hw3).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(tensor, size=(H, W), mode="bilinear", align_corners=False)
    return up.squeeze(0).permute(1, 2, 0).numpy()


def normalise_with_bounds(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def project_scene_into_subspace(
    scene: str,
    *,
    cfg: ExperimentConfig,
    spec: ModelSpec,
    subspace: SubspaceModel,
    percentiles: list[dict[str, float]],
    channel_names: list[str],
) -> None:
    meta_file = cfg.output_root / spec.key / "meta" / f"{scene}.json"
    if not meta_file.exists():
        print(f"[warn:{spec.key}] skipping scene '{scene}' (missing meta: {meta_file})")
        return
    with meta_file.open("r", encoding="utf-8") as fh:
        meta_info = json.load(fh)

    tokens_path = Path(meta_info["tokens_path"]).resolve()
    tokens, meta = load_tokens_npz(tokens_path)
    Hp = int(meta["H_patches"])
    Wp = int(meta["W_patches"])
    resize_meta = make_resize_meta(meta)

    tokens_hw = tokens.reshape(Hp, Wp, -1)
    projections = project_tokens(tokens_hw, subspace)

    channel_maps = []
    for idx in range(min(len(channel_names), projections.shape[2])):
        bounds = percentiles[idx]
        norm = normalise_with_bounds(projections[..., idx], bounds["low"], bounds["high"])
        channel_maps.append(norm)

    while len(channel_maps) < len(channel_names):
        channel_maps.append(np.zeros((Hp, Wp), dtype=np.float32))

    hw3 = np.stack(channel_maps[:3], axis=2)
    up_rgb = upsample_hwk(hw3, resize_meta.target_h, resize_meta.target_w)

    scene_dir = cfg.output_root / spec.key / "pca" / "knife" / scene
    ensure_dir(scene_dir)
    save_image_rgb(up_rgb, scene_dir / "pca_rgb_full.png")

    for idx, name in enumerate(channel_names):
        ch = channel_maps[idx]
        ch_up = upsample_hwk(np.repeat(ch[..., None], 3, axis=2), resize_meta.target_h, resize_meta.target_w)[..., 0]
        save_image_colormap(ch_up, scene_dir / f"{name}_full.png", cmap_name="viridis")


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_file(args.config)
    mask_override = Path(args.mask).resolve() if args.mask else None
    use_mask = bool(mask_override) or cfg.use_mask
    mask_path = mask_override if mask_override is not None else cfg.mask_path
    if use_mask and mask_path is None:
        raise ValueError("Mask usage enabled but no mask_path provided in config or CLI.")

    selected: Iterable[str] | None = None
    if args.model or args.models:
        queue: list[str] = []
        if args.model:
            queue.append(args.model)
        if args.models:
            queue.extend(args.models)
        selected = queue

    for spec in iter_models(cfg, selected):
        meta_file = cfg.output_root / spec.key / "meta" / "knife.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Missing meta file for model '{spec.key}': {meta_file}")
        with meta_file.open("r", encoding="utf-8") as fh:
            meta_info = json.load(fh)

        tokens_path = Path(meta_info["tokens_path"]).resolve()
        tokens, meta = load_tokens_npz(tokens_path)
        Hp = int(meta["H_patches"])
        Wp = int(meta["W_patches"])
        patch = int(meta["patch_size"])

        resize_meta = make_resize_meta(meta)
        if use_mask:
            mask_arr = letterbox_mask(mask_path, resize_meta)
            selection = mask_to_roi(mask_arr, patch_size=patch, threshold=args.threshold, dilate_iters=args.dilate)
            if selection.indices.size == 0:
                print(f"[warn:{spec.key}] ROI empty with threshold {args.threshold}; retry with threshold=0.0")
                selection = mask_to_roi(mask_arr, patch_size=patch, threshold=0.0, dilate_iters=max(args.dilate, 1))
            if selection.indices.size == 0:
                raise ValueError(f"ROI selection returned no tokens for model '{spec.key}'.")

            mask_flat = np.zeros(Hp * Wp, dtype=bool)
            mask_flat[selection.indices] = True
        else:
            source_path_str = meta.get("source_path")
            mask_arr = None
            if source_path_str:
                source_path = Path(source_path_str)
                if source_path.exists():
                    mask_arr = mask_from_alpha(source_path, resize_meta)
            if mask_arr is None:
                mask_arr = np.ones((resize_meta.target_h, resize_meta.target_w), dtype=np.float32)

            selection = mask_to_roi(mask_arr, patch_size=patch, threshold=args.threshold, dilate_iters=args.dilate)
            if selection.indices.size == 0:
                selection = mask_to_roi(mask_arr, patch_size=patch, threshold=0.0, dilate_iters=max(args.dilate, 1))
            if selection.indices.size == 0:
                selection = ROISelection(
                    indices=np.arange(Hp * Wp, dtype=np.int64),
                    mask_hw=np.ones((Hp, Wp), dtype=np.uint8),
                    weights=None,
                )
            mask_flat = np.zeros(Hp * Wp, dtype=bool)
            mask_flat[selection.indices] = True
        roi_tokens = tokens[mask_flat]
        if roi_tokens.shape[0] < MIN_ROI_TOKENS:
            print(f"[warn:{spec.key}] ROI tokens too few ({roi_tokens.shape[0]}), results may be noisy.")

        mu = roi_tokens.mean(axis=0, keepdims=True)
        centered_all = tokens - mu
        centered_roi = centered_all[mask_flat]

        pca = PCA(
            n_components=3,
            svd_solver="randomized",
            iterated_power=ITER_POWER,
            random_state=RAND_STATE,
        )
        scores_roi = pca.fit_transform(centered_roi)
        scores_all = centered_all @ pca.components_.T

        scaled_all = []
        scaled_roi = []
        percentiles = []
        for idx in range(min(3, pca.components_.shape[0])):
            roi_vals = scores_roi[:, idx]
            all_vals = scores_all[:, idx]
            roi_scaled, lo, hi = percentile_stretch(roi_vals, LOW_PCT, HIGH_PCT)
            if hi <= lo:
                all_scaled = np.zeros_like(all_vals, dtype=np.float32)
            else:
                all_scaled = np.clip((all_vals - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            scaled_all.append(all_scaled)
            scaled_roi.append(roi_scaled)
            percentiles.append({"low": lo, "high": hi})

        while len(scaled_all) < 3:
            scaled_all.append(np.zeros_like(scores_all[:, 0], dtype=np.float32))
            filler_roi = np.zeros(scores_roi.shape[0], dtype=np.float32)
            scaled_roi.append(filler_roi)
            percentiles.append({"low": 0.0, "high": 0.0})

        scaled_all = np.stack(scaled_all[:3], axis=1)
        scaled_roi = np.stack(scaled_roi[:3], axis=1)

        hw3_all = scaled_all.reshape(Hp, Wp, 3)
        hw3_roi = np.zeros_like(hw3_all, dtype=np.float32)
        hw3_roi.reshape(-1, 3)[mask_flat] = scaled_roi

        target_h = int(meta.get("resized_h", meta.get("final_h", resize_meta.final_h)))
        target_w = int(meta.get("resized_w", meta.get("final_w", resize_meta.final_w)))
        up_all = upsample_hwk(hw3_all, target_h, target_w)
        up_roi = upsample_hwk(hw3_roi, target_h, target_w)

        model_root = cfg.output_root / spec.key / "pca" / "knife"
        ensure_dir(model_root)

        save_image_rgb(up_roi, model_root / "pca_rgb_roi.png")
        save_image_rgb(up_all, model_root / "pca_rgb_full.png")

        channel_names = ["pc1", "pc2", "pc3"]
        for ch_idx, name in enumerate(channel_names):
            roi_ch = np.zeros((Hp, Wp), dtype=np.float32)
            roi_ch.reshape(-1)[mask_flat] = scaled_roi[:, ch_idx]
            roi_ch_up = upsample_hwk(np.repeat(roi_ch[..., None], 3, axis=2), target_h, target_w)[..., 0]
            save_image_colormap(roi_ch_up, model_root / f"{name}_roi.png", cmap_name="viridis")

        for ch_idx, name in enumerate(channel_names):
            full_ch = hw3_all[..., ch_idx]
            full_ch_up = upsample_hwk(np.repeat(full_ch[..., None], 3, axis=2), target_h, target_w)[..., 0]
            save_image_colormap(full_ch_up, model_root / f"{name}_full.png", cmap_name="viridis")

        subspace = SubspaceModel(
            mean=mu.squeeze(0).astype(np.float32),
            components=pca.components_.T.astype(np.float32),
            eigenvalues=pca.explained_variance_.astype(np.float32),
        )
        subspace_path = model_root / "knife_subspace.npz"
        subspace.save(subspace_path)
        perc_arr = np.stack(
            [
                [percentiles[idx]["low"] for idx in range(3)],
                [percentiles[idx]["high"] for idx in range(3)],
            ],
            axis=0,
        ).astype(np.float32)
        np.save(model_root / "knife_percentiles.npy", perc_arr)
        np.save(model_root / "knife_roi_indices.npy", selection.indices)

        meta_out = {
            "model_key": spec.key,
            "tokens_path": str(tokens_path),
            "mask_path": str(mask_path) if (use_mask and mask_path is not None) else None,
            "use_mask": bool(use_mask),
            "subspace_path": str(subspace_path),
            "grid": {"H_p": Hp, "W_p": Wp, "patch": patch},
            "target_size": {"W": target_w, "H": target_h},
            "roi": {
                "count": int(mask_flat.sum()),
                "total": int(tokens.shape[0]),
                "ratio": float(mask_flat.mean()),
                "dilate_iters": int(args.dilate),
                "threshold": float(args.threshold),
            },
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "percentiles": {name: percentiles[idx] for idx, name in enumerate(channel_names)},
            "solver": "randomized",
            "iterated_power": int(ITER_POWER),
            "stretch_pct": [float(LOW_PCT), float(HIGH_PCT)],
            "random_state": int(RAND_STATE),
        }
        with (model_root / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta_out, fh, indent=2)

        for scene in ("kitchen", "office"):
            project_scene_into_subspace(
                scene,
                cfg=cfg,
                spec=spec,
                subspace=subspace,
                percentiles=percentiles,
                channel_names=channel_names,
            )

        print(f"[pca:{spec.key}] knife ROI PCA exported to {model_root}")


if __name__ == "__main__":
    main()
