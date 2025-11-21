#!/usr/bin/env python3
"""Extract tokens for all configured models and images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"

from modules.config import ExperimentConfig, ModelSpec
from modules.feature import create_extractor
from modules.io import ensure_dir, save_tokens_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to settings.yaml")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of model keys to process (defaults to all configured models).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Convenience flag for processing单个模型，等价于传入 --models <key>。",
    )
    return parser.parse_args()


def iter_models(cfg: ExperimentConfig, selected: Iterable[str] | None) -> Iterable[ModelSpec]:
    if selected is None:
        return cfg.models.values()
    keys = list(selected)
    if not keys:
        return cfg.models.values()
    return [cfg.get_model(key) for key in keys]


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_file(args.config)

    selected_keys: list[str] | None = None
    if args.model or args.models:
        selected_keys = []
        if args.model:
            selected_keys.append(args.model)
        if args.models:
            selected_keys.extend(args.models)
    model_specs = iter_models(cfg, selected_keys)

    for spec in model_specs:
        extractor = create_extractor(spec)
        model_root = cfg.output_root / spec.key
        tokens_dir = model_root / "tokens"
        meta_dir = model_root / "meta"
        ensure_dir(tokens_dir)
        ensure_dir(meta_dir)

        images_map = dict(cfg.images)
        if spec.images_override:
            images_map.update(spec.images_override)

        for name, image_path in images_map.items():
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            target_override = None
            if (
                name == "knife"
                and spec.use_letterbox
                and spec.target_size[0] >= 640
                and spec.target_size[1] >= 480
            ):
                target_override = (640, 480)
            tokens, Hp, Wp, meta = extractor.extract_image(image_path, target_size=target_override)
            token_path = tokens_dir / f"{name}.{spec.key}.{meta.target_w}x{meta.target_h}.npz"
            grid_meta = {
                "H_patches": Hp,
                "W_patches": Wp,
                "patch_size": spec.patch_size,
                "resized_h": meta.final_h,
                "resized_w": meta.final_w,
                "model": spec.key,
                "model_type": spec.type,
                "preprocess": "normalise_meanstd",
                "source_path": str(image_path),
                "normalization_mean": spec.normalization.mean,
                "normalization_std": spec.normalization.std,
                **meta.as_dict(),
            }
            save_tokens_npz(
                token_path,
                tokens,
                grid_meta,
            )

            grid = {"H": Hp, "W": Wp, "patch": spec.patch_size}
            meta_json = {
                "model_key": spec.key,
                "model_type": spec.type,
                "tokens_path": str(token_path),
                "resize_meta": meta.as_dict(),
                "grid": grid,
                "normalization": {
                    "mean": spec.normalization.mean,
                    "std": spec.normalization.std,
                },
            }
            meta_path = meta_dir / f"{name}.json"
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta_json, fh, indent=2)

            print(f"[extract:{spec.key}] {name} -> {token_path}")


if __name__ == "__main__":
    main()
