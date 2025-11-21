#!/usr/bin/env python3
"""Extract DINOv3 dense tokens for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from dino.src.feature_extraction import (
    extract_last_tokens,
    load_dinov3,
    resize_letterbox_to,
    save_tokens_npz,
    to_tensor_norm,
)
from dino.src.settings import get_settings

PATCH_SIZE = 16
MODEL_NAME = "dinov3_vit7b16"


def _default_output(image: Path) -> Path:
    settings = get_settings()
    cache_root = settings.paths.get("cache_root")
    if cache_root is None:
        raise RuntimeError("cache_root not configured; update configs/defaults.yaml")
    return cache_root / "tokens" / "manual" / (image.stem + ".vit7b16.{}x{}.last.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the RGB image to encode.")
    parser.add_argument("--output", type=Path, help="Destination .npz path (defaults under outputs/cache)")
    parser.add_argument("--width", type=int, default=1280, help="Target width (multiple of 16)")
    parser.add_argument("--height", type=int, default=960, help="Target height (multiple of 16)")
    parser.add_argument("--model", default=MODEL_NAME, help="Torch hub model name inside the Dinov3 repo")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.exists():
        raise FileNotFoundError(image)

    width, height = args.width, args.height
    if width % PATCH_SIZE or height % PATCH_SIZE:
        raise ValueError("Width/height must be multiples of the patch size (16)")

    output = args.output
    if output is None:
        template = _default_output(image)
        output = Path(str(template).format(width, height))
    output = output.expanduser().resolve()
    if output.exists() and not args.force:
        print(f"[skip] output exists: {output}")
        return

    model = load_dinov3(args.model)

    with Image.open(image) as img:
        img_rgb = img.convert("RGB")
    img_resized, meta = resize_letterbox_to(img_rgb, (width, height), PATCH_SIZE)
    tokens, Hp, Wp = extract_last_tokens(model, to_tensor_norm(img_resized))

    grid_meta = dict(
        H_patches=Hp,
        W_patches=Wp,
        patch_size=PATCH_SIZE,
        resized_h=meta.final_h,
        resized_w=meta.final_w,
        model=args.model,
        preprocess="imagenet_meanstd",
        source_path=str(image),
        **meta.as_dict(),
    )

    save_tokens_npz(tokens, grid_meta, output)
    print(f"[ok] saved tokens to {output} shape={tokens.shape}")


if __name__ == "__main__":
    main()
