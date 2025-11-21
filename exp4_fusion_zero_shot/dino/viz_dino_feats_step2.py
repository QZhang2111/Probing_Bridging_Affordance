#!/usr/bin/env python3
"""Visualise similarity heatmaps from pre-extracted DINO feature npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dino.src.settings import get_settings

PATCH_SIZE = 16


def default_paths() -> tuple[Path | None, Path]:
    settings = get_settings()
    project_root = Path(__file__).resolve().parent
    image_root = settings.paths.get("data_root", project_root / "data")
    sample_image = next((p for p in image_root.rglob("*.jpg")), None) if image_root.exists() else None
    return sample_image, project_root


def parse_args() -> argparse.Namespace:
    sample_image, project_root = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=sample_image, help="Path to the RGB image used for feature extraction")
    parser.add_argument("--npz", type=Path, required=True, help="Path to the cached feature npz file")
    parser.add_argument("--seed", type=int, nargs=2, metavar=("X", "Y"), required=True, help="Seed pixel coordinates (x y) in the original image")
    parser.add_argument("--resize", type=int, default=896, help="Resize resolution used when extracting features")
    parser.add_argument("--output", type=Path, default=project_root / "dino_similarity_heatmap.png", help="Output image path")
    args = parser.parse_args()
    if args.image is None:
        parser.error("Image path is required; pass --image /path/to/image.jpg")
    return args


def main() -> None:
    args = parse_args()

    img_path = args.image.expanduser().resolve()
    npz_path = args.npz.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    H_orig, W_orig = img_bgr.shape[:2]
    x_resized = int(args.seed[0] / W_orig * args.resize)
    y_resized = int(args.seed[1] / H_orig * args.resize)
    grid_size = args.resize // PATCH_SIZE
    px = x_resized // PATCH_SIZE
    py = y_resized // PATCH_SIZE
    seed_idx = py * grid_size + px

    data = np.load(npz_path)
    patch_feats = data["patch_feats"]
    H, W, _ = int(data["H"]), int(data["W"]), int(data["dim"])

    seed_feat = patch_feats[seed_idx]
    sim = patch_feats @ seed_feat
    sim_map = sim.reshape(H, W)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].scatter([args.seed[0]], [args.seed[1]], c="red", marker="x", s=50)
    axes[0].set_title("Original with Seed")
    axes[0].axis("off")

    im = axes[1].imshow(sim_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("DINO Similarity")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
