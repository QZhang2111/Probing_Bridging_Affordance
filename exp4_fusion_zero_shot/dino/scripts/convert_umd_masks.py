#!/usr/bin/env python3
"""Convert UMD .mat label files to mask PNGs under the experiments data root."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.io import loadmat

DEFAULT_MAT_ROOT = Path(
    "/home/li325/qing_workspace/dataset/UMD/part-affordance-dataset/tools"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/li325/qing_workspace/exps/affordance-experiments/dino/data/UMD"
)

LABEL_KEYS: Iterable[str] = ("gt_label", "label", "Label", "gtLabel")


def find_label_array(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    for key in LABEL_KEYS:
        if key in data:
            array = np.asarray(data[key])
            if array.ndim >= 2:
                return array.squeeze()
    raise KeyError(f"No known label key found in {mat_path}")


def save_mask(label: np.ndarray, dest_path: Path, force: bool) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        return
    binary = (label > 0).astype(np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    image.save(dest_path)


def save_label_map(label: np.ndarray, dest_path: Path, force: bool) -> None:
    if dest_path.exists() and not force:
        return
    np.savez_compressed(dest_path, label=label.astype(np.uint8))


def convert(mat_root: Path, output_root: Path, force: bool) -> None:
    mat_paths = sorted(mat_root.rglob("*_label.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No *_label.mat files found under {mat_root}")

    for mat_path in mat_paths:
        label = find_label_array(mat_path)
        base = mat_path.stem.replace("_label", "")
        obj_name = base.split("_")[0]
        dest_dir = output_root / obj_name
        mask_path = dest_dir / f"{base}_mask.png"
        label_path = dest_dir / f"{base}_parts.npz"
        save_mask(label, mask_path, force)
        save_label_map(label, label_path, force)

    print(f"Converted {len(mat_paths)} label files to masks under {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat_root", type=Path, default=DEFAULT_MAT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing masks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert(args.mat_root, args.output_root, args.force)


if __name__ == "__main__":
    main()
