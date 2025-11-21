#!/usr/bin/env python3
"""Generate an affordance overlay for a single UMD dataset RGB frame.

Given a path to an RGB frame and an affordance category (name or id),
the script looks up the corresponding ground-truth mask, darkens the
original image, overlays the affordance region in red, and saves the result.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from cv2 import error as Cv2Error
from scipy.io import loadmat

AFFORDANCE_ID_TO_NAME = {
    1: "grasp",
    2: "cut",
    3: "scoop",
    4: "contain",
    5: "pound",
    6: "support",
    7: "wrap-grasp",
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


NAME_TO_AFFORDANCE_ID = {
    _normalize_name(name): idx for idx, name in AFFORDANCE_ID_TO_NAME.items()
}


def parse_affordance(value: str) -> Tuple[int, str]:
    if value.isdigit():
        idx = int(value)
        if idx not in AFFORDANCE_ID_TO_NAME:
            raise ValueError(
                f"Affordance id {idx} is not in the valid range "
                f"{sorted(AFFORDANCE_ID_TO_NAME)}"
            )
        return idx, AFFORDANCE_ID_TO_NAME[idx]

    key = _normalize_name(value)
    if key not in NAME_TO_AFFORDANCE_ID:
        valid = ", ".join(AFFORDANCE_ID_TO_NAME.values())
        raise ValueError(
            f"Unknown affordance '{value}'. "
            f"Accepted names: {valid} or the corresponding ids."
        )
    idx = NAME_TO_AFFORDANCE_ID[key]
    return idx, AFFORDANCE_ID_TO_NAME[idx]


def find_label_file(image_path: Path) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    stem = image_path.stem
    if stem.endswith("_rgb"):
        stem = stem[:-4]
    label_candidates = [
        image_path.with_name(f"{stem}_label.mat"),
        image_path.with_name(f"{stem}_label_rank.mat"),
    ]
    for candidate in label_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a ground-truth label .mat file. "
        "Checked: " + ", ".join(str(p) for p in label_candidates)
    )


def build_output_path(image_path: Path, affordance_name: str, output: Optional[Path]) -> Path:
    image_path = image_path.resolve()
    safe_name = affordance_name.replace(" ", "-")
    suffix = image_path.suffix or ".jpg"
    filename = f"{image_path.stem}_{safe_name}_overlay{suffix}"

    if output is None:
        base_dir = Path(__file__).resolve().parent
        return base_dir / filename

    output = Path(output)
    if output.exists() and output.is_dir():
        return output / filename
    if output.suffix == "":
        return output / filename
    return output


def load_mask(label_path: Path) -> np.ndarray:
    mat = loadmat(label_path)
    if "gt_label" not in mat:
        raise KeyError(f"'gt_label' not found in {label_path}")
    mask = mat["gt_label"]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask from {label_path}, got shape {mask.shape}")
    return mask


def apply_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    affordance_id: int,
    darken_factor: float = 0.35,
    mask_alpha: float = 0.7,
) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask spatial dimensions do not match: "
            f"image {image.shape[:2]}, mask {mask.shape}"
        )

    affordance_mask = mask == affordance_id
    if not np.any(affordance_mask):
        raise ValueError(
            f"No pixels found for affordance id {affordance_id} in provided mask."
        )

    image_float = image.astype(np.float32)
    darkened = np.clip(image_float * darken_factor, 0, 255).astype(np.uint8)

    overlay = darkened.copy()
    mask_color = np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR
    overlay_region = overlay[affordance_mask]
    blended = (
        (1.0 - mask_alpha) * overlay_region.astype(np.float32)
        + mask_alpha * mask_color.astype(np.float32)
    )
    overlay[affordance_mask] = blended.astype(np.uint8)
    return overlay


def save_overlay(image: np.ndarray, output_path: Path) -> Path:
    try:
        ok = cv2.imwrite(str(output_path), image)
        if ok:
            return output_path
        raise RuntimeError("cv2.imwrite returned False")
    except (Cv2Error, RuntimeError) as exc:
        fallback = output_path.parent / f"{output_path.stem}.png"
        ok = cv2.imwrite(str(fallback), image)
        if not ok:
            raise RuntimeError(
                f"Failed to write overlay to {output_path} or fallback {fallback}"
            ) from exc
        print(
            f"Warning: saving overlay as {fallback.name} because "
            f"{output_path.suffix or 'the original format'} is unsupported."
        )
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an affordance overlay for a UMD dataset RGB frame."
    )
    parser.add_argument("image_path", type=Path, help="Path to a *_rgb.jpg frame.")
    parser.add_argument(
        "affordance",
        type=str,
        help="Affordance category (name or id). Names: "
        + ", ".join(f"{idx}:{name}" for idx, name in AFFORDANCE_ID_TO_NAME.items()),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file or directory for the overlay image (default: script folder).",
    )
    parser.add_argument(
        "--darken",
        type=float,
        default=0.55,
        help="Factor in (0,1] to darken the background image before overlay.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Blend factor for the red affordance mask (0=no red, 1=solid red).",
    )

    args = parser.parse_args()

    affordance_id, affordance_name = parse_affordance(args.affordance)

    label_path = find_label_file(args.image_path)
    mask = load_mask(label_path)

    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image_path}")

    overlay = apply_overlay(
        image,
        mask,
        affordance_id=affordance_id,
        darken_factor=args.darken,
        mask_alpha=args.alpha,
    )

    output_path = build_output_path(args.image_path, affordance_name, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_path = save_overlay(overlay, output_path)
    print(f"Overlay written to {saved_path}")


if __name__ == "__main__":
    main()
