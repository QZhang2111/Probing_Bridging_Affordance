#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warp_heatmap_to_original.py
==========================

Map a FLUX Kontext token heatmap back to the original image coordinate system.
The script takes an (original image, edited image, heatmap) triplet and does
not require manual interaction or dense optical flow.

Pipeline
--------
1. Load original image, edited image, and heatmap; resize original if needed.
2. Detect ORB keypoints and build matches using BFMatcher + ratio test.
3. Estimate an affine transform with RANSAC (estimateAffinePartial2D).
4. Inverse-warp the heatmap to original resolution and save:
   - `*_heat_on_original.png`: grayscale heatmap aligned to original image
   - `*_overlay_on_original.png`: heatmap overlay on original image
   - `*_affine.npy`: 3x3 homogeneous affine matrix for reuse

Dependencies
------------
- OpenCV (`cv2`)
- NumPy

Example
-------
```
python warp_heatmap_to_original.py \
    --original axe_000692.jpg \
    --edited kontext_outputs/.../gen.png \
    --heatmap kontext_outputs/.../per_token/heat_tok05_▁grip.png \
    --out_dir kontext_outputs/.../mapped
```

The script creates `out_dir` automatically and names outputs based on the
heatmap filename stem.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_if_needed(src: np.ndarray, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
    """
    Resize src to target_shape (h, w).
    Returns (resized_image, sx, sy). If already same shape, no resize is done.
    """
    h_t, w_t = target_shape
    h_s, w_s = src.shape[:2]
    if (h_s, w_s) == (h_t, w_t):
        return src, 1.0, 1.0
    resized = cv2.resize(src, (w_t, h_t), interpolation=cv2.INTER_LINEAR)
    sx, sy = w_t / w_s, h_t / h_s
    return resized, sx, sy


def detect_and_match(
    orig: np.ndarray,
    edit: np.ndarray,
    ratio: float = 0.75,
    min_matches: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match keypoints using ORB + BFMatcher (Hamming) + ratio test.
    Returns matched point pairs as (N, 2).
    """
    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(orig, None)
    kp2, des2 = orb.detectAndCompute(edit, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Not enough keypoints detected (descriptor is None).")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_matches:
        raise RuntimeError(f"Too few valid matches: {len(good)} < {min_matches}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def estimate_affine(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    ransac_thresh: float = 3.0,
    min_inliers: int = 6,
) -> np.ndarray:
    """
    Estimate a 2x3 affine matrix with RANSAC.
    Raises if estimation fails.
    """
    M, inliers = cv2.estimateAffinePartial2D(
        pts_src,
        pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if M is None:
        raise RuntimeError("estimateAffinePartial2D failed to produce a valid affine matrix.")
    if inliers is not None and inliers.sum() < min_inliers:
        raise RuntimeError(f"Too few affine inliers: {int(inliers.sum())}")
    return M


def warp_heatmap_to_original(
    original_path: Path,
    edited_path: Path,
    heatmap_path: Path,
    out_dir: Path,
    alpha: float = 0.5,
) -> None:
    """
    Main routine: estimate affine transform and warp heatmap back to original image.
    """
    ensure_dir(out_dir)

    orig = load_image(original_path)
    edit = load_image(edited_path)
    heat = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if heat is None:
        raise FileNotFoundError(f"Failed to read heatmap: {heatmap_path}")

    # Resize original to edited-image size for keypoint matching only.
    edit_h, edit_w = edit.shape[:2]
    orig_resized, sx, sy = resize_if_needed(orig, (edit_h, edit_w))

    attempts = [
        {"ratio": 0.75, "min_matches": 10, "ransac_thresh": 3.0, "min_inliers": 6},
        {"ratio": 0.85, "min_matches": 6, "ransac_thresh": 5.0, "min_inliers": 3},
    ]

    M_resized = None
    success = False
    last_error = None
    for attempt in attempts:
        try:
            pts_orig, pts_edit = detect_and_match(
                orig_resized,
                edit,
                ratio=attempt["ratio"],
                min_matches=attempt["min_matches"],
            )
            M_resized = estimate_affine(
                pts_edit,
                pts_orig,
                ransac_thresh=attempt["ransac_thresh"],
                min_inliers=attempt["min_inliers"],
            )
            success = True
            break
        except RuntimeError as err:
            last_error = err

    fallback_identity = False
    if not success or M_resized is None:
        # Final fallback: resize heatmap directly without affine mapping.
        fallback_identity = True
        M_resized = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        print(f"[WARN] Affine estimation failed ({last_error}); using direct resize fallback.")

    # Convert affine transform back to original-image scale.
    S = np.array([[1.0 / sx, 0.0], [0.0, 1.0 / sy]], dtype=np.float32)
    t = np.array([[0.0], [0.0]], dtype=np.float32)
    M3 = np.vstack([M_resized, [0, 0, 1]])  # 3x3
    scale_mat = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float32)
    M3_full = scale_mat @ M3
    M = M3_full[:2, :]

    heat_norm = heat.astype(np.float32) / 255.0
    if fallback_identity:
        heat_on_orig = cv2.resize(
            heat_norm, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        M3_full = np.eye(3, dtype=np.float32)
    else:
        heat_on_orig = cv2.warpAffine(
            heat_norm,
            M,
            (orig.shape[1], orig.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    heat_uint8 = np.clip(heat_on_orig * 255.0, 0, 255).astype(np.uint8)

    # Generate overlay.
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1 - alpha, heat_color, alpha, 0.0)

    stem = heatmap_path.stem
    heat_out = out_dir / f"{stem}_on_original.png"
    overlay_out = out_dir / f"{stem}_overlay_on_original.png"
    affine_out = out_dir / f"{stem}_affine.npy"

    cv2.imwrite(str(heat_out), heat_uint8)
    cv2.imwrite(str(overlay_out), overlay)
    np.save(affine_out, M3_full)

    print(f"[OK] Saved warped heatmap to {heat_out}")
    print(f"[OK] Saved overlay to {overlay_out}")
    print(f"[OK] Saved affine matrix to {affine_out}")


def main():
    parser = argparse.ArgumentParser(description="Warp a Kontext heatmap back to original image coordinates.")
    parser.add_argument("--original", type=Path, required=True, help="Path to original image.")
    parser.add_argument("--edited", type=Path, required=True, help="Path to Kontext edited image.")
    parser.add_argument("--heatmap", type=Path, required=True, help="Path to grayscale heatmap.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay heatmap alpha in range [0, 1].")
    args = parser.parse_args()

    warp_heatmap_to_original(
        original_path=args.original,
        edited_path=args.edited,
        heatmap_path=args.heatmap,
        out_dir=args.out_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
