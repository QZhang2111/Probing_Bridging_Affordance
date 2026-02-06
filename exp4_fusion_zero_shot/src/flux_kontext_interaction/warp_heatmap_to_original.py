#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warp_heatmap_to_original.py
==========================

自动将 FLUX Kontext 生成图上的 token 热图 (heatmap) 映射回原始图像坐标系。
适用于「原图 + 编辑图 + heatmap」三元组，不需要人工交互，也不使用稠密光流。

流程概述
--------
1. 读入原图、编辑图、heatmap，并在必要时把原图缩放到与编辑图同尺寸。
2. 使用 ORB 检测关键点，BFMatcher + 比例测试建立匹配。
3. 基于 RANSAC 估计仿射变换 (estimateAffinePartial2D)。
4. 将 heatmap 反向 warp 到原图尺寸，并输出：
   - `*_heat_on_original.png`：与原图对齐的灰度热图；
   - `*_overlay_on_original.png`：热图覆盖在原图上的可视化；
   - `*_affine.npy`：3×3 齐次仿射矩阵，便于后续复用。

依赖
----
- OpenCV (`cv2`)
- NumPy

用法示例
--------
```
python warp_heatmap_to_original.py \
    --original axe_000692.jpg \
    --edited kontext_outputs/.../gen.png \
    --heatmap kontext_outputs/.../per_token/heat_tok05_▁grip.png \
    --out_dir kontext_outputs/.../mapped
```

脚本会自动创建 `out_dir`，输出文件名以 heatmap 名称为基准。
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
    调整 src 到 target_shape (h, w)。返回 resized 图像以及缩放系数 (sx, sy)。
    如果已有相同尺寸则不缩放。
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
    使用 ORB + BFMatcher(汉明距离) + 比例测试，返回匹配的点对 (Nx2)。
    """
    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(orig, None)
    kp2, des2 = orb.detectAndCompute(edit, None)

    if des1 is None or des2 is None:
        raise RuntimeError("无法检测足够的关键点（descriptor 为 None）。")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_matches:
        raise RuntimeError(f"有效匹配点过少：{len(good)} < {min_matches}")

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
    估计仿射矩阵 (2x3)，基于 RANSAC。若失败抛出异常。
    """
    M, inliers = cv2.estimateAffinePartial2D(
        pts_src,
        pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if M is None:
        raise RuntimeError("estimateAffinePartial2D 未能找到有效仿射矩阵。")
    if inliers is not None and inliers.sum() < min_inliers:
        raise RuntimeError(f"仿射估计内点过少：{int(inliers.sum())}")
    return M


def warp_heatmap_to_original(
    original_path: Path,
    edited_path: Path,
    heatmap_path: Path,
    out_dir: Path,
    alpha: float = 0.5,
) -> None:
    """
    核心流程：估计仿射并将 heatmap warp 回原图。
    """
    ensure_dir(out_dir)

    orig = load_image(original_path)
    edit = load_image(edited_path)
    heat = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if heat is None:
        raise FileNotFoundError(f"Failed to read heatmap: {heatmap_path}")

    # resize 原图到编辑图尺寸（仅用于匹配）
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
        # 最终 fallback：仅 resize heatmap，不做仿射
        fallback_identity = True
        M_resized = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        print(f"[WARN] 仿射估计失败 ({last_error}); 使用直接 resize 作为 fallback。")

    # 把仿射转换到原始尺度
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

    # 生成 overlay
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
    parser = argparse.ArgumentParser(description="将 Kontext 热图 warp 回原图坐标系")
    parser.add_argument("--original", type=Path, required=True, help="原始图像路径")
    parser.add_argument("--edited", type=Path, required=True, help="Kontext 编辑后的图像路径")
    parser.add_argument("--heatmap", type=Path, required=True, help="热图（灰度）路径")
    parser.add_argument("--out_dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay 时热图占比 (0~1)")
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
