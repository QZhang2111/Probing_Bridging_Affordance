#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_heatmap_metrics.py
==========================

对齐后的 Kontext 热图与 AGD20K GT 掩码之间计算 LOCATE 中使用的 3 个指标：
  - mKLD
  - mSIM
  - mNSS

示例：
    python compare_heatmap_metrics.py \
        --heatmap kontext_outputs/.../mapped/heat_tok05_▁grip_on_original.png \
        --gt /home/li325/qing_workspace/dataset/AGD20K/AGD20K/Unseen/testset/GT/hold/axe/axe_000692.png

注意：
  * 建议使用灰度热图 (`*_on_original.png`)，不要直接用 overlay，否则会混入原图颜色。
  * 如果热图与 GT 尺寸不同，脚本会自动把热图缩放到 GT 大小。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img = img.astype(np.float32)
    return img


def normalize_to_255(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    arr_norm = (arr - mn) / (mx - mn)
    return arr_norm * 255.0


def resize_to(match: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if match.shape[:2] == (h, w):
        return match
    return cv2.resize(match, (w, h), interpolation=cv2.INTER_LINEAR)


def cal_kl(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    return float(np.sum(map2 * np.log(map2 / (map1 + eps) + eps)))


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)
    return float(np.sum(intersection))


def cal_nss(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    if std < eps:
        return 0.0
    smap = (pred - np.mean(pred)) / std
    if np.max(gt) - np.min(gt) < eps:
        return 0.0
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + eps)
    fixation_map = (fixation_map > 0.1).astype(np.float32)
    denom = fixation_map.sum()
    if denom < eps:
        return 0.0
    nss = np.sum(smap * fixation_map) / (denom + eps)
    return float(nss)


def main():
    parser = argparse.ArgumentParser(description="Compute mKLD/mSIM/mNSS between heatmap and GT")
    parser.add_argument("--heatmap", type=Path, required=True, help="对齐后的灰度热图路径")
    parser.add_argument("--gt", type=Path, required=True, help="GT 掩码路径")
    parser.add_argument("--save_resized_heatmap", type=Path, default=None, help="可选：保存缩放后的热图供检查")
    args = parser.parse_args()

    heat = load_grayscale(args.heatmap)
    gt = load_grayscale(args.gt)

    heat = resize_to(heat, gt.shape[:2])
    heat_norm = normalize_to_255(heat)

    gt_norm = gt.copy()
    if gt_norm.max() > 255.0:
        gt_norm = normalize_to_255(gt_norm)
    elif gt_norm.max() > 1.0:
        gt_norm = np.clip(gt_norm, 0, 255)
    else:
        gt_norm = gt_norm * 255.0

    if args.save_resized_heatmap:
        cv2.imwrite(str(args.save_resized_heatmap), np.clip(heat_norm, 0, 255).astype(np.uint8))

    mKLD = cal_kl(heat_norm, gt_norm)
    mSIM = cal_sim(heat_norm, gt_norm)
    mNSS = cal_nss(heat_norm, gt_norm)

    print(f"mKLD = {mKLD:.6f}")
    print(f"mSIM = {mSIM:.6f}")
    print(f"mNSS = {mNSS:.6f}")


if __name__ == "__main__":
    main()
