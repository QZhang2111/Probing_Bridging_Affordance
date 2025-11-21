#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch PCA visualization from cached DINOv3 tokens (.npz)

Input:
  experiments/_cache/tokens/{umd,fun3du,kitchen}/*.npz
    - tokens_last: [N_patches, C] (float16 saved)
    - grid_meta: dict with at least:
        H_patches, W_patches, patch_size
        resized_h, resized_w           # final size after letterbox
        # optional (if present we will mask padding to black):
        pad_left, pad_right, pad_top, pad_bottom
        # optional historical keys:
        final_h/final_w, inner_h/inner_w

Output (per file) -> experiments/F1_PCA/runs/{umd|fun3du|kitchen}/{basename}/
  - pca_rgb.png
  - pca_meta.json

Notes:
  - PCA per image (no cross-image color alignment)
  - solver='randomized', n_iter=5 for speed & stability
  - percentile stretch (1%, 99%) per component to [0,1]
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

from dino.src.settings import get_settings


# ============ USER PATHS ============
SETTINGS = get_settings()
EXPERIMENTS_ROOT = SETTINGS.paths.get("experiments_root", Path(__file__).resolve().parents[2])
CACHE_ROOT = SETTINGS.paths.get("cache_root", EXPERIMENTS_ROOT / "_cache")
ROOT = str(EXPERIMENTS_ROOT.resolve())
TOKENS_ROOT = os.path.join(str(CACHE_ROOT.resolve()), "tokens")
OUT_ROOT = os.path.join(ROOT, "F1_PCA", "runs")
DATASETS    = ["umd", "fun3du", "kitchen"]   # processed in this order

# ============ VIS CONFIG ============
LOW_PCT, HIGH_PCT = 1.0, 99.0      # percentile stretch per component
SAVE_COMPONENTS   = True          # optionally save pc1/pc2/pc3 grayscale
SAVE_SCORES_NPZ   = True          # optionally save pca_scores.npz
RANDOM_STATE      = 0              # reproducible PCA
N_ITER_RANDOMIZED = 5              # for randomized SVD


# ============ UTILS ============
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def detect_dataset_from_path(path: str) -> str:
    path_low = path.lower()
    for ds in DATASETS:
        if f"/{ds}/" in path_low or path_low.endswith(f"/{ds}"):
            return ds
    # fallback to parent folder name
    return os.path.basename(os.path.dirname(path_low))

def load_npz(npz_path: str) -> Tuple[np.ndarray, Dict]:
    data = np.load(npz_path, allow_pickle=True)
    tokens = data["tokens_last"].astype(np.float32)  # [N,C]
    meta = data["grid_meta"].item()
    return tokens, meta

def get_final_size(meta: Dict) -> Tuple[int, int]:
    # prefer new keys (resized_*), fallback to final_* (older cache)
    H_final = int(meta.get("resized_h", meta.get("final_h", 0)))
    W_final = int(meta.get("resized_w", meta.get("final_w", 0)))
    if H_final == 0 or W_final == 0:
        raise ValueError("Missing resized_h/w (or final_h/w) in grid_meta")
    return H_final, W_final

def get_padding(meta: Dict) -> Tuple[int, int, int, int]:
    # return (top, bottom, left, right); zero if not present
    pt = int(meta.get("pad_top", 0))
    pb = int(meta.get("pad_bottom", 0))
    pl = int(meta.get("pad_left", 0))
    pr = int(meta.get("pad_right", 0))
    return pt, pb, pl, pr

def percentile_stretch(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    x: [N] array, returns scaled to [0,1] using robust percentiles.
    """
    lo = np.percentile(x, low)
    hi = np.percentile(x, high)
    if hi <= lo:
        # degenerate case; return zeros to be safe
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32), float(lo), float(hi)

def to_uint8_rgb(img01: np.ndarray) -> np.ndarray:
    """
    img01: [H,W,3] float in [0,1] -> uint8 RGB
    """
    arr = np.clip(img01, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr

def mask_letterbox(rgb: np.ndarray, pad: Tuple[int,int,int,int]) -> np.ndarray:
    """
    Zero out padding regions (top,bottom,left,right) on an [H,W,3] uint8 image.
    """
    H, W, _ = rgb.shape
    pt, pb, pl, pr = pad
    if pt > 0: rgb[:pt, :, :] = 0
    if pb > 0: rgb[H-pb:H, :, :] = 0
    if pl > 0: rgb[:, :pl, :] = 0
    if pr > 0: rgb[:, W-pr:W, :] = 0
    return rgb


# ============ CORE PCA ============
def pca_rgb_from_tokens(tokens: np.ndarray,
                        H_p: int, W_p: int,
                        H_final: int, W_final: int,
                        low_pct: float = LOW_PCT,
                        high_pct: float = HIGH_PCT,
                        save_scores: bool = False):
    """
    tokens: [N,C] float32
    returns:
      rgb_uint8: [H_final, W_final, 3]
      meta_pca: dict with explained_variance_ratio & percentiles
      (optional) scores_hw3: [H_p, W_p, 3] float32 in [0,1] after stretch
    """
    N, C = tokens.shape
    assert N == H_p * W_p, f"tokens N={N} != H_p*W_p={H_p*W_p}"

    # center (column-wise)
    X = tokens - tokens.mean(axis=0, keepdims=True)

    # PCA -> 3 comps
    pca = PCA(n_components=3,
              svd_solver="randomized",
              iterated_power=N_ITER_RANDOMIZED,
              random_state=RANDOM_STATE)
    scores = pca.fit_transform(X)   # [N,3], raw scores (not scaled to [0,1])

    # percentile stretch per component to [0,1]
    scaled = []
    percs  = []  # [(lo,hi) * 3]
    for k in range(3):
        sk = scores[:, k]
        y, lo, hi = percentile_stretch(sk, low_pct, high_pct)
        scaled.append(y)
        percs.append([lo, hi])
    scaled = np.stack(scaled, axis=1)  # [N,3] in [0,1]

    # reshape to [H_p,W_p,3]
    hw3 = scaled.reshape(H_p, W_p, 3)

    # upsample to final size using torch (bilinear)
    t = torch.from_numpy(hw3).permute(2,0,1).unsqueeze(0)   # [1,3,H_p,W_p]
    t_up = F.interpolate(t, size=(H_final, W_final), mode="bilinear", align_corners=False)
    up01 = t_up.squeeze(0).permute(1,2,0).numpy()           # [H_final,W_final,3] float [0,1]

    # meta for PCA
    meta_pca = {
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "percentiles": {
            "pc1": {"low": percs[0][0], "high": percs[0][1]},
            "pc2": {"low": percs[1][0], "high": percs[1][1]},
            "pc3": {"low": percs[2][0], "high": percs[2][1]},
        },
        "solver": "randomized",
        "iterated_power": int(N_ITER_RANDOMIZED),
        "low_pct": float(low_pct),
        "high_pct": float(high_pct)
    }

    if save_scores:
        return up01, meta_pca, hw3  # hw3 in [0,1]
    else:
        return up01, meta_pca, None


def process_one_npz(npz_path: str):
    tokens, meta = load_npz(npz_path)
    H_p = int(meta["H_patches"]);  W_p = int(meta["W_patches"])
    H_final, W_final = get_final_size(meta)
    pt, pb, pl, pr = get_padding(meta)

    dataset = detect_dataset_from_path(npz_path)
    base    = os.path.splitext(os.path.basename(npz_path))[0]
    out_dir = os.path.join(OUT_ROOT, dataset, base)
    ensure_dir(out_dir)

    # —— 只有在需要单通道或保存 scores 时才让函数返回 hw3 —— #
    need_scores = SAVE_COMPONENTS or SAVE_SCORES_NPZ
    up01, meta_pca, hw3 = pca_rgb_from_tokens(
        tokens=tokens,
        H_p=H_p, W_p=W_p,
        H_final=H_final, W_final=W_final,
        low_pct=LOW_PCT, high_pct=HIGH_PCT,
        save_scores=need_scores
    )

    # 主图
    rgb_u8 = to_uint8_rgb(up01)
    rgb_u8 = mask_letterbox(rgb_u8, (pt, pb, pl, pr))
    Image.fromarray(rgb_u8).save(os.path.join(out_dir, "pca_rgb.png"))

    # —— 单通道灰度：直接用已拉伸到 [0,1] 的 hw3[..., k] —— #
    if SAVE_COMPONENTS and hw3 is not None:
        for k, name in enumerate(["pc1", "pc2", "pc3"]):
            ch01 = hw3[..., k]  # [H_p, W_p], 已经在 [0,1]
            t = torch.from_numpy(ch01).unsqueeze(0).unsqueeze(0)  # [1,1,H_p,W_p]
            t_up = F.interpolate(t, size=(H_final, W_final), mode="bilinear", align_corners=False)
            up01 = t_up.squeeze(0).squeeze(0).numpy()  # [H_final, W_final]
            ch_u8 = to_uint8_rgb(np.stack([up01, up01, up01], axis=-1))
            ch_u8 = mask_letterbox(ch_u8, (pt, pb, pl, pr))
            Image.fromarray(ch_u8).save(os.path.join(out_dir, f"{name}.png"))

    # 可选：保存 scores（同样基于 hw3）
    if SAVE_SCORES_NPZ and hw3 is not None:
        np.savez_compressed(
            os.path.join(out_dir, "pca_scores.npz"),
            scores01=hw3.astype(np.float16),
            H_patches=H_p, W_patches=W_p
        )

    # 元信息
    meta_out = dict(
        source_path=meta.get("source_path",""),
        H_patches=H_p, W_patches=W_p, patch_size=int(meta.get("patch_size",16)),
        resized_h=H_final, resized_w=W_final,
        pad_top=pt, pad_bottom=pb, pad_left=pl, pad_right=pr,
        explained_variance_ratio=meta_pca["explained_variance_ratio"],
        percentiles=meta_pca["percentiles"],
        solver=meta_pca["solver"],
        iterated_power=meta_pca["iterated_power"],
        low_pct=meta_pca["low_pct"], high_pct=meta_pca["high_pct"]
    )
    with open(os.path.join(out_dir, "pca_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    return out_dir


def main():
    for ds in DATASETS:
        in_dir = os.path.join(TOKENS_ROOT, ds)
        if not os.path.isdir(in_dir):
            print(f"[warn] skip missing dir: {in_dir}")
            continue
        files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
        if not files:
            print(f"[warn] no npz in: {in_dir}")
            continue

        print(f"[start] {ds}: {len(files)} files")
        for fpath in tqdm(files, desc=ds, ncols=80):
            try:
                process_one_npz(fpath)
            except Exception as e:
                print(f"[ERR] {ds} failed: {fpath}\n  -> {repr(e)}")
        print(f"[done] {ds}")

if __name__ == "__main__":
    main()
