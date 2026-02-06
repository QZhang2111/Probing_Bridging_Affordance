from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union

from PIL import Image, ImageOps
from dino.pipeline.common.image import ResizeMeta


def resize_letterbox_to(img: Image.Image, target_wh: Tuple[int, int], patch_size: int) -> Tuple[Image.Image, ResizeMeta]:
    tw, th = target_wh
    ow, oh = img.size

    scale = min(tw / ow, th / oh)
    rw = max(int(ow * scale) // patch_size * patch_size, patch_size)
    rh = max(int(oh * scale) // patch_size * patch_size, patch_size)
    rw = min(rw, tw)
    rh = min(rh, th)

    img_resized = img.resize((rw, rh), Image.BICUBIC)

    pad_w = tw - rw
    pad_h = th - rh
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = ImageOps.expand(
        img_resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0,
    )
    meta = ResizeMeta(
        inner_w=rw,
        inner_h=rh,
        final_w=tw,
        final_h=th,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        scale=scale,
        orig_w=ow,
        orig_h=oh,
    )
    return img_padded, meta


def resize_with_letterbox(mask: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    resized = cv2.resize(mask, (meta.inner_w, meta.inner_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((meta.final_h, meta.final_w), dtype=np.float32)
    top, left = meta.pad_top, meta.pad_left
    canvas[top : top + meta.inner_h, left : left + meta.inner_w] = resized
    return canvas


def downsample_to_tokens(mask_letterbox: np.ndarray, Hp: int, Wp: int, patch: int) -> np.ndarray:
    if mask_letterbox.shape[0] != Hp * patch or mask_letterbox.shape[1] != Wp * patch:
        raise ValueError("letterbox mask size mismatch with token grid")
    reshaped = mask_letterbox.reshape(Hp, patch, Wp, patch)
    ratios = reshaped.mean(axis=(1, 3))
    return ratios


def restore_to_original(canvas: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    inner = canvas[meta.pad_top : meta.pad_top + meta.inner_h, meta.pad_left : meta.pad_left + meta.inner_w]
    restored = cv2.resize(inner, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)
    return restored


def compute_roi_tokens(letterbox_mask: np.ndarray, Hp: int, Wp: int, patch: int, threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    ratios = downsample_to_tokens(letterbox_mask, Hp, Wp, patch)
    token_mask = (ratios >= threshold).astype(np.uint8)
    indices = np.flatnonzero(token_mask.reshape(-1))
    if indices.size == 0:
        flat = ratios.reshape(-1)
        order = np.argsort(flat)[::-1]
        indices = order[: min(200, flat.size)]
        token_mask = np.zeros_like(flat, dtype=np.uint8)
        token_mask[indices] = 1
        token_mask = token_mask.reshape(Hp, Wp)
    return indices, token_mask


def build_roi_mask(
    object_heatmap: Union[Path, np.ndarray],
    meta: ResizeMeta,
    *,
    percentile: float = 85.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if isinstance(object_heatmap, (str, Path)):
        arr = cv2.imread(str(object_heatmap), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(object_heatmap)
        normalized = arr.astype(np.float32) / 255.0
    else:
        arr_np = np.asarray(object_heatmap, dtype=np.float32)
        if arr_np.ndim == 3:
            arr_np = cv2.cvtColor(arr_np, cv2.COLOR_RGB2GRAY)
        normalized = np.clip(arr_np, 0.0, 1.0)

    positive = normalized[normalized > 0.02]
    if positive.size == 0:
        thr = 0.2
    else:
        thr = max(0.2, float(np.percentile(positive, percentile)))
    mask = (normalized >= thr).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask >= 0.5).astype(np.float32)
    letterbox = resize_with_letterbox(mask, meta)
    info = {
        "mode": "percentile",
        "percentile": percentile,
        "threshold": thr,
    }
    return mask, letterbox, info
