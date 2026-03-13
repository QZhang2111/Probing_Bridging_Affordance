"""Shared resize metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResizeMeta:
    orig_w: int
    orig_h: int
    target_w: int
    target_h: int
    inner_w: int
    inner_h: int
    final_w: int
    final_h: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "target_w": self.target_w,
            "target_h": self.target_h,
            "inner_w": self.inner_w,
            "inner_h": self.inner_h,
            "final_w": self.final_w,
            "final_h": self.final_h,
            "resized_w": self.final_w,
            "resized_h": self.final_h,
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
        }


def restore_original_resolution(array: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    squeeze_back = False
    if array.ndim == 2:
        array = array[..., None]
        squeeze_back = True

    top = max(meta.pad_top, 0)
    left = max(meta.pad_left, 0)
    bottom = min(top + meta.inner_h, array.shape[0])
    right = min(left + meta.inner_w, array.shape[1])
    cropped = array[top:bottom, left:right, :]
    if cropped.size == 0:
        raise ValueError("Cropped region is empty; verify ResizeMeta padding values.")

    tensor = torch.from_numpy(cropped.transpose(2, 0, 1)).unsqueeze(0).float()
    restored = F.interpolate(
        tensor,
        size=(meta.orig_h, meta.orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0).cpu().numpy()

    if squeeze_back:
        restored = restored[..., 0]
    return restored
