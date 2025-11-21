"""Image preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageOps


@dataclass
class ResizeMeta:
    inner_w: int
    inner_h: int
    final_w: int
    final_h: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int
    scale: float
    orig_w: int
    orig_h: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "inner_w": self.inner_w,
            "inner_h": self.inner_h,
            "final_w": self.final_w,
            "final_h": self.final_h,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "scale": self.scale,
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
        }


def round_to_multiple(x: int, k: int, mode: str = "floor") -> int:
    if mode == "floor":
        return (x // k) * k
    if mode == "ceil":
        return ((x + k - 1) // k) * k
    raise ValueError("mode must be 'floor' or 'ceil'")


def resize_letterbox_to(
    img: Image.Image,
    target_wh: Tuple[int, int],
    patch_size: int,
) -> tuple[Image.Image, ResizeMeta]:
    tw, th = target_wh
    ow, oh = img.size

    scale = min(tw / ow, th / oh)
    rw = max(round_to_multiple(int(ow * scale), patch_size, "floor"), patch_size)
    rh = max(round_to_multiple(int(oh * scale), patch_size, "floor"), patch_size)
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


def pick_target_by_orientation(
    img: Image.Image,
    land_wh: Tuple[int, int],
    port_wh: Tuple[int, int],
) -> Tuple[int, int]:
    w, h = img.size
    return land_wh if w >= h else port_wh
