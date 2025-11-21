"""Utilities for visualizing ROI masks on top of images."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from ..common.fs import ensure_dir
from ..common.image import resize_letterbox_to
from dino.src.settings import get_settings

MASK_COLOR = (64, 255, 128)
MASK_ALPHA = 0.6
DEFAULT_PATCH = 16


def _load_rgb_path(entry) -> Path:
    if isinstance(entry, np.ndarray):
        if entry.dtype.type is np.bytes_ or entry.dtype.type is np.object_:
            entry = entry.item()
    return Path(str(entry))


def overlay_mask_on_image(
    image: Image.Image,
    mask_up: np.ndarray,
    color: Tuple[int, int, int] = MASK_COLOR,
    alpha: float = MASK_ALPHA,
) -> Image.Image:
    """Blend a binary mask onto an RGB image."""

    mask = np.clip(mask_up.astype(np.float32), 0.0, 1.0)
    color_arr = np.array(color, dtype=np.float32)
    rgb_np = np.asarray(image, dtype=np.float32)
    alpha_map = mask[..., None] * alpha
    blended = rgb_np * (1.0 - alpha_map) + color_arr * alpha_map
    blended = blended.clip(0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def render_mask_overlay_from_npz(
    mask_npz: Path,
    *,
    color: Tuple[int, int, int] = MASK_COLOR,
    alpha: float = MASK_ALPHA,
) -> Image.Image:
    """Load a cached mask npz and return an overlay image."""

    with np.load(mask_npz, allow_pickle=True) as data:
        mask_tokens = data["mask_tokens"].astype(np.uint8)
        target_w = int(np.squeeze(data.get("target_w", mask_tokens.shape[1] * DEFAULT_PATCH)))
        target_h = int(np.squeeze(data.get("target_h", mask_tokens.shape[0] * DEFAULT_PATCH)))
        patch = int(np.squeeze(data.get("patch_size", DEFAULT_PATCH)))
        rgb_entry = data.get("rgb_path")
        if rgb_entry is None:
            raise KeyError("mask npz missing 'rgb_path'")
        rgb_path = _load_rgb_path(rgb_entry)

    if not rgb_path.exists():
        settings = get_settings()
        data_root = settings.paths.get("data_root")
        if data_root is None:
            data_root = Path(__file__).resolve().parents[3] / "data"
        cls = mask_npz.parent.name
        stem = mask_npz.stem.split(".")[0]
        candidate = data_root / "UMD" / cls / f"{stem}_rgb.jpg"
        if candidate.exists():
            rgb_path = candidate
        else:
            raise FileNotFoundError(f"RGB image not found for mask {mask_npz}")

    with Image.open(rgb_path) as img:
        rgb = img.convert("RGB")

    if rgb.size != (target_w, target_h):
        rgb, _meta = resize_letterbox_to(rgb, (target_w, target_h), patch)

    mask_img = Image.fromarray((mask_tokens > 0).astype(np.uint8) * 255)
    mask_up = np.asarray(
        mask_img.resize((target_w, target_h), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0
    return overlay_mask_on_image(rgb, mask_up, color=color, alpha=alpha)


def save_mask_overlays_for_first_instances(
    mask_root: Path,
    out_root: Path,
    *,
    color: Tuple[int, int, int] = MASK_COLOR,
    alpha: float = MASK_ALPHA,
) -> None:
    """For each class directory, render the first mask overlay to ``out_root``."""

    for class_dir in sorted(mask_root.iterdir()):
        if not class_dir.is_dir():
            continue
        mask_files = sorted(class_dir.glob("*.npz"))
        if not mask_files:
            continue
        mask_path = mask_files[0]
        overlay = render_mask_overlay_from_npz(mask_path, color=color, alpha=alpha)
        out_dir = out_root / class_dir.name
        ensure_dir(out_dir)
        out_path = out_dir / f"{mask_path.stem}.png"
        overlay.save(out_path)
