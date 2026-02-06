from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    """Run a subprocess and stream stdout/stderr."""

    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_token(tok: str) -> str:
    t = tok.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return t[:24]


def save_overlay(image_path: Path, mask: np.ndarray, out_path: Path, color=(255, 0, 0), alpha: float = 0.5, darken: float = 0.65) -> None:
    base = Image.open(image_path).convert("RGB")
    base_np = np.asarray(base, dtype=np.float32)
    if darken > 0:
        base_np = np.clip(base_np * darken, 0.0, 255.0)
    base_dark = Image.fromarray(base_np.astype(np.uint8), mode="RGB")

    overlay_color = (255, 0, 0)
    overlay = Image.new("RGB", base.size, overlay_color)

    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.shape[:2] != (base.height, base.width):
        mask_arr = cv2.resize(mask_arr, (base.width, base.height), interpolation=cv2.INTER_LINEAR)
    mask_img = Image.fromarray((np.clip(mask_arr, 0, 1) * 255).astype(np.uint8), mode="L")
    composite = Image.composite(overlay, base_dark, mask_img)
    blended = Image.blend(base_dark, composite, alpha)
    blended.save(out_path)


def save_colormap(values: np.ndarray, out_path: Path) -> None:
    heat = (np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    Image.fromarray(colored).save(out_path)


def save_colormap_overlay(
    image_path: Path,
    values: np.ndarray,
    out_path: Path,
    *,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> None:
    """Blend a Viridis colormap heatmap with the original RGB image."""

    base = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    heat = np.clip(values, 0.0, 1.0)
    if heat.ndim == 3:
        heat = cv2.cvtColor(heat.astype(np.float32), cv2.COLOR_RGB2GRAY)
    heat_u8 = (heat * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(heat_u8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha_map = np.clip(heat[..., None] * alpha, 0.0, 1.0)
    blended = base * (1.0 - alpha_map) + colored * alpha_map
    Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)).save(out_path)
