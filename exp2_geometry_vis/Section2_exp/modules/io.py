"""I/O helpers for token caches and visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib import cm
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tokens_npz(path: Path, tokens: np.ndarray, meta: Dict[str, object]) -> None:
    """Persist tokens using the legacy F1_PCA schema.

    Downstream tooling expects float16 ``tokens_last`` and a ``grid_meta``
    mapping.  Mirroring that structure keeps the modern multi模型配置兼容旧脚本。
    """

    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        tokens_last=tokens.astype(np.float16),
        grid_meta=meta,
    )


def load_tokens_npz(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load tokens produced by :func:`save_tokens_npz`.

    If an older Section2 cache (with ``tokens``/``meta`` keys) is encountered we
    gracefully convert it so mixed runs remain readable.
    """

    with np.load(path, allow_pickle=True) as data:
        if "tokens_last" in data and "grid_meta" in data:
            tokens = data["tokens_last"].astype(np.float32)
            meta_raw = data["grid_meta"]
        else:  # backward compatibility with the short-lived Section2 format
            tokens = data["tokens"].astype(np.float32)
            meta_raw = data["meta"]
        meta = meta_raw.item() if hasattr(meta_raw, "item") else dict(meta_raw)
    return tokens, meta


def save_image_rgb(array_hw3: np.ndarray, path: Path) -> None:
    array = np.clip(array_hw3, 0.0, 1.0)
    u8 = (array * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def save_image_gray(array_hw: np.ndarray, path: Path) -> None:
    array = np.clip(array_hw, 0.0, 1.0)
    u8 = (array * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def save_image_colormap(array_hw: np.ndarray, path: Path, cmap_name: str = "viridis") -> None:
    """Save a single-channel map using the requested matplotlib colormap."""

    array = np.clip(array_hw, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(array)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
