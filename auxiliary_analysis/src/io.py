"""I/O helpers re-exported from shared common utilities."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io_vis import ensure_dir, save_image_colormap, save_image_gray, save_image_rgb

import numpy as np
from typing import Dict, Tuple


def load_tokens_npz(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        if "tokens_last" in data and "grid_meta" in data:
            tokens = data["tokens_last"].astype(np.float32)
            meta_raw = data["grid_meta"]
        else:
            tokens = data["tokens"].astype(np.float32)
            meta_raw = data["meta"]
        meta = meta_raw.item() if hasattr(meta_raw, "item") else dict(meta_raw)
    return tokens, meta
