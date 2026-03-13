"""IO helpers for token caches and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .fs import ensure_dir


TokenMeta = Dict[str, object]


def save_tokens_npz(tokens, meta: TokenMeta, path: Path | str) -> None:
    """Persist token tensors and grid metadata to a compressed NPZ."""

    np_path = Path(path)
    ensure_dir(np_path.parent)
    # store as float16 to match previous disk footprint
    arr = np.asarray(tokens, dtype=np.float16)
    np.savez_compressed(np_path, tokens_last=arr, grid_meta=meta)


def load_tokens_npz(npz_path: Path | str) -> tuple[np.ndarray, int, int, TokenMeta]:
    """Load cached tokens and return (tokens, H, W, metadata)."""

    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as data:
        tokens = data["tokens_last"].astype(np.float32)
        meta = data["grid_meta"].item()
    Hp = int(meta["H_patches"])
    Wp = int(meta["W_patches"])
    return tokens, Hp, Wp, meta


def normalise_npz(tokens_path: Path | str) -> Dict[str, Iterable]:
    """Load a cached .npz and coerce scalar arrays into Python types."""

    with np.load(tokens_path, allow_pickle=True) as data:
        return {k: data[k].item() if data[k].shape == () else data[k] for k in data}
