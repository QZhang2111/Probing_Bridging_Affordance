"""Backward-compatible wrappers around the shared pipeline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

from dino.pipeline.common.image import ResizeMeta, pick_target_by_orientation, resize_letterbox_to
from dino.pipeline.common.io import load_tokens_npz, normalise_npz, save_tokens_npz
from dino.pipeline.common.stats import percentile_stretch
from dino.pipeline.common.tensor import sweep_cuda, to_tensor_norm
from dino.pipeline.features import FeatureExtractor, extract_last_tokens, load_dinov3

# Export the constants and helpers used by existing scripts.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_NAME = FeatureExtractor()._model_name  # type: ignore[attr-defined]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ResizeMeta",
    "pick_target_by_orientation",
    "resize_letterbox_to",
    "load_dinov3",
    "extract_last_tokens",
    "FeatureExtractor",
    "save_tokens_npz",
    "load_tokens_npz",
    "normalise_npz",
    "percentile_stretch",
    "to_tensor_norm",
    "sweep_cuda",
]
