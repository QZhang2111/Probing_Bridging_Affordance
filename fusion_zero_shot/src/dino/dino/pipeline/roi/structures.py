"""Data structures describing ROI selections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class SelectionStrategy(str, Enum):
    STRICT = "strict"          # binary mask
    RELAXED = "relaxed"        # area thresholding
    SOFT = "soft"              # continuous weights


@dataclass
class ROI:
    name: str
    token_indices: np.ndarray
    weights: Optional[np.ndarray] = None
    polygon_img: Optional[Sequence[tuple[float, float]]] = None

    def __post_init__(self) -> None:
        self.token_indices = np.asarray(self.token_indices, dtype=np.int64)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=np.float32)
            if self.weights.shape[0] != self.token_indices.shape[0]:
                raise ValueError("weights must match token_indices length")


@dataclass
class ROISelection:
    strategy: SelectionStrategy
    rois: List[ROI]
    token_mask: np.ndarray  # binary HxW map
    token_weights: Optional[np.ndarray] = None  # same shape as token_mask float32

    def flattened_indices(self) -> np.ndarray:
        return np.where(self.token_mask.reshape(-1) > 0)[0]


@dataclass
class ROIBatch:
    image_path: Path
    tokens_path: Path
    grid_size: tuple[int, int]
    patch_size: int
    selection: ROISelection
    metadata: dict = field(default_factory=dict)
