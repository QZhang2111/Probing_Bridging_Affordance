"""Tensor utilities."""

from __future__ import annotations

from typing import Sequence

import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def to_tensor_norm(img) -> torch.Tensor:
    """Convert a PIL image to a normalized float tensor."""

    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def sweep_cuda() -> None:
    """Synchronize and clear CUDA caches (no-op on CPU-only setups)."""

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
