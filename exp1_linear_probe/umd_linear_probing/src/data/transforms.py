"""Image transformation presets for DINOv3 linear probing."""

from __future__ import annotations

from typing import Sequence

from torchvision import transforms as T

__all__ = ["get_default_image_transform", "get_inference_transform", "get_sam_image_transform"]


_DINOV3_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
_DINOV3_STD: Sequence[float] = (0.229, 0.224, 0.225)


def get_default_image_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=_DINOV3_MEAN, std=_DINOV3_STD),
        ]
    )


def get_inference_transform() -> T.Compose:
    return get_default_image_transform()


def get_sam_image_transform() -> T.Compose:
    """Transform for SAM backbones: only convert to tensor.

    SAMBackbone performs its own mean/std normalization internally to match
    the original SAM preprocessing. Therefore, we should avoid applying any
    dataset-level normalization here to prevent double normalization.
    """
    return T.Compose([T.ToTensor()])
