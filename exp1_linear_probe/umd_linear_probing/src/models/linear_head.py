"""Multi-layer linear probe heads."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["MultiLayerLinearHead"]


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 2:
        return tensor[:, :, None, None]
    raise ValueError(f"Expected tensor with 2D or 4D shape, got {tuple(tensor.shape)}")


class MultiLayerLinearHead(nn.Module):
    """BatchNorm + 1x1 conv head that fuses multiple feature maps."""

    def __init__(
        self,
        *,
        feature_keys: Sequence[Hashable],
        in_channels: Mapping[Hashable, int],
        num_classes: int,
        primary_key: Hashable,
        fuse_mode: str = "concat",
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        affine_bn: bool = True,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if not feature_keys:
            raise ValueError("feature_keys must contain at least one entry.")

        self.feature_keys = list(feature_keys)
        self.primary_key = primary_key
        self.align_corners = align_corners

        if primary_key not in in_channels:
            raise ValueError(f"primary_key {primary_key!r} not present in in_channels mapping.")

        for key in self.feature_keys:
            if key not in in_channels:
                raise ValueError(f"feature_key {key!r} not present in in_channels mapping.")

        fuse_mode = fuse_mode.lower()
        if fuse_mode not in {"concat", "sum", "mean"}:
            raise ValueError("fuse_mode must be one of {'concat', 'sum', 'mean'}.")
        self.fuse_mode = fuse_mode

        if fuse_mode == "concat":
            fused_channels = sum(in_channels[key] for key in self.feature_keys)
        else:
            expected = in_channels[self.feature_keys[0]]
            for key in self.feature_keys[1:]:
                if in_channels[key] != expected:
                    raise ValueError(
                        f"fuse_mode '{fuse_mode}' requires all feature maps to have the same number "
                        f"of channels (got {in_channels[key]} for key {key!r}, expected {expected})."
                    )
            fused_channels = expected

        self.fused_channels = fused_channels
        self.num_classes = num_classes

        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(fused_channels, affine=affine_bn)
        else:
            self.bn = nn.Identity()
        self.proj = nn.Conv2d(fused_channels, num_classes, kernel_size=1, bias=True)

    def _resize(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if tensor.shape[-2:] == target_size:
            return tensor
        return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=self.align_corners)

    def _fuse(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if self.fuse_mode == "concat":
            return torch.cat(tensors, dim=1)
        if self.fuse_mode == "sum":
            return torch.stack(tensors, dim=0).sum(dim=0)
        # fuse_mode == "mean"
        return torch.stack(tensors, dim=0).mean(dim=0)

    def forward(self, features: Mapping[Hashable, torch.Tensor]) -> torch.Tensor:
        if self.primary_key not in features:
            raise KeyError(f"Primary feature {self.primary_key!r} missing from features.")
        primary = _ensure_4d(features[self.primary_key])
        target_size = primary.shape[-2:]

        fused_inputs = []
        for key in self.feature_keys:
            if key not in features:
                raise KeyError(f"Feature {key!r} missing from features.")
            feat = _ensure_4d(features[key])
            feat = self._resize(feat, target_size)
            fused_inputs.append(feat)

        fused = self._fuse(fused_inputs)
        fused = self.dropout(fused)
        fused = self.bn(fused)
        return self.proj(fused)
