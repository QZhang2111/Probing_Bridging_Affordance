"""Linear probing heads for dense prediction."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["LinearProbeHead"]


class LinearProbeHead(nn.Module):
    """BatchNorm + 1x1 conv head as used in DINOv3 linear probes."""

    def __init__(self, embed_dim: int, num_classes: int, affine_bn: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(embed_dim, affine=affine_bn)
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.bn(x))
