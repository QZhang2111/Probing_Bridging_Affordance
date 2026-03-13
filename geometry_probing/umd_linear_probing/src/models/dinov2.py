"""Helpers to load DINOv2 backbones and extract patch tokens."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn

__all__ = ["DINOv2Backbone", "load_dinov2_backbone"]


def load_dinov2_backbone(
    model_name: str,
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    model = torch.hub.load(
        repo_or_dir=str(repo_path),
        model=model_name,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class DINOv2Backbone(nn.Module):
    """Wrapper that exposes patch tokens for a frozen DINOv2 model."""

    def __init__(
        self,
        model_name: str,
        repo_path: Path,
        checkpoint_path: Path,
        layers_to_hook: Sequence[int],
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = load_dinov2_backbone(model_name, repo_path, checkpoint_path, self.device)

        total_blocks = getattr(self.model, "n_blocks", len(getattr(self.model, "blocks")))
        resolved_layers: List[int] = []
        for layer in layers_to_hook:
            idx = layer
            if layer < 0:
                idx = total_blocks + layer
            if idx < 0 or idx >= total_blocks:
                raise ValueError(f"Layer index {layer} resolves to {idx}, outside [0, {total_blocks})")
            resolved_layers.append(idx)
        seen = set()
        ordered: List[int] = []
        for idx in resolved_layers:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        self.layer_order = ordered

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        autocast_precision: str | None = "bf16",
    ) -> OrderedDict[int, torch.Tensor]:
        autocast_dtype = None
        if autocast_precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif autocast_precision == "fp16":
            autocast_dtype = torch.float16

        images = images.to(self.device, non_blocking=True)
        if autocast_dtype is not None and self.device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            ctx = nullcontext()

        layer_indices = self.layer_order or [self.model.n_blocks - 1]
        with ctx:
            features = self.model.get_intermediate_layers(
                images,
                n=layer_indices,
                reshape=True,
                norm=True,
            )
        tokens = OrderedDict()
        for idx, feat in zip(layer_indices, features):
            tokens[idx] = feat.to(torch.float32)
        return tokens
