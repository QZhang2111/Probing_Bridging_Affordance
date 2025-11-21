"""OpenCLIP vision backbone wrapper that exposes patch tokens."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from typing import List, Optional, OrderedDict as OrderedDictType, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

__all__ = ["OpenCLIPBackbone"]


def _build_transform(preprocess_val: T.Compose) -> T.Compose:
    normalize = None
    for transform in getattr(preprocess_val, "transforms", []):
        if isinstance(transform, T.Normalize):
            normalize = transform
            break
    if normalize is None:
        raise ValueError("OpenCLIP preprocess does not contain a Normalize transform.")
    mean = list(getattr(normalize, "mean", normalize.mean))
    std = list(getattr(normalize, "std", normalize.std))
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def _import_open_clip(repo_path: Optional[str]):
    if repo_path:
        repo_src = Path(repo_path)
        if repo_src.is_dir():
            resolved = repo_src.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
        else:
            raise FileNotFoundError(f"Provided OpenCLIP repo path does not exist: {repo_path}")

    import open_clip  # type: ignore
    return open_clip


class OpenCLIPBackbone(nn.Module):
    """Wrapper around OpenCLIP visual encoder that returns patch tokens."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        precision: str = "fp32",
        repo_path: Optional[str] = None,
        layers_to_hook: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model_id = model_id
        self.requested_precision = precision

        open_clip_module = _import_open_clip(repo_path)
        model, _, preprocess_val = open_clip_module.create_model_and_transforms(model_id)

        self.visual = model.visual.to(self.device)
        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad_(False)

        transformer = getattr(self.visual, "transformer", None)
        block_seq = None
        if transformer is not None:
            block_seq = getattr(transformer, "resblocks", None)
            if block_seq is None:
                block_seq = getattr(transformer, "layers", None)
        if block_seq is None:
            raise ValueError("OpenCLIP visual transformer does not expose block sequence.")
        self.total_blocks = len(block_seq)

        resolved_layers: List[int] = []
        requested = layers_to_hook if layers_to_hook is not None else (-1,)
        for layer in requested:
            idx = layer
            if layer < 0:
                idx = self.total_blocks + layer
            if idx < 0 or idx >= self.total_blocks:
                raise ValueError(
                    f"Layer index {layer} resolves to {idx}, outside [0, {self.total_blocks})"
                )
            if idx not in resolved_layers:
                resolved_layers.append(idx)
        self.layer_order = resolved_layers or [self.total_blocks - 1]
        self.default_layer = self.layer_order[-1]

        base_grid = getattr(self.visual, "grid_size", None)
        if base_grid is not None:
            base_grid = tuple(int(dim) for dim in base_grid)
        else:
            tokens = self.visual.positional_embedding.shape[0] - 1
            size = int(round(tokens ** 0.5))
            base_grid = (size, size)
        self.base_grid = base_grid

        patch_size = getattr(self.visual, "patch_size", (16, 16))
        if isinstance(patch_size, tuple):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))

        self.transform = _build_transform(preprocess_val)

    def _resize_positional_embedding(self, height: int, width: int) -> torch.Tensor:
        pos_embed = self.visual.positional_embedding
        if pos_embed.ndim != 2:
            raise ValueError("Unexpected positional embedding shape.")

        cls_pos = pos_embed[0:1]
        patch_pos = pos_embed[1:]

        base_h, base_w = self.base_grid
        patch_pos = patch_pos.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)

        new_h = height // self.patch_size[0]
        new_w = width // self.patch_size[1]
        if new_h <= 0 or new_w <= 0:
            raise ValueError(f"Invalid spatial size for positional embedding: {(height, width)}")

        if (new_h, new_w) != (base_h, base_w):
            patch_pos = F.interpolate(
                patch_pos,
                size=(new_h, new_w),
                mode="bicubic",
                align_corners=False,
            )

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
        resized = torch.cat([cls_pos.unsqueeze(0), patch_pos], dim=1)
        return resized.squeeze(0)

    @contextmanager
    def _patched_visual_state(self, height: int, width: int):
        new_grid = (height // self.patch_size[0], width // self.patch_size[1])
        if new_grid == self.base_grid:
            yield
            return

        resized = self._resize_positional_embedding(height, width)
        resized = resized.to(self.visual.positional_embedding.device, dtype=self.visual.positional_embedding.dtype)

        original_pos = self.visual.positional_embedding
        original_grid = getattr(self.visual, "grid_size", None)

        try:
            with torch.no_grad():
                self.visual.positional_embedding = nn.Parameter(resized, requires_grad=False)
                if original_grid is not None:
                    self.visual.grid_size = new_grid
            yield
        finally:
            with torch.no_grad():
                self.visual.positional_embedding = original_pos
                if original_grid is not None:
                    self.visual.grid_size = original_grid

    # Legacy implementation retained for reference; superseded by forward_intermediates.
    # def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
    #     ...

    def forward(
        self,
        images: torch.Tensor,
        *,
        autocast_precision: str | None = "bf16",
    ) -> OrderedDictType[int, torch.Tensor]:
        autocast_dtype = None
        precision = autocast_precision or self.requested_precision
        if precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif precision == "fp16":
            autocast_dtype = torch.float16
        elif precision in ("fp32", "float32"):
            autocast_dtype = None

        images = images.to(self.device, non_blocking=True)
        if autocast_dtype is not None and self.device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            ctx = nullcontext()

        height, width = images.shape[-2:]
        with ctx:
            with self._patched_visual_state(height, width):
                output = self.visual.forward_intermediates(
                    images,
                    indices=self.layer_order,
                    normalize_intermediates=True,
                    intermediates_only=True,
                    output_fmt="NCHW",
                )

        features = output.get("image_intermediates")
        if features is None:
            raise RuntimeError("OpenCLIP forward_intermediates did not return image_intermediates.")

        tokens: OrderedDictType[int, torch.Tensor] = OrderedDict()
        for idx, feat in zip(self.layer_order, features):
            tokens[idx] = feat.to(torch.float32)

        # Legacy path:
        # patch_tokens = self._extract_patch_tokens(images)
        # tokens[self.default_layer] = patch_tokens
        return tokens
