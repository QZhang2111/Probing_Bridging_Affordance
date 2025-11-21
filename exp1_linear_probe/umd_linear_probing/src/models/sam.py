"""SAM (Segment Anything) backbone wrapper for linear probing."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from segment_anything import sam_model_registry

__all__ = ["SAMBackbone"]


class SAMBackbone(nn.Module):
    """Frozen SAM image encoder that outputs dense patch features."""

    def __init__(
        self,
        *,
        arch: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        patch_size: int = 16,
        device: str = "cuda",
        precision: str = "bf16",
        layers_to_hook: Optional[Sequence[int]] = None,
        primary_layer: Optional[int] = None,
    ) -> None:
        super().__init__()
        if arch not in {"vit_b", "vit_l", "vit_h"}:
            raise ValueError(f"Unsupported SAM architecture: {arch}")

        self.device = torch.device(device)
        self.patch_size = patch_size
        self.requested_precision = precision
        ckpt_map = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }
        if checkpoint_path:
            ckpt_file = Path(checkpoint_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(f"SAM checkpoint not found at {ckpt_file}. Provide a valid path.")
        else:
            ckpt_dir = Path(__file__).resolve().parent / "checkpoint_weights"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_file = ckpt_dir / ckpt_map[arch]
            if not ckpt_file.exists():
                url = f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt_map[arch]}"
                torch.hub.download_url_to_file(url, ckpt_file)

        sam = sam_model_registry[arch](checkpoint=str(ckpt_file))
        self.encoder = sam.image_encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        if self.encoder.patch_embed.proj.kernel_size[0] != self.patch_size:
            raise ValueError("SAM image encoder patch size mismatch.")

        self.embed_dim = self.encoder.neck[0].in_channels
        self.image_size = (
            self.encoder.pos_embed.shape[1] * self.patch_size,
            self.encoder.pos_embed.shape[2] * self.patch_size,
        )
        num_layers = len(self.encoder.blocks)
        default_layer = num_layers - 1
        resolved_layers: list[int] = []
        if layers_to_hook:
            for layer in layers_to_hook:
                idx = layer if layer >= 0 else num_layers + layer
                if idx < 0 or idx >= num_layers:
                    raise ValueError(f"Layer index {layer} out of bounds for SAM encoder.")
                if idx not in resolved_layers:
                    resolved_layers.append(idx)
        else:
            resolved_layers = [default_layer]

        if primary_layer is None:
            primary_idx = resolved_layers[-1]
        else:
            primary_idx = primary_layer if primary_layer >= 0 else num_layers + primary_layer
            if primary_idx < 0 or primary_idx >= num_layers:
                raise ValueError(f"Primary layer index {primary_layer} out of bounds for SAM encoder.")
            if primary_idx not in resolved_layers:
                resolved_layers = [primary_idx, *resolved_layers]
            else:
                resolved_layers = [primary_idx, *[idx for idx in resolved_layers if idx != primary_idx]]

        self.layer_order = resolved_layers
        self.primary_layer = primary_idx
        self.layer_key = self.primary_layer

        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1) / 255.0
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1) / 255.0
        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        autocast_precision: Optional[str] = None,
    ) -> OrderedDict[str, torch.Tensor]:
        if images.numel() == 0:
            raise ValueError("Received empty image batch.")

        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        batch, _, height, width = images.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dimensions {(height, width)} must be divisible by patch size {self.patch_size}."
            )

        normed = (images - self.pixel_mean) / self.pixel_std
        encoder_dtype = self.encoder.patch_embed.proj.weight.dtype
        normed = normed.to(dtype=encoder_dtype)

        features = self._run_encoder(normed)
        target_hw = (height // self.patch_size, width // self.patch_size)

        outputs = OrderedDict()
        for idx in self.layer_order:
            feature_map = features[idx].to(torch.float32)
            if feature_map.shape[-2:] != target_hw:
                feature_map = torch.nn.functional.interpolate(
                    feature_map,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs[idx] = feature_map

        return outputs

    def _run_encoder(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        vit = self.encoder
        _, _, h, w = images.shape

        patch = vit.patch_embed.proj.kernel_size[0]
        target_h = ((h + patch - 1) // patch) * patch
        target_w = ((w + patch - 1) // patch) * patch
        if (target_h, target_w) != self.image_size:
            self._resize_pos_embed((target_h, target_w))

        x = vit.patch_embed(images)
        pos_embed = vit.pos_embed
        if (x.shape[1], x.shape[2]) != (pos_embed.shape[1], pos_embed.shape[2]):
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=x.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        x = x + pos_embed

        collected: dict[int, torch.Tensor] = {}
        target_indices = set(self.layer_order)

        for idx, block in enumerate(vit.blocks):
            x = block(x)
            if idx in target_indices:
                collected[idx] = x.permute(0, 3, 1, 2).contiguous()

        return collected

    def _resize_pos_embed(self, image_size: tuple[int, int]) -> None:
        patch = self.encoder.patch_embed.proj.kernel_size[0]
        h = image_size[0] // patch
        w = image_size[1] // patch
        pos_embed = self.encoder.pos_embed
        pos_embed = torch.nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        self.encoder.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self.image_size = image_size
