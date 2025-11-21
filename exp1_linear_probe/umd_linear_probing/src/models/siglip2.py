"""SigLIP2 backbone that matches the linear probing contract."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

__all__ = ["SigLIP2Backbone"]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _parse_precision(precision: Optional[str]) -> Optional[torch.dtype]:
    if precision is None:
        return None
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32
    return None


class SigLIP2Backbone(nn.Module):
    """Wraps SigLIP2 vision encoder and returns dense patch tokens."""

    def __init__(
        self,
        *,
        model_id: str = "google/siglip2-giant-opt-patch16-384",
        device: str = "cuda",
        precision: str = "bf16",
        layers_to_hook: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model_id = model_id
        self.requested_precision = precision

        processor = AutoProcessor.from_pretrained(model_id)
        self.processor = processor

        dtype = _parse_precision(precision) or torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        for param in self.model.parameters():
            param.requires_grad_(False)

        image_mean = torch.tensor(processor.image_processor.image_mean, dtype=torch.float32)
        image_std = torch.tensor(processor.image_processor.image_std, dtype=torch.float32)
        self.register_buffer("siglip_mean", image_mean.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("siglip_std", image_std.view(1, -1, 1, 1), persistent=False)

        self.register_buffer("imagenet_mean", _IMAGENET_MEAN.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", _IMAGENET_STD.view(1, -1, 1, 1), persistent=False)

        self.patch_size = 16
        encoder_layers = getattr(self.model.vision_model.encoder, "layers")
        self.total_blocks = len(encoder_layers)

        if layers_to_hook:
            resolved = []
            for layer in layers_to_hook:
                idx = layer if layer >= 0 else self.total_blocks + layer
                if idx < 0 or idx >= self.total_blocks:
                    raise ValueError(f"Layer index {layer} out of bounds for SigLIP2 encoder.")
                if idx not in resolved:
                    resolved.append(idx)
            self.layer_order = resolved
        else:
            self.layer_order = [self.total_blocks - 1]
        self.default_layer = self.layer_order[-1]

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        autocast_precision: Optional[str] = None,
    ) -> OrderedDict[str, torch.Tensor]:
        device = self.device
        dtype = _parse_precision(autocast_precision) or _parse_precision(self.requested_precision) or self.model_dtype

        images = images.to(device, dtype=torch.float32, non_blocking=True)
        height, width = images.shape[-2:]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        imagenet_std = self.imagenet_std.to(device=device)
        imagenet_mean = self.imagenet_mean.to(device=device)
        pixels = images * imagenet_std + imagenet_mean
        pixels = pixels.clamp(0.0, 1.0)

        siglip_mean = self.siglip_mean.to(device=device)
        siglip_std = self.siglip_std.to(device=device)
        pixel_values = (pixels - siglip_mean) / siglip_std
        pixel_values = pixel_values.to(dtype=dtype)

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )

        if isinstance(vision_outputs, dict):
            hidden_states = vision_outputs.get("hidden_states")
            last_hidden = vision_outputs.get("last_hidden_state")
        else:
            hidden_states = getattr(vision_outputs, "hidden_states", None)
            last_hidden = getattr(vision_outputs, "last_hidden_state", None)

        if hidden_states is None and last_hidden is None:
            raise ValueError("SigLIP2 vision model did not return hidden states or last_hidden_state.")

        outputs = OrderedDict()
        for layer_idx in self.layer_order:
            if hidden_states is not None:
                sequence = hidden_states[layer_idx]
            else:
                sequence = last_hidden

            if sequence.ndim == 2:
                raise ValueError(
                    "SigLIP2 get_image_features returned pooled embeddings; expected per-patch tokens. "
                    "Ensure the installed transformers version supports hidden_states for SigLIP2."
                )

            seq_len = sequence.shape[1]
            expected_tokens = grid_h * grid_w
            if seq_len == expected_tokens + 1:
                patch_tokens = sequence[:, 1:, :]
            elif seq_len == expected_tokens:
                patch_tokens = sequence
            else:
                raise ValueError(
                    f"Expected {expected_tokens} (or +1) tokens, received {seq_len}. "
                    "Check image preprocessing or model configuration."
                )

            patch_tokens = patch_tokens.to(torch.float32)
            batch_size = patch_tokens.shape[0]
            feature_dim = patch_tokens.shape[-1]
            patch_tokens = patch_tokens.reshape(batch_size, grid_h, grid_w, feature_dim).permute(0, 3, 1, 2).contiguous()
            outputs[layer_idx] = patch_tokens

        return outputs
