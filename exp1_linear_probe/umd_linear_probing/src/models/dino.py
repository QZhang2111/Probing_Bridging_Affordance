"""Helpers to load frozen DINO and DINOv3 backbones and extract patch tokens."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import torch

__all__ = ["load_dino_backbone", "DINOBackbone", "load_dinov3_backbone", "DINOv3Backbone"]


_PREFIXES_TO_STRIP = ("module.", "backbone.")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for pref in _PREFIXES_TO_STRIP:
            if new_key.startswith(pref):
                new_key = new_key[len(pref) :]
        cleaned[new_key] = value
    return cleaned


def _resolve_state_dict(
    raw_state: Dict[str, torch.Tensor],
    checkpoint_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = raw_state
    if checkpoint_key and checkpoint_key in state_dict:
        candidate = state_dict[checkpoint_key]
        if isinstance(candidate, dict):
            state_dict = candidate
    for key in ("state_dict", "model", "teacher", "student"):
        candidate = state_dict.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break
    return _clean_state_dict(state_dict)


def _resolve_layers(layers_to_hook: Sequence[int], total_blocks: int) -> List[int]:
    resolved: List[int] = []
    seen = set()
    for layer in layers_to_hook:
        idx = layer
        if layer < 0:
            idx = total_blocks + layer
        if idx < 0 or idx >= total_blocks:
            raise ValueError(f"Layer index {layer} resolves to {idx}, outside [0, {total_blocks})")
        if idx not in seen:
            resolved.append(idx)
            seen.add(idx)
    if not resolved:
        raise ValueError("layers_to_hook must resolve to at least one valid block index.")
    return resolved


def _as_hw_tuple(patch_size: object) -> Tuple[int, int]:
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2:
            raise ValueError(f"Expected 2 elements for patch size, received {patch_size!r}")
        return int(patch_size[0]), int(patch_size[1])
    value = int(patch_size)
    return value, value


def _tokens_to_spatial(
    tokens: torch.Tensor,
    *,
    grid_hw: Tuple[int, int],
) -> torch.Tensor:
    """Convert ViT token sequence (with CLS) to 4D spatial tensor."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens with shape (B, N, C); received {tuple(tokens.shape)}")
    batch, num_tokens, channels = tokens.shape
    num_patches = num_tokens - 1
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != num_patches:
        raise ValueError(
            f"Grid {grid_hw} incompatible with {num_patches} patch tokens (total tokens {num_tokens})"
        )
    patch_tokens = tokens[:, 1:, :].transpose(1, 2)
    return patch_tokens.reshape(batch, channels, grid_h, grid_w)


def load_dino_backbone(
    model_name: str,
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    checkpoint_key: Optional[str] = None,
    hub_kwargs: Optional[Dict[str, object]] = None,
) -> torch.nn.Module:
    model = torch.hub.load(
        repo_or_dir=str(repo_path),
        model=model_name,
        source="local",
        pretrained=False,
        **(hub_kwargs or {}),
    )
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected checkpoint {checkpoint_path} to contain a state dict mapping.")
    state_dict = _resolve_state_dict(raw_state, checkpoint_key=checkpoint_key)
    msg = model.load_state_dict(state_dict, strict=False)
    unexpected = set(getattr(msg, "unexpected_keys", []))
    missing = set(getattr(msg, "missing_keys", []))
    if unexpected or missing:
        warnings.warn(
            f"DINO checkpoint load issues (missing: {sorted(missing)}, unexpected: {sorted(unexpected)})",
            RuntimeWarning,
        )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class DINOBackbone(torch.nn.Module):
    """Wrapper that exposes patch tokens for a frozen DINO (ViT / XCiT) model."""

    def __init__(
        self,
        model_name: str,
        repo_path: Path,
        checkpoint_path: Path,
        layers_to_hook: Sequence[int],
        *,
        device: str = "cuda",
        checkpoint_key: Optional[str] = None,
        hub_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = load_dino_backbone(
            model_name,
            repo_path,
            checkpoint_path,
            self.device,
            checkpoint_key=checkpoint_key,
            hub_kwargs=hub_kwargs,
        )
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise AttributeError("Loaded DINO model does not expose a 'blocks' attribute.")
        self.total_blocks = len(blocks)
        self.layer_order = _resolve_layers(layers_to_hook, self.total_blocks)
        if not hasattr(self.model, "patch_embed"):
            raise AttributeError("Loaded DINO model does not expose a 'patch_embed' attribute.")
        patch_attr = getattr(self.model.patch_embed, "patch_size")
        self.patch_hw = _as_hw_tuple(patch_attr)

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

        grid_h = images.shape[-2] // self.patch_hw[0]
        grid_w = images.shape[-1] // self.patch_hw[1]
        if grid_h * self.patch_hw[0] != images.shape[-2] or grid_w * self.patch_hw[1] != images.shape[-1]:
            raise ValueError(
                f"Input spatial size {tuple(images.shape[-2:])} is not divisible by patch size {self.patch_hw}"
            )

        with ctx:
            features = self.model.get_intermediate_layers(images, n=self.total_blocks)

        selected = {idx: features[idx] for idx in self.layer_order}
        tokens = OrderedDict()
        for idx in self.layer_order:
            tokens[idx] = _tokens_to_spatial(selected[idx], grid_hw=(grid_h, grid_w)).to(torch.float32)
        return tokens


def load_dinov3_backbone(
    model_name: str,
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = torch.hub.load(
        repo_or_dir=str(repo_path),
        model=model_name,
        source="local",
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(_clean_state_dict(state_dict), strict=False)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class DINOv3Backbone(torch.nn.Module):
    """Wrapper that exposes patch tokens for a frozen DINOv3 model."""

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
        self.model = load_dinov3_backbone(model_name, repo_path, checkpoint_path, self.device)

        total_blocks = len(getattr(self.model, "blocks"))
        resolved_layers: List[int] = []
        for layer in layers_to_hook:
            idx = layer
            if layer < 0:
                idx = total_blocks + layer
            if idx < 0 or idx >= total_blocks:
                raise ValueError(f"Layer index {layer} resolves to {idx}, outside [0, {total_blocks})")
            resolved_layers.append(idx)
        # Preserve declaration order while removing duplicates.
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

        with ctx:
            features = self.model.get_intermediate_layers(
                images,
                n=self.layer_order,
                reshape=True,
                norm=True,
            )
        if len(self.layer_order) == 1:
            features = (features[0],)
        tokens = OrderedDict()
        for idx, feat in zip(self.layer_order, features):
            tokens[idx] = feat.to(torch.float32)
        return tokens
