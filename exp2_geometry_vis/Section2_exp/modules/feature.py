"""Feature extractor registry supporting multiple backbones."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from .config import ModelSpec

try:
    import open_clip
except ImportError:  # pragma: no cover - optional dependency
    open_clip = None

try:
    from segment_anything import sam_model_registry
except ImportError:  # pragma: no cover - optional dependency
    sam_model_registry = None

try:
    from transformers import SiglipVisionModel
except ImportError:  # pragma: no cover - optional dependency
    SiglipVisionModel = None

try:
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    try:
        from diffusers.models.unet_2d_condition import UNet2DConditionModel
    except ImportError:  # pragma: no cover - fallback for older diffusers
        from diffusers import UNet2DConditionModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None  # type: ignore[assignment]
    DDIMScheduler = None  # type: ignore[assignment]
    UNet2DConditionModel = None  # type: ignore[assignment]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ResizeMeta:
    orig_w: int
    orig_h: int
    target_w: int
    target_h: int
    inner_w: int
    inner_h: int
    final_w: int
    final_h: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "target_w": self.target_w,
            "target_h": self.target_h,
            "inner_w": self.inner_w,
            "inner_h": self.inner_h,
            "final_w": self.final_w,
            "final_h": self.final_h,
            "resized_w": self.final_w,
            "resized_h": self.final_h,
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
        }

    @property
    def resized_w(self) -> int:
        return self.inner_w

    @property
    def resized_h(self) -> int:
        return self.inner_h


def _round_to_multiple(x: int, k: int, *, mode: str = "floor") -> int:
    if mode == "floor":
        return (x // k) * k
    if mode == "ceil":
        return ((x + k - 1) // k) * k
    raise ValueError("mode must be 'floor' or 'ceil'")


def letterbox_image(
    img: Image.Image,
    target_size: Tuple[int, int],
    patch_size: int,
    *,
    keep_full_resolution: bool = False,
) -> Tuple[Image.Image, ResizeMeta]:
    target_w_cfg, target_h_cfg = target_size
    orig_w, orig_h = img.size

    if keep_full_resolution:
        min_w = _round_to_multiple(orig_w, patch_size, mode="ceil")
        min_h = _round_to_multiple(orig_h, patch_size, mode="ceil")
        target_w = max(target_w_cfg, min_w)
        target_h = max(target_h_cfg, min_h)

        if target_w % patch_size or target_h % patch_size:
            raise ValueError("target dimensions must be divisible by patch size")

        inner_w = orig_w
        inner_h = orig_h
        pad_w = max(0, target_w - inner_w)
        pad_h = max(0, target_h - inner_h)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(img, (pad_left, pad_top))
        scale = 1.0
    else:
        target_w = target_w_cfg
        target_h = target_h_cfg
        if target_w % patch_size or target_h % patch_size:
            raise ValueError("target dimensions must be divisible by patch size")

        scale = min(target_w / float(orig_w), target_h / float(orig_h))
        inner_w = max(_round_to_multiple(int(orig_w * scale), patch_size, mode="floor"), patch_size)
        inner_h = max(_round_to_multiple(int(orig_h * scale), patch_size, mode="floor"), patch_size)
        inner_w = min(inner_w, target_w)
        inner_h = min(inner_h, target_h)

        resized_img = img.resize((inner_w, inner_h), Image.BICUBIC)
        pad_w = target_w - inner_w
        pad_h = target_h - inner_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(resized_img, (pad_left, pad_top))

    meta = ResizeMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=target_w,
        target_h=target_h,
        inner_w=inner_w,
        inner_h=inner_h,
        final_w=target_w,
        final_h=target_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    return canvas, meta


def _pad_to_patch_multiple(
    img: Image.Image,
    patch_size: int,
) -> Tuple[Image.Image, ResizeMeta]:
    orig_w, orig_h = img.size
    target_w = _round_to_multiple(orig_w, patch_size, mode="ceil")
    target_h = _round_to_multiple(orig_h, patch_size, mode="ceil")
    pad_left = 0
    pad_top = 0
    pad_right = max(0, target_w - orig_w)
    pad_bottom = max(0, target_h - orig_h)

    if pad_right or pad_bottom:
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(img, (pad_left, pad_top))
    else:
        canvas = img

    meta = ResizeMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=target_w,
        target_h=target_h,
        inner_w=orig_w,
        inner_h=orig_h,
        final_w=target_w,
        final_h=target_h,
        scale=1.0,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    return canvas, meta


def _identity_meta(image: Image.Image) -> ResizeMeta:
    orig_w, orig_h = image.size
    return ResizeMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=orig_w,
        target_h=orig_h,
        inner_w=orig_w,
        inner_h=orig_h,
        final_w=orig_w,
        final_h=orig_h,
        scale=1.0,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
    )


def restore_original_resolution(array: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    """Remove letterbox padding and resize back to the original resolution."""

    squeeze_back = False
    if array.ndim == 2:
        array = array[..., None]
        squeeze_back = True

    top = max(meta.pad_top, 0)
    left = max(meta.pad_left, 0)
    bottom = min(top + meta.inner_h, array.shape[0])
    right = min(left + meta.inner_w, array.shape[1])
    cropped = array[top:bottom, left:right, :]
    if cropped.size == 0:
        raise ValueError("Cropped region is empty; verify ResizeMeta padding values.")

    tensor = torch.from_numpy(cropped.transpose(2, 0, 1)).unsqueeze(0).float()
    restored = F.interpolate(
        tensor,
        size=(meta.orig_h, meta.orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0).cpu().numpy()

    if squeeze_back:
        restored = restored[..., 0]
    return restored


class BaseExtractor:
    """Abstract feature extractor."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Optional[torch.nn.Module] = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.spec.device)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def extract_image(self, image_path: Path | str, *, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int, int, ResizeMeta]:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        prepared, meta = self.preprocess(rgb, target_size=target_size)
        tensor = self.to_tensor(prepared)
        tokens, Hp, Wp = self.encode_tokens(tensor, meta)
        return tokens, Hp, Wp, meta

    def preprocess(self, image: Image.Image, *, target_size: Optional[Tuple[int, int]] = None) -> Tuple[Image.Image, ResizeMeta]:
        if self.spec.use_letterbox:
            target = target_size if target_size is not None else self.spec.target_size
            letterboxed, meta = letterbox_image(
                image,
                target,
                self.spec.patch_size,
                keep_full_resolution=self.spec.keep_full_resolution,
            )
            return letterboxed, meta

        prepared = image
        if self.spec.pad_to_patch_multiple:
            prepared, meta = _pad_to_patch_multiple(prepared, self.spec.patch_size)
        else:
            if prepared.size[0] % self.spec.patch_size or prepared.size[1] % self.spec.patch_size:
                raise ValueError(
                    f"Image dimensions {prepared.size} are not divisible by patch size {self.spec.patch_size}. "
                    "Enable 'pad_to_patch_multiple' or provide divisible inputs."
                )
            meta = _identity_meta(prepared)
        return prepared, meta

    def to_tensor(self, prepared: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(prepared)
        norm = self.spec.normalization
        tensor = TF.normalize(tensor, mean=norm.mean, std=norm.std)
        return tensor.unsqueeze(0).to(self.device).float()

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        raise NotImplementedError


class TorchHubViTExtractor(BaseExtractor):
    hub_model_name: Optional[str] = None

    def build_model(self) -> torch.nn.Module:
        if self.spec.repo is None:
            raise ValueError(f"Model '{self.spec.key}' requires 'repo' in configuration")
        model_name = self.spec.hub_model or self.hub_model_name
        if not model_name:
            raise ValueError(f"Model '{self.spec.key}' must specify a hub model name")

        model = torch.hub.load(
            repo_or_dir=str(self.spec.repo),
            model=model_name,
            source="local",
            pretrained=False,
        )
        state = None
        if self.spec.checkpoint is not None:
            checkpoint = torch.load(self.spec.checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            else:
                state = checkpoint
        if state is not None:
            cleaned = {}
            for key, value in state.items():
                new_key = key
                for prefix in ("module.", "backbone."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                cleaned[new_key] = value
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[{self.spec.key}] missing keys: {len(missing)}")
            if unexpected:
                print(f"[{self.spec.key}] unexpected keys: {len(unexpected)}")
        model.eval().to(self.device)
        blocks = getattr(model, "blocks", None)
        self.total_blocks = len(blocks) if blocks is not None else None
        self.layer_indices = self._resolve_layers(self.spec.layers_to_hook)
        return model

    def _resolve_layers(self, layers: Optional[Tuple[int, ...]]) -> List[int]:
        if self.total_blocks is None:
            return [0]
        resolved: List[int] = []
        if layers is None or len(layers) == 0:
            return [self.total_blocks - 1]
        for layer in layers:
            idx = layer
            if layer < 0:
                idx = self.total_blocks + layer
            if idx < 0 or idx >= self.total_blocks:
                raise ValueError(
                    f"Layer index {layer} resolves to {idx}, outside [0, {self.total_blocks}) for '{self.spec.key}'"
                )
            if idx not in resolved:
                resolved.append(idx)
        return resolved or [self.total_blocks - 1]

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        with torch.inference_mode():
            _ = self.model  # ensure build_model executed
            layers = getattr(self, "layer_indices", None)
            if layers is None:
                layers = self._resolve_layers(self.spec.layers_to_hook)
            if len(layers) != 1:
                raise ValueError(
                    f"Model '{self.spec.key}' received {len(layers)} layer requests; expected exactly one."
                )
            idx = layers[0]
            if self.total_blocks is None:
                outputs = self.model.get_intermediate_layers(tensor, n=1)
                output = outputs[0]
            else:
                n_last = max(1, self.total_blocks - idx)
                outputs = self.model.get_intermediate_layers(tensor, n=n_last)
                offset = idx - (self.total_blocks - n_last)
                output = outputs[offset]
        tokens = output.squeeze(0).to(torch.float32)
        Hp = meta.target_h // self.spec.patch_size
        Wp = meta.target_w // self.spec.patch_size
        expected = Hp * Wp
        if tokens.shape[0] == expected + 1:
            tokens = tokens[1:]
        elif tokens.shape[0] != expected:
            raise ValueError(f"Unexpected token count {tokens.shape[0]} for grid {Hp}x{Wp}")
        return tokens.cpu().numpy(), Hp, Wp


class Dinov3Extractor(TorchHubViTExtractor):
    hub_model_name = "dinov3_vit7b16"

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            _ = self.model
            layers = getattr(self, "layer_indices", None)
            if layers is None:
                layers = self._resolve_layers(self.spec.layers_to_hook)
            feats = self.model.get_intermediate_layers(tensor, n=layers, reshape=True, norm=True)
        if len(feats) != 1:
            raise ValueError(
                f"DINOv3 extractor expected a single layer; received {len(feats)}. "
                "Set layers_to_hook to a single index."
            )
        x = feats[0].squeeze(0).cpu().float()  # [C, H, W]
        C, Hp, Wp = x.shape
        tokens = x.view(C, Hp * Wp).permute(1, 0).contiguous().numpy()
        return tokens, Hp, Wp


class DinoExtractor(TorchHubViTExtractor):
    hub_model_name = "dino_vitb16"


class Dinov2Extractor(TorchHubViTExtractor):
    hub_model_name = "dinov2_vitb14"


class SamExtractor(BaseExtractor):
    def build_model(self) -> torch.nn.Module:
        if sam_model_registry is None:
            raise ImportError("segment_anything is required for SAM extraction")
        if self.spec.checkpoint is None:
            raise ValueError(f"Model '{self.spec.key}' requires a checkpoint path")
        model = sam_model_registry["vit_b"](checkpoint=str(self.spec.checkpoint))
        model.eval().to(self.device)
        self.encoder = model.image_encoder
        self.patch_size_hw = (
            int(self.encoder.patch_embed.proj.kernel_size[0]),
            int(self.encoder.patch_embed.proj.kernel_size[1]),
        )
        self.base_pos_embed = self.encoder.pos_embed
        self.base_image_size = (
            self.base_pos_embed.shape[1] * self.patch_size_hw[0],
            self.base_pos_embed.shape[2] * self.patch_size_hw[1],
        )
        num_layers = len(self.encoder.blocks)
        layers_cfg = self.spec.layers_to_hook
        self.layer_order = self._resolve_layers(layers_cfg, num_layers)
        if len(self.layer_order) != 1:
            raise ValueError(
                f"SAM extractor for '{self.spec.key}' expects a single layer. "
                "Please configure layers_to_hook or layer_variants with one index."
            )
        self.target_layer = self.layer_order[0]
        return model

    def preprocess(self, image: Image.Image, *, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, ResizeMeta]:
        if self.spec.use_letterbox:
            target = target_size if target_size is not None else self.spec.target_size
            letterboxed, meta = letterbox_image(image, target, self.spec.patch_size)
            arr = np.asarray(letterboxed, dtype=np.float32)
        else:
            if self.spec.pad_to_patch_multiple:
                padded_img, meta = _pad_to_patch_multiple(image, self.spec.patch_size)
            else:
                if image.size[0] % self.spec.patch_size or image.size[1] % self.spec.patch_size:
                    raise ValueError(
                        f"Image dimensions {image.size} are not divisible by patch size {self.spec.patch_size}. "
                        "Enable 'pad_to_patch_multiple' or provide divisible inputs."
                    )
                padded_img = image
                meta = _identity_meta(image)
            arr = np.asarray(padded_img, dtype=np.float32)
        return arr, meta

    def to_tensor(self, prepared: np.ndarray) -> torch.Tensor:
        arr = prepared
        if getattr(self.model, "image_format", "RGB") == "BGR":
            arr = arr[..., ::-1]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        tensor = tensor.float()
        pixel_mean = self.model.pixel_mean.view(1, 3, 1, 1).to(self.device)
        pixel_std = self.model.pixel_std.view(1, 3, 1, 1).to(self.device)
        tensor = (tensor - pixel_mean) / pixel_std
        return tensor

    def _resize_pos_embed(self, size_hw: Tuple[int, int]) -> None:
        patch = self.patch_size_hw[0]
        h = size_hw[0] // patch
        w = size_hw[1] // patch
        pos_embed = self.encoder.pos_embed
        pos_embed = torch.nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        self.encoder.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)

    def _resolve_layers(self, layers: Optional[Tuple[int, ...]], total: int) -> List[int]:
        resolved: List[int] = []
        if layers is None or len(layers) == 0:
            return [total - 1]
        for layer in layers:
            idx = layer
            if layer < 0:
                idx = total + layer
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"SAM layer index {layer} resolves to {idx}, outside [0, {total})"
                )
            if idx not in resolved:
                resolved.append(idx)
        return resolved or [total - 1]

    def _run_encoder(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        vit = self.encoder
        _, _, h, w = images.shape

        patch = vit.patch_embed.proj.kernel_size[0]
        target_h = ((h + patch - 1) // patch) * patch
        target_w = ((w + patch - 1) // patch) * patch
        if (target_h, target_w) != self.base_image_size:
            self._resize_pos_embed((target_h, target_w))
            self.base_image_size = (target_h, target_w)

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

        collected: Dict[int, torch.Tensor] = {}
        target_indices = set(self.layer_order)

        for idx, block in enumerate(vit.blocks):
            x = block(x)
            if idx in target_indices:
                collected[idx] = x.permute(0, 3, 1, 2).contiguous()

        return collected

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        with torch.inference_mode():
            _ = self.model
            if not hasattr(self, "patch_size_hw"):
                value = int(self.spec.patch_size)
                self.patch_size_hw = (value, value)
            height = tensor.shape[-2]
            width = tensor.shape[-1]
            if height % self.patch_size_hw[0] or width % self.patch_size_hw[1]:
                raise ValueError(
                    f"Input spatial size {(height, width)} must be divisible by patch size {self.patch_size_hw}."
                )
            current_size = (height, width)
            if current_size != self.base_image_size:
                self._resize_pos_embed(current_size)
            features = self._run_encoder(tensor)
        if self.target_layer not in features:
            raise ValueError(f"SAM layer {self.target_layer} was not captured; available {list(features.keys())}")
        feature_map = features[self.target_layer].to(torch.float32)
        target_hw = (meta.target_h // self.spec.patch_size, meta.target_w // self.spec.patch_size)
        if feature_map.shape[-2:] != target_hw:
            feature_map = torch.nn.functional.interpolate(
                feature_map,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        x = feature_map.squeeze(0).cpu().float()  # [C, H, W]
        C, Hp, Wp = x.shape
        tokens = x.view(C, Hp * Wp).permute(1, 0).contiguous().numpy()
        return tokens, Hp, Wp


class OpenCLIPExtractor(BaseExtractor):
    def build_model(self) -> torch.nn.Module:
        if open_clip is None:
            raise ImportError("open_clip_torch is required for CLIP extraction")
        if self.spec.model_id is None:
            raise ValueError(f"Model '{self.spec.key}' must define 'model_id'")
        result = open_clip.create_model_from_pretrained(self.spec.model_id)
        if not isinstance(result, tuple):
            model = result
        else:
            model = result[0]
        model.visual.output_tokens = True
        model.eval().to(self.device)
        visual = model.visual.to(self.device)
        visual.eval()

        base_grid = getattr(visual, "grid_size", None)
        if base_grid is not None:
            base_grid = tuple(int(dim) for dim in base_grid)
        else:
            tokens = visual.positional_embedding.shape[0] - 1
            size = int(round(tokens ** 0.5))
            base_grid = (size, size)
        patch_size = getattr(visual, "patch_size", (16, 16))
        if isinstance(patch_size, tuple):
            patch_hw = (int(patch_size[0]), int(patch_size[1]))
        else:
            patch_hw = (int(patch_size), int(patch_size))

        transformer = getattr(visual, "transformer", None)
        block_seq = None
        if transformer is not None:
            block_seq = getattr(transformer, "resblocks", None)
            if block_seq is None:
                block_seq = getattr(transformer, "layers", None)
        if block_seq is None:
            raise ValueError("OpenCLIP visual transformer does not expose block sequence.")
        self.total_blocks = len(block_seq)
        self.layer_indices = self._resolve_layer_indices(self.spec.layers_to_hook)

        self.visual = visual
        self.base_grid = base_grid
        self.patch_size_hw = patch_hw
        return model

    def _resolve_layer_indices(self, layers: Optional[Tuple[int, ...]]) -> List[int]:
        resolved: List[int] = []
        if layers is None or not layers:
            return [self.total_blocks - 1]
        for layer in layers:
            idx = layer
            if layer < 0:
                idx = self.total_blocks + layer
            if idx < 0 or idx >= self.total_blocks:
                raise ValueError(
                    f"Layer index {layer} resolves to {idx}, outside [0, {self.total_blocks}) for '{self.spec.key}'"
                )
            if idx not in resolved:
                resolved.append(idx)
        return resolved or [self.total_blocks - 1]

    def _resize_positional_embedding(self, height: int, width: int) -> torch.Tensor:
        pos_embed = self.visual.positional_embedding
        if pos_embed.ndim != 2:
            raise ValueError("Unexpected positional embedding shape for OpenCLIP.")
        cls_pos = pos_embed[0:1]
        patch_pos = pos_embed[1:]

        base_h, base_w = self.base_grid
        patch_pos = patch_pos.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)

        new_h = height // self.patch_size_hw[0]
        new_w = width // self.patch_size_hw[1]
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
        new_grid = (height // self.patch_size_hw[0], width // self.patch_size_hw[1])
        original_pos = self.visual.positional_embedding
        original_grid = getattr(self.visual, "grid_size", None)

        if new_grid == self.base_grid:
            yield
            return

        resized = self._resize_positional_embedding(height, width)
        resized = resized.to(original_pos.device, dtype=original_pos.dtype)

        try:
            with torch.no_grad():
                self.visual.positional_embedding = torch.nn.Parameter(resized, requires_grad=False)
                if original_grid is not None:
                    self.visual.grid_size = new_grid
            yield
        finally:
            with torch.no_grad():
                self.visual.positional_embedding = original_pos
                if original_grid is not None:
                    self.visual.grid_size = original_grid

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        _ = self.model  # ensure build_model executed
        height = tensor.shape[-2]
        width = tensor.shape[-1]
        if height % self.patch_size_hw[0] or width % self.patch_size_hw[1]:
            raise ValueError(
                f"Input spatial size {(height, width)} must be divisible by patch size {self.patch_size_hw}."
            )
        with torch.inference_mode():
            with self._patched_visual_state(height, width):
                outputs = self.visual.forward_intermediates(
                    tensor,
                    indices=self.layer_indices,
                    normalize_intermediates=True,
                    intermediates_only=True,
                    output_fmt="NCHW",
                )
        feats = outputs.get("image_intermediates") if isinstance(outputs, dict) else outputs
        if not feats:
            raise RuntimeError("OpenCLIP forward_intermediates did not return image_intermediates.")
        if len(feats) != 1:
            raise ValueError(
                f"OpenCLIP extractor expected a single layer; received {len(feats)}. "
                "Set layers_to_hook to a single index or use layer_variants."
            )
        feature = feats[0].squeeze(0).to(torch.float32)  # [C, H, W]
        C, Hp, Wp = feature.shape
        tokens = feature.view(C, Hp * Wp).permute(1, 0).contiguous()
        return tokens.cpu().numpy(), Hp, Wp


class SiglipExtractor(BaseExtractor):
    def build_model(self) -> torch.nn.Module:
        if SiglipVisionModel is None:
            raise ImportError("transformers is required for SigLIP extraction")
        if self.spec.model_id is None:
            raise ValueError(f"Model '{self.spec.key}' must define 'model_id'")
        model = SiglipVisionModel.from_pretrained(self.spec.model_id)
        model.eval().to(self.device)
        embeddings = model.vision_model.embeddings
        patch_module = getattr(embeddings, "patch_embeddings", None)
        if patch_module is None:
            patch_module = getattr(embeddings, "patch_embedding", None)
        if patch_module is None:
            patch_attr = self.spec.patch_size
        else:
            patch_attr = getattr(patch_module, "patch_size", self.spec.patch_size)
        if isinstance(patch_attr, (tuple, list)):
            self.patch_size_hw = (int(patch_attr[0]), int(patch_attr[1]))
        else:
            value = int(patch_attr)
            self.patch_size_hw = (value, value)
        pos_embed = embeddings.position_embedding.weight.detach().clone()
        base_tokens = pos_embed.shape[0]
        base_size = int(round(base_tokens ** 0.5))
        self.base_grid = (base_size, base_size)
        self.base_positional_embedding = pos_embed
        encoder_layers = getattr(model.vision_model.encoder, "layers")
        self.total_layers = len(encoder_layers)
        self.layer_indices = self._resolve_layer_indices(self.spec.layers_to_hook)
        return model

    def _resolve_layer_indices(self, layers: Optional[Tuple[int, ...]]) -> List[int]:
        if not hasattr(self, "total_layers"):
            self.total_layers = len(getattr(self.model.vision_model.encoder, "layers"))
        total = self.total_layers
        resolved: List[int] = []
        if layers is None or not layers:
            return [total - 1]
        for layer in layers:
            idx = layer
            if layer < 0:
                idx = total + layer
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"Layer index {layer} resolves to {idx}, outside [0, {total}) for '{self.spec.key}'"
                )
            if idx not in resolved:
                resolved.append(idx)
        return resolved or [total - 1]

    @contextmanager
    def _patched_positional_embedding(self, height: int, width: int):
        new_grid = (height // self.patch_size_hw[0], width // self.patch_size_hw[1])
        base_h, base_w = self.base_grid
        if new_grid == self.base_grid:
            yield
            return

        embeddings = self.model.vision_model.embeddings
        device = embeddings.position_embedding.weight.device
        dtype = embeddings.position_embedding.weight.dtype
        base_pos = self.base_positional_embedding.to(device=device, dtype=dtype)

        patch_pos = base_pos.reshape(base_h, base_w, -1).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            patch_pos,
            size=new_grid,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0).reshape(-1, base_pos.shape[-1])

        original = embeddings.position_embedding.weight
        original_ids = embeddings.position_ids
        new_tokens = new_grid[0] * new_grid[1]
        embeddings.position_embedding.weight = torch.nn.Parameter(resized, requires_grad=False)
        embeddings.register_buffer(
            "position_ids",
            torch.arange(new_tokens, device=device).unsqueeze(0),
            persistent=False,
        )
        try:
            yield
        finally:
            embeddings.position_embedding.weight = torch.nn.Parameter(base_pos, requires_grad=False)
            embeddings.register_buffer("position_ids", original_ids, persistent=False)

    def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
        with torch.inference_mode():
            _ = self.model
            if not hasattr(self, "patch_size_hw"):
                value = int(self.spec.patch_size)
                self.patch_size_hw = (value, value)
            if not hasattr(self, "base_grid") or not hasattr(self, "base_positional_embedding"):
                embeddings = self.model.vision_model.embeddings
                base_pos = embeddings.position_embedding.weight.detach().clone()
                base_tokens = base_pos.shape[0]
                base_size = int(round(base_tokens ** 0.5))
                self.base_grid = (base_size, base_size)
                self.base_positional_embedding = base_pos
            if not hasattr(self, "layer_indices"):
                self.layer_indices = self._resolve_layer_indices(self.spec.layers_to_hook)
            height = tensor.shape[-2]
            width = tensor.shape[-1]
            if height % self.patch_size_hw[0] or width % self.patch_size_hw[1]:
                raise ValueError(
                    f"Input spatial size {(height, width)} must be divisible by patch size {self.patch_size_hw}."
                )
            with self._patched_positional_embedding(height, width):
                outputs = self.model(
                    pixel_values=tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            sequence = outputs.last_hidden_state
            layer_idx = -1
        else:
            layers = self.layer_indices
            if layers is None or len(layers) != 1:
                raise ValueError(
                    f"SigLIP extractor expected a single layer; received {layers}. "
                    "Configure layers_to_hook with exactly one index."
                )
            layer_idx = layers[0]
            sequence = hidden_states[layer_idx]
        if sequence is None:
            raise ValueError("SigLIP model did not return hidden states.")
        sequence = sequence.squeeze(0).to(torch.float32)
        seq_len = sequence.shape[0]

        grid_h = meta.final_h // self.patch_size_hw[0]
        grid_w = meta.final_w // self.patch_size_hw[1]
        expected = grid_h * grid_w

        if seq_len == expected + 1:
            patch_tokens = sequence[1:]
        elif seq_len == expected:
            patch_tokens = sequence
        else:
            raise ValueError(
                f"SigLIP token count mismatch: {seq_len} vs expected {expected} (grid {grid_h}x{grid_w})."
            )

        patch_tokens = patch_tokens.reshape(grid_h, grid_w, -1)
        tokens_2d = patch_tokens.reshape(grid_h * grid_w, patch_tokens.shape[-1])
        return tokens_2d.cpu().numpy(), grid_h, grid_w


if StableDiffusionPipeline is not None and UNet2DConditionModel is not None and DDIMScheduler is not None:

    @dataclass(frozen=True)
    class SDLayerSpec:
        index: int
        name: str
        feat_dim: int


    class MyUNet2DConditionModel(UNet2DConditionModel):  # type: ignore[misc]
        """UNet variant that exposes up-block activations."""

        def forward(  # type: ignore[override]
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            up_ft_indices: Sequence[int],
            encoder_hidden_states: torch.Tensor,
        ) -> Dict[str, Dict[int, torch.Tensor]]:
            default_overall_up_factor = 2 ** self.num_upsamplers
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                forward_upsample_size = True

            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            timesteps = timestep
            if isinstance(timesteps, (int, float)):
                timesteps = torch.tensor([timesteps], device=sample.device, dtype=sample.dtype)
            elif timesteps.ndim == 0:
                timesteps = timesteps[None].to(sample.device)
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
            emb = self.time_embedding(t_emb, None)

            sample = self.conv_in(sample)

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                has_ca = hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention
                if has_ca:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=None,
                        cross_attention_kwargs=None,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                down_block_res_samples += res_samples

            if self.mid_block is not None:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )

            collected: Dict[int, torch.Tensor] = {}
            for idx, upsample_block in enumerate(self.up_blocks):
                if idx > max(up_ft_indices):
                    break

                is_final_block = idx == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                has_ca = hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention
                if has_ca:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=None,
                        upsample_size=upsample_size,
                        attention_mask=None,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

                if idx in up_ft_indices:
                    collected[idx] = sample

            return {"up_ft": collected}


    class OneStepSDPipeline(StableDiffusionPipeline):  # type: ignore[misc]
        """Pipeline wrapper that returns UNet activations in a single call."""

        def __call__(  # type: ignore[override]
            self,
            img_tensor: torch.Tensor,
            timestep: torch.Tensor,
            up_ft_indices: Sequence[int],
            prompt_embeds: Optional[torch.Tensor] = None,
        ) -> Dict[str, Dict[int, torch.Tensor]]:
            device = self._execution_device
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

            if timestep.ndim == 0:
                timesteps = timestep[None].to(device)
            else:
                timesteps = timestep.to(device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, timesteps)
            return self.unet(
                latents_noisy,
                timesteps,
                up_ft_indices=up_ft_indices,
                encoder_hidden_states=prompt_embeds,
            )


    class SDFeaturizer(torch.nn.Module):
        """Wrap Stable Diffusion to expose intermediate UNet activations."""

        def __init__(self, model_id: str, device: torch.device, torch_dtype: torch.dtype) -> None:
            super().__init__()
            unet = MyUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)
            pipe = OneStepSDPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                unet=unet,
                safety_checker=None,
            )
            pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe.vae.decoder = None
            pipe = pipe.to(device)

            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder.to(device)
            self.unet = pipe.unet.to(device)
            self.vae = pipe.vae.to(device)
            self.scheduler = pipe.scheduler
            self.device = device
            self.dtype = torch_dtype

            for module in (self.text_encoder, self.unet, self.vae):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

        @torch.no_grad()
        def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            text_outputs = self.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
            return text_outputs[0].to(dtype=self.dtype)

        @torch.no_grad()
        def forward(  # type: ignore[override]
            self,
            images: torch.Tensor,
            prompts: Sequence[str],
            timestep: int,
            up_ft_indices: Sequence[int],
        ) -> Dict[int, torch.Tensor]:
            prompt_embeds = self.encode_prompt(prompts)
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(images).latent_dist.mode()

            t = torch.tensor(timestep, dtype=torch.long, device=self.device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            output = self.unet(
                latents_noisy,
                t,
                up_ft_indices=up_ft_indices,
                encoder_hidden_states=prompt_embeds,
            )
            return output["up_ft"]


    class StableDiffusionBackbone(torch.nn.Module):
        """Stable Diffusion 2.1 feature extractor mirroring the UMD pipeline."""

        LAYER_SPECS: Sequence[SDLayerSpec] = (
            SDLayerSpec(index=0, name="up_0", feat_dim=1280),
            SDLayerSpec(index=1, name="up_1", feat_dim=1280),
            SDLayerSpec(index=2, name="up_2", feat_dim=640),
            SDLayerSpec(index=3, name="up_3", feat_dim=320),
        )

        def __init__(
            self,
            *,
            model_id: str,
            time_step: int,
            layers_to_hook: Sequence[Union[str, int]],
            primary_layer: Optional[Union[str, int]] = None,
            device: torch.device,
        ) -> None:
            super().__init__()
            self.device = device
            self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            self.time_step = int(time_step)
            self.patch_size = 16

            self.layer_lookup: Dict[str, SDLayerSpec] = {spec.name: spec for spec in self.LAYER_SPECS}
            self.layer_lookup.update({str(spec.index): spec for spec in self.LAYER_SPECS})

            self.layer_order = self._resolve_layers(layers_to_hook)
            if primary_layer is None:
                self.primary_layer = self.layer_order[-1]
            else:
                self.primary_layer = self._resolve_single(primary_layer)

            self.featurizer = SDFeaturizer(
                model_id=model_id,
                device=device,
                torch_dtype=self.dtype,
            )

            mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32)[:, None, None]
            std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32)[:, None, None]
            self.register_buffer("imagenet_mean", mean, persistent=False)
            self.register_buffer("imagenet_std", std, persistent=False)

        def _resolve_single(self, layer: Union[str, int]) -> str:
            if isinstance(layer, int):
                idx = layer if layer >= 0 else len(self.LAYER_SPECS) + layer
                if idx < 0 or idx >= len(self.LAYER_SPECS):
                    raise ValueError(f"Stable Diffusion layer index {layer} is out of bounds.")
                return self.LAYER_SPECS[idx].name
            name = str(layer)
            if name not in self.layer_lookup:
                raise ValueError(f"Unknown Stable Diffusion layer identifier '{layer}'.")
            return name

        def _resolve_layers(self, layers: Sequence[Union[str, int]]) -> List[str]:
            resolved: List[str] = []
            seen: set[str] = set()
            for layer in layers:
                name = self._resolve_single(layer)
                if name not in seen:
                    resolved.append(name)
                    seen.add(name)
            if not resolved:
                raise ValueError("layers_to_hook must specify at least one Stable Diffusion layer.")
            return resolved

        @torch.no_grad()
        def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> OrderedDict[str, torch.Tensor]:
            images = images.to(self.device, dtype=torch.float32)
            _, _, height, width = images.shape

            pixels = (images * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
            pipeline_input = (pixels * 2.0 - 1.0).to(self.dtype)

            layer_indices = [self.layer_lookup[name].index for name in self.layer_order]
            feats = self.featurizer(
                pipeline_input,
                prompts=prompts,
                timestep=self.time_step,
                up_ft_indices=layer_indices,
            )

            grid_h = height // self.patch_size
            grid_w = width // self.patch_size

            outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
            for name in self.layer_order:
                idx = self.layer_lookup[name].index
                feature = feats[idx].to(torch.float32)
                feature = torch.nn.functional.interpolate(
                    feature,
                    size=(grid_h, grid_w),
                    mode="bilinear",
                    align_corners=False,
                )
                outputs[name] = feature
            return outputs


    class StableDiffusionExtractor(BaseExtractor):
        def build_model(self) -> torch.nn.Module:
            model_id = self.spec.model_id or "stabilityai/stable-diffusion-2-1"
            time_step = self.spec.time_step or 250
            layer = self.spec.primary_layer or "up_3"
            self._sd_prompt = self.spec.prompt or ""
            device = torch.device(self.spec.device)
            backbone = StableDiffusionBackbone(
                model_id=model_id,
                time_step=time_step,
                layers_to_hook=[layer],
                primary_layer=layer,
                device=device,
            )
            self._sd_layer = layer
            return backbone

        def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
            with torch.inference_mode():
                outputs = self.model(tensor, prompts=[self._sd_prompt])
            feature = outputs[self._sd_layer].squeeze(0)
            C, Hp, Wp = feature.shape
            tokens = feature.permute(1, 2, 0).reshape(Hp * Wp, C)
            return tokens.cpu().numpy(), Hp, Wp

else:

    class StableDiffusionExtractor(BaseExtractor):
        def build_model(self) -> torch.nn.Module:
            raise ImportError("diffusers is required to use Stable Diffusion models in Section2.")

        def encode_tokens(self, tensor: torch.Tensor, meta: ResizeMeta) -> Tuple[np.ndarray, int, int]:
            raise RuntimeError("Stable Diffusion extractor is unavailable because diffusers is missing.")


EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {
    "dinov3": Dinov3Extractor,
    "dino": DinoExtractor,
    "dinov2": Dinov2Extractor,
    "sam": SamExtractor,
    "clip": OpenCLIPExtractor,
    "siglip": SiglipExtractor,
    "stable_diffusion": StableDiffusionExtractor,
}


def create_extractor(spec: ModelSpec) -> BaseExtractor:
    extractor_cls = EXTRACTOR_REGISTRY.get(spec.type)
    if extractor_cls is None:
        raise KeyError(f"No extractor registered for type '{spec.type}'")
    return extractor_cls(spec)
