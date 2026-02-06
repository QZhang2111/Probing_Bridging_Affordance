"""Shared DINO feature extraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from dino.src.settings import get_settings

from ..common.fs import append_jsonl, ensure_dir
from ..common.image import ResizeMeta, resize_letterbox_to
from ..common.io import save_tokens_npz
from ..common.tensor import to_tensor_norm

DEFAULT_MODEL_NAME = "dinov3_vit7b16"
PATCH_SIZE = 16


@dataclass
class ExtractionSpec:
    target_size: Tuple[int, int]
    patch_size: int = PATCH_SIZE
    output_path: Optional[Path] = None
    overwrite: bool = False


def load_dinov3(model_name: str = DEFAULT_MODEL_NAME) -> torch.nn.Module:
    """Load the DINOv3 model specified in the repo settings."""

    settings = get_settings()
    model = torch.hub.load(
        repo_or_dir=str(settings.model_repo),
        model=model_name,
        source="local",
        pretrained=False,
    )
    checkpoint_path = settings.require_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    prefixes = ("module.", "backbone.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for pref in prefixes:
            if new_key.startswith(pref):
                new_key = new_key[len(pref) :]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (expected for head)")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")
    model.eval().cuda()
    return model


def extract_last_tokens(model: torch.nn.Module, img_tensor: torch.Tensor) -> tuple[np.ndarray, int, int]:
    """Return (tokens, Hp, Wp) as float32 numpy arrays."""

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        blocks = getattr(model, "blocks", None)
        last_idx = len(blocks) - 1 if blocks is not None else -1
        feats = model.get_intermediate_layers(img_tensor.unsqueeze(0).cuda(), n=[last_idx], reshape=True, norm=True)
    x = feats[0].squeeze(0).cpu().float()  # [C, H_p, W_p]
    C, Hp, Wp = x.shape
    tokens = x.view(C, Hp * Wp).permute(1, 0).contiguous().numpy()
    return tokens, Hp, Wp


class FeatureExtractor:
    """High-level helper for exporting DINO dense tokens and manifests."""

    def __init__(self, model: Optional[torch.nn.Module] = None, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_dinov3(self._model_name)
        return self._model

    def extract_image(
        self,
        image_path: Path,
        target_wh: Tuple[int, int],
        patch_size: int = PATCH_SIZE,
    ) -> tuple[np.ndarray, int, int, ResizeMeta]:
        """Load and preprocess an image, returning tokens and resizing metadata."""

        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
        letterbox, meta = resize_letterbox_to(img_rgb, target_wh, patch_size)
        tensor = to_tensor_norm(letterbox)
        tokens, Hp, Wp = extract_last_tokens(self.model, tensor)
        return tokens, Hp, Wp, meta

    def export_image(
        self,
        image_path: Path,
        spec: ExtractionSpec,
        manifest_path: Optional[Path] = None,
        extra_manifest_fields: Optional[Dict[str, object]] = None,
    ) -> Path:
        """Extract tokens for one image and save them to disk.

        Returns the output path used.
        """

        if spec.output_path is not None:
            out_path = spec.output_path
        else:
            out_path = self._default_output_path(image_path, spec.target_size)
        out_path = out_path.expanduser().resolve()

        if out_path.exists() and not spec.overwrite:
            print(f"[skip] output exists: {out_path}")
            return out_path

        tokens, Hp, Wp, meta = self.extract_image(image_path, spec.target_size, spec.patch_size)
        grid_meta = dict(
            H_patches=Hp,
            W_patches=Wp,
            patch_size=spec.patch_size,
            resized_h=meta.final_h,
            resized_w=meta.final_w,
            model=self._model_name,
            preprocess="imagenet_meanstd",
            source_path=str(image_path),
            **meta.as_dict(),
        )
        save_tokens_npz(tokens, grid_meta, out_path)
        print(f"[ok] saved tokens to {out_path} shape={tokens.shape}")

        if manifest_path is not None:
            entry = {
                "image_path": str(image_path),
                "tokens_path": str(out_path),
                "target_w": spec.target_size[0],
                "target_h": spec.target_size[1],
                "patch_size": spec.patch_size,
                "model": self._model_name,
            }
            if extra_manifest_fields:
                entry.update(extra_manifest_fields)
            append_jsonl(entry, manifest_path)
        return out_path

    def _default_output_path(self, image: Path, target_wh: Tuple[int, int]) -> Path:
        settings = get_settings()
        cache_root = settings.paths.get("cache_root")
        if cache_root is None:
            raise RuntimeError("cache_root not configured; update configs/defaults.yaml")
        w, h = target_wh
        template = cache_root / "tokens" / "manual" / f"{image.stem}.vit7b16.{w}x{h}.last.npz"
        ensure_dir(template.parent)
        return template
