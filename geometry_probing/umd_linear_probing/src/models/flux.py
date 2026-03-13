"""Flux.1-dev backbone for dense linear probing."""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

# -----------------------------------------------------------------------------
# Locate the Flux reference implementation shipped with the workspace.
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[5]
_FLUX_SRC = _WORKSPACE_ROOT / "export" / "flux" / "src"
if _FLUX_SRC.exists():
    sys.path.append(str(_FLUX_SRC))
else:
    raise FileNotFoundError(
        "Expected Flux reference implementation under 'export/flux/src'."
    )

from flux.model import Flux, FluxParams  # type: ignore
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams  # type: ignore
def _optionally_expand_state_dict(model: nn.Module, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    for name, param in model.named_parameters():
        if name in state_dict and state_dict[name].shape != param.shape:
            expanded = torch.zeros_like(param, device=state_dict[name].device)
            slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
            expanded[slices] = state_dict[name]
            state_dict[name] = expanded
    return state_dict


def _ensure_even(value: int) -> int:
    if value % 2 != 0:
        raise ValueError(f"Expected an even spatial dimension; received {value}.")
    return value


def _rearrange_img_tokens(latents: Tensor) -> Tensor:
    b, c, h, w = latents.shape
    h_half = _ensure_even(h) // 2
    w_half = _ensure_even(w) // 2
    latents = latents.reshape(b, c, h_half, 2, w_half, 2)
    latents = latents.permute(0, 2, 4, 3, 5, 1).reshape(b, h_half * w_half, 4 * c)
    return latents


def _tokens_to_grid(tokens: Tensor, grid_hw: Tuple[int, int]) -> Tensor:
    b, tokens_count, c = tokens.shape
    h, w = grid_hw
    if tokens_count != h * w:
        raise ValueError(
            f"Token count {tokens_count} does not match target grid {grid_hw}."
        )
    return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _make_img_ids(height: int, width: int, batch: int, device: torch.device) -> Tensor:
    h_half = _ensure_even(height) // 2
    w_half = _ensure_even(width) // 2
    ids = torch.zeros(h_half, w_half, 3, device=device, dtype=torch.float32)
    ids[..., 1] += torch.arange(h_half, device=device)[:, None]
    ids[..., 2] += torch.arange(w_half, device=device)[None, :]
    ids = ids.reshape(1, h_half * w_half, 3).repeat(batch, 1, 1)
    return ids


def _make_txt_ids(length: int, batch: int, device: torch.device) -> Tensor:
    return torch.zeros(batch, length, 3, device=device, dtype=torch.float32)


@dataclass(frozen=True)
class FluxBackboneConfig:
    params: FluxParams
    ae_params: AutoEncoderParams


class FluxBackbone(nn.Module):
    """Frozen Flux.1-dev backbone returning dense patch-level features."""

    DEFAULT = FluxBackboneConfig(
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    )

    def __init__(
        self,
        *,
        model_dir: Union[str, Path],
        flow_checkpoint: Optional[Union[str, Path]] = None,
        ae_checkpoint: Optional[Union[str, Path]] = None,
        layers_to_hook: Optional[Sequence[Union[str, int]]] = None,
        primary_layer: Optional[str] = None,
        time_step: float = 1.0,
        guidance_strength: float = 1.0,
        disable_guidance: bool = False,
        device: str = "cuda",
        max_pixels_per_forward: int = 786_432,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.time_step = float(time_step)
        self.guidance_strength = float(guidance_strength)
        self.disable_guidance = bool(disable_guidance)
        self.max_pixels_per_forward = int(max(1, max_pixels_per_forward))

        base_dir = Path(model_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Flux model directory not found: {base_dir}")

        flow_path = Path(flow_checkpoint) if flow_checkpoint else base_dir / "flux1-dev.safetensors"
        if not flow_path.is_absolute():
            flow_path = base_dir / flow_path
        ae_path = Path(ae_checkpoint) if ae_checkpoint else base_dir / "ae.safetensors"
        if not ae_path.is_absolute():
            ae_path = base_dir / ae_path
        if not flow_path.exists() or not ae_path.exists():
            raise FileNotFoundError("Missing Flux checkpoints (flow or autoencoder).")

        with torch.device("meta"):
            self.transformer = Flux(self.DEFAULT.params)
            self.autoencoder = AutoEncoder(self.DEFAULT.ae_params)

        transformer_sd = load_safetensors(str(flow_path), device=str(self.device))
        transformer_sd = _optionally_expand_state_dict(self.transformer, transformer_sd)
        missing, unexpected = self.transformer.load_state_dict(transformer_sd, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load Flux transformer weights. "
                f"Missing keys: {missing[:3]}..., unexpected: {unexpected[:3]}..."
            )

        ae_sd = load_safetensors(str(ae_path), device=str(self.device))
        missing_ae, unexpected_ae = self.autoencoder.load_state_dict(ae_sd, strict=False, assign=True)
        if missing_ae or unexpected_ae:
            raise RuntimeError(
                "Failed to load Flux autoencoder weights. "
                f"Missing keys: {missing_ae[:3]}..., unexpected: {unexpected_ae[:3]}..."
            )

        self.transformer.to(self.device, dtype=self.dtype).eval()
        # Optionally disable guidance branch at runtime to approximate "off"
        if self.disable_guidance:
            try:
                # This flag controls whether guidance is consumed in forward()
                self.transformer.params.guidance_embed = False  # type: ignore[attr-defined]
            except Exception:
                pass
        self.autoencoder.to(self.device, dtype=self.dtype).eval()
        for module in (self.transformer, self.autoencoder):
            for param in module.parameters():
                param.requires_grad_(False)

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(str(base_dir / "tokenizer"))
        self.clip_text_encoder = CLIPTextModel.from_pretrained(
            str(base_dir / "text_encoder"), torch_dtype=self.dtype
        ).to(self.device)
        self.clip_text_encoder.eval()
        for p in self.clip_text_encoder.parameters():
            p.requires_grad_(False)

        self.t5_tokenizer = T5Tokenizer.from_pretrained(str(base_dir / "tokenizer_2"))
        self.t5_encoder = T5EncoderModel.from_pretrained(
            str(base_dir / "text_encoder_2"), torch_dtype=self.dtype
        ).to(self.device)
        self.t5_encoder.eval()
        for p in self.t5_encoder.parameters():
            p.requires_grad_(False)

        blank_txt, blank_vec = self._encode_prompts([""])
        self.register_buffer("blank_txt", blank_txt, persistent=False)
        self.register_buffer("blank_vec", blank_vec, persistent=False)
        self.register_buffer(
            "blank_txt_ids",
            _make_txt_ids(blank_txt.shape[1], blank_txt.shape[0], self.device),
            persistent=False,
        )

        self.available_layers: List[str] = [
            *(f"double_{i}" for i in range(self.transformer.params.depth)),
            *(f"single_{i}" for i in range(self.transformer.params.depth_single_blocks)),
        ]
        self.layer_order = self._resolve_layers(layers_to_hook or ["single_37"])
        self.primary_layer = self._resolve_single(primary_layer) if primary_layer else self.layer_order[-1]

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None],
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None],
            persistent=False,
        )

        self._feature_cache: Dict[str, Tensor] = {}
        self._current_txt_len: int = 0
        self._register_hooks()

    def _register_hooks(self) -> None:
        requested_layers = set(self.layer_order)
        requested_layers.add(self.primary_layer)

        def double_hook(name: str):
            def _hook(_: nn.Module, __, output):
                img_tokens, _ = output
                self._feature_cache[name] = img_tokens.detach()

            return _hook

        def single_hook(name: str):
            def _hook(_: nn.Module, __, output):
                img_tokens = output[:, self._current_txt_len :, :]
                self._feature_cache[name] = img_tokens.detach()

            return _hook

        for idx, block in enumerate(self.transformer.double_blocks):
            name = f"double_{idx}"
            if name in requested_layers:
                block.register_forward_hook(double_hook(name))
        for idx, block in enumerate(self.transformer.single_blocks):
            name = f"single_{idx}"
            if name in requested_layers:
                block.register_forward_hook(single_hook(name))

    def _resolve_single(self, layer: Union[str, int]) -> str:
        if isinstance(layer, str):
            if layer not in self.available_layers:
                raise ValueError(f"Unknown Flux layer '{layer}'.")
            return layer
        total = len(self.available_layers)
        idx = total + layer if layer < 0 else layer
        if idx < 0 or idx >= total:
            raise ValueError(f"Layer index {layer} resolves to {idx}, outside [0, {total}).")
        return self.available_layers[idx]

    def _resolve_layers(self, layers: Sequence[Union[str, int]]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for entry in layers:
            resolved = self._resolve_single(entry)
            if resolved not in seen:
                ordered.append(resolved)
                seen.add(resolved)
        if not ordered:
            raise ValueError("layers_to_hook must contain at least one valid entry.")
        return ordered

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> Tuple[Tensor, Tensor]:
        clip_inputs = self.clip_tokenizer(
            list(prompts),
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        clip_outputs = self.clip_text_encoder(
            input_ids=clip_inputs["input_ids"].to(self.device),
            attention_mask=None,
        )
        pooled = clip_outputs.pooler_output.to(dtype=self.dtype)

        t5_inputs = self.t5_tokenizer(
            list(prompts),
            truncation=True,
            max_length=self.t5_tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        hidden = self.t5_encoder(
            input_ids=t5_inputs["input_ids"].to(self.device),
            attention_mask=None,
        ).last_hidden_state.to(dtype=self.dtype)
        return hidden, pooled

    def _prepare_text(
        self,
        batch_size: int,
        prompts: Optional[Sequence[str]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if prompts is None or len(prompts) == 0:
            txt = self.blank_txt.repeat(batch_size, 1, 1)
            vec = self.blank_vec.repeat(batch_size, 1)
            txt_ids = self.blank_txt_ids.repeat(batch_size, 1, 1)
            return txt, vec, txt_ids
        txt, vec = self._encode_prompts(prompts)
        txt_ids = _make_txt_ids(txt.shape[1], txt.shape[0], self.device)
        return txt, vec, txt_ids

    def _denormalize(self, images: Tensor) -> Tensor:
        return images * self.imagenet_std + self.imagenet_mean

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        autocast_precision: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> OrderedDict[str, Tensor]:
        if categories is not None and prompts is not None:
            raise ValueError("Specify either categories or prompts, not both.")
        if categories is not None:
            prompts = [f"a photo of a {cat}" for cat in categories]
        prompt_list = list(prompts) if prompts is not None else None

        images = images.to(self.device, dtype=torch.float32)
        if images.numel() == 0:
            raise ValueError("Received empty image chunk.")
        batch = images.shape[0]
        if prompt_list is not None and len(prompt_list) != batch:
            raise ValueError(
                f"Received {len(prompt_list)} prompts for batch size {batch}; lengths must match."
            )

        return self._forward_chunk(
            images,
            autocast_precision=autocast_precision,
            prompts=prompt_list,
        )

    @torch.no_grad()
    def _forward_chunk(
        self,
        images: Tensor,
        *,
        autocast_precision: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> OrderedDict[str, Tensor]:
        """Runs Flux in memory-friendly chunks over the batch dimension.

        Aligns behavior with Stable Diffusion backbone:
        - assumes images are ImageNet-normalized, denormalizes back to pixels
        - encodes with the autoencoder to latents
        - rearranges latents to patch tokens and builds positional ids
        - encodes prompts (T5 hidden + CLIP pooled) or uses blanks
        - forwards the transformer once, collecting requested layers via hooks
        - reshapes tokens back to dense grids at H/16 x W/16 and returns float32
        """
        device = self.device
        dtype = self.dtype

        batch, _, height, width = images.shape
        # Target grid matches SD backbone convention
        target_hw = (height // 16, width // 16)

        # Compute a batch chunk size based on pixel budget (no spatial tiling)
        pixels_per_image = int(height * width)
        max_images = max(1, self.max_pixels_per_forward // max(1, pixels_per_image))

        # Helper to build outputs incrementally
        ordered_keys: List[str] = [self.primary_layer, *[k for k in self.layer_order if k != self.primary_layer]]
        collected_batches: Dict[str, list[Tensor]] = {key: [] for key in ordered_keys}

        for start in range(0, batch, max_images):
            end = min(batch, start + max_images)
            imgs = images[start:end]

            # 1) Denormalize to pixel space, clamp to [0,1], then map to [-1,1]
            #    Align AE input range with common VAE/AE training conventions
            pixels = (self._denormalize(imgs).clamp(0.0, 1.0) * 2.0 - 1.0).to(device)

            # 2) AE encode to latents (B, C=16, H/8, W/8) in bfloat16
            latents = self.autoencoder.encode(pixels.to(dtype))

            # 3) Rearrange latents to image tokens: (B, H/16*W/16, 4*C=64)
            img_tokens = _rearrange_img_tokens(latents)

            # 4) Prepare text conditioning and ids
            if prompts is None:
                txt, vec, txt_ids = self._prepare_text(img_tokens.shape[0], None)
            else:
                txt, vec, txt_ids = self._prepare_text(img_tokens.shape[0], prompts[start:end])
            # Derive ids from latent spatial size to ensure consistency:
            # latents: (B, C, H_lat, W_lat) -> token grid is (H_lat/2, W_lat/2)
            h_lat, w_lat = int(latents.shape[-2]), int(latents.shape[-1])
            # _make_img_ids expects an input 'height' whose half defines the token grid height.
            # Therefore pass (H_lat, W_lat) so that h_half=H_lat//2, w_half=W_lat//2 match tokens.
            img_ids = _make_img_ids(h_lat, w_lat, img_tokens.shape[0], device)

            # 5) Timesteps and guidance
            timesteps = torch.full((img_tokens.shape[0],), float(self.time_step), device=device, dtype=dtype)
            if getattr(self.transformer.params, "guidance_embed", False):
                guidance = torch.full(
                    (img_tokens.shape[0],), float(self.guidance_strength), device=device, dtype=dtype
                )
            else:
                guidance = None

            # 6) Run transformer once; hooks collect requested features
            self._feature_cache.clear()
            self._current_txt_len = int(txt.shape[1])
            _ = self.transformer(
                img=img_tokens,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=vec,
                guidance=guidance,
            )

            # 7) Convert collected tokens to dense grids, align to H/16 x W/16
            for key in ordered_keys:
                if key not in self._feature_cache:
                    # Skip missing keys silently; upstream layer config ensures presence
                    continue
                tokens = self._feature_cache[key]  # (B, T, C)
                # Grid size implied by tokens/latents
                grid_hw_tokens = (h_lat // 2, w_lat // 2)
                grid = _tokens_to_grid(tokens, grid_hw_tokens)  # (B, C, H_t, W_t)
                if grid.shape[-2:] != target_hw:
                    grid = F.interpolate(grid, size=target_hw, mode="bilinear", align_corners=False)
                collected_batches[key].append(grid.to(torch.float32))

        # 8) Concatenate along batch and return ordered dict
        outputs = OrderedDict()
        for key in ordered_keys:
            if collected_batches[key]:
                outputs[key] = torch.cat(collected_batches[key], dim=0)
        return outputs
