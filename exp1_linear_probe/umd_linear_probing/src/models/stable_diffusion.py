"""Stable Diffusion 2.1 backbone for dense linear probing."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
try:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
except ImportError:  # diffusers < 0.30 fallback
    from diffusers import UNet2DConditionModel  # type: ignore
from torch import Tensor, nn
from torch.nn.functional import interpolate


class MyUNet2DConditionModel(UNet2DConditionModel):
    """UNet that exposes up-block features (adapted from DIFT)."""

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        up_ft_indices: Sequence[int],
        encoder_hidden_states: Tensor,
    ) -> Dict[str, Dict[int, Tensor]]:
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

        collected: Dict[int, Tensor] = {}
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


class OneStepSDPipeline(StableDiffusionPipeline):
    """Pipeline that returns UNet up-block activations in a single denoising call."""

    def __call__(
        self,
        img_tensor: Tensor,
        timestep: Tensor,
        up_ft_indices: Sequence[int],
        prompt_embeds: Optional[Tensor] = None,
    ) -> Dict[str, Dict[int, Tensor]]:
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


class SDFeaturizer(nn.Module):
    """Wraps Stable Diffusion Pipeline to expose intermediate UNet activations."""

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

        modules = [self.text_encoder, self.unet, self.vae]
        for module in modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def encode_prompt(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_outputs = self.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
        return text_outputs[0].to(dtype=self.dtype)

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        prompts: Sequence[str],
        timestep: int,
        up_ft_indices: Sequence[int],
    ) -> Dict[int, Tensor]:
        device = images.device
        prompt_embeds = self.encode_prompt(prompts, device=device)

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(images).latent_dist.mode()

        t = torch.tensor(timestep, dtype=torch.long, device=device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        output = self.unet(
            latents_noisy,
            t,
            up_ft_indices=up_ft_indices,
            encoder_hidden_states=prompt_embeds,
        )
        return output["up_ft"]


@dataclass(frozen=True)
class SDLayerSpec:
    index: int
    name: str
    feat_dim: int


class StableDiffusionBackbone(nn.Module):
    """Stable Diffusion 2.1 backbone that matches the linear probe contract."""

    LAYER_SPECS: Sequence[SDLayerSpec] = (
        SDLayerSpec(index=0, name="up_0", feat_dim=1280),
        SDLayerSpec(index=1, name="up_1", feat_dim=1280),
        SDLayerSpec(index=2, name="up_2", feat_dim=640),
        SDLayerSpec(index=3, name="up_3", feat_dim=320),
    )

    def __init__(
        self,
        *,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        time_step: int = 250,
        layers_to_hook: Optional[Sequence[Union[str, int]]] = None,
        primary_layer: Optional[Union[str, int]] = None,
        output_mode: str = "dense",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if output_mode not in {"dense", "gap"}:
            raise ValueError("output_mode must be 'dense' or 'gap'.")

        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.time_step = int(time_step)
        self.output_mode = output_mode
        self.patch_size = 16

        self.layer_lookup: Dict[str, SDLayerSpec] = {spec.name: spec for spec in self.LAYER_SPECS}
        self.layer_lookup.update({str(spec.index): spec for spec in self.LAYER_SPECS})

        self.layer_order = self._resolve_layers(layers_to_hook or ["up_3"])
        self.primary_layer = self._resolve_single(primary_layer) if primary_layer is not None else self.layer_order[-1]

        self.featurizer = SDFeaturizer(
            model_id=model_id,
            device=self.device,
            torch_dtype=self.dtype,
        )

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

    def _resolve_single(self, layer: Union[str, int]) -> str:
        if isinstance(layer, int):
            idx = layer if layer >= 0 else len(self.LAYER_SPECS) + layer
            if idx < 0 or idx >= len(self.LAYER_SPECS):
                raise ValueError(f"Layer index {layer} is out of bounds for Stable Diffusion backbone.")
            return self.LAYER_SPECS[idx].name
        key = layer
        if key not in self.layer_lookup:
            raise ValueError(f"Unknown Stable Diffusion layer identifier: {layer}")
        return self.layer_lookup[key].name

    def _resolve_layers(self, layers: Sequence[Union[str, int]]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for layer in layers:
            name = self._resolve_single(layer)
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        if not ordered:
            raise ValueError("layers_to_hook must contain at least one valid layer identifier.")
        return ordered

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
            prompts = [f"a photo of a {category}" for category in categories]
        if prompts is None:
            prompts = [""] * images.shape[0]

        images = images.to(self.device, dtype=torch.float32)
        batch, _, height, width = images.shape

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
        outputs: OrderedDict[str, Tensor] = OrderedDict()

        for name in self.layer_order:
            idx = self.layer_lookup[name].index
            feature = feats[idx].to(torch.float32)
            if self.output_mode == "gap":
                feature = feature.mean(dim=(2, 3))
                feature = feature.view(batch, feature.shape[1], 1, 1)
            else:
                feature = interpolate(feature, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            outputs[name] = feature

        return outputs