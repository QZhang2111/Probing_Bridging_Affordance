#!/usr/bin/env python3
"""
Cross-attention visualiser for Flux/Stable Diffusion img2img pipelines.

Given an RGB image and a prompt, the script
  1. encodes the image into latents,
  2. runs the requested diffusion pipeline with the provided prompt,
  3. records the true cross-attention probabilities between image queries
     and the selected text tokens, and
  4. exports Viridis heatmaps / overlays per token.

Only code under section4_probing/ is touched; the rest of the repo remains
untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

# Deferred heavy imports (torch/diffusers) happen inside `main`.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sd", "flux"), required=True, help="Choose Stable Diffusion or Flux backend.")
    parser.add_argument("--model-id", type=str, required=True, help="HF repo id or local path to the pipeline weights.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the reference RGB image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt whose tokens will be probed.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--tokens", nargs="*", default=None, help="Whitespace-separated list of token strings to track (defaults to every token).")
    parser.add_argument("--output-root", type=Path, default=Path("probe_outputs"), help="Root directory for results.")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance", type=float, default=2.5, help="Classifier-free guidance scale.")
    parser.add_argument("--strength", type=float, default=0.7, help="Img2Img strength (only SD uses it; Flux ignores).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")
    return parser.parse_args()


class AttentionAccumulator:
    """Collects per-token attention maps and handles visualisation."""

    def __init__(self, token_map: Dict[str, int]):
        self.token_map = token_map
        self.storage: Dict[str, List[np.ndarray]] = {name: [] for name in token_map}

    def _infer_hw(self, length: int) -> Optional[int]:
        side = int(round(math.sqrt(length)))
        if side * side == length:
            return side
        return None

    def add_from_probs(self, probs, token_dim: int, layer_tag: str):
        """
        Args:
            probs: torch.Tensor (B, H, N_query, N_key)
            token_dim: number of tokens (N_key)
        """
        import torch

        with torch.no_grad():
            data = probs.detach().float().mean(dim=1)  # (B, Nq, Nk)
            data = data[0]  # batch size 1
            hw = self._infer_hw(data.shape[0])
            if hw is None:
                return
            attn_grid = data.view(hw, hw, token_dim)
            for name, idx in self.token_map.items():
                if idx >= token_dim:
                    continue
                plane = attn_grid[..., idx].cpu().numpy()
                self.storage[name].append(plane)

    def summary(self) -> Dict[str, np.ndarray]:
        maps = {}
        for name, planes in self.storage.items():
            if not planes:
                continue
            stack = np.stack(planes, axis=0)
            maps[name] = stack.mean(axis=0)
        return maps

    def export(self, maps: Dict[str, np.ndarray], base_image: Image.Image, out_dir: Path):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        out_dir.mkdir(parents=True, exist_ok=True)
        base_rgb = base_image.convert("RGB")
        base_np = np.asarray(base_rgb, dtype=np.float32) / 255.0

        for name, arr in maps.items():
            if arr.size == 0:
                continue
            arr = arr - arr.min()
            denom = arr.max()
            norm = arr / denom if denom > 1e-8 else arr
            heat_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").resize(
                base_rgb.size, Image.BICUBIC
            )
            heat = np.asarray(heat_img, dtype=np.float32) / 255.0
            colored = cm.get_cmap("viridis")(heat)[..., :3]
            overlay = (0.65 * colored + 0.35 * base_np).clip(0.0, 1.0)

            Image.fromarray((colored * 255).astype(np.uint8)).save(out_dir / f"{name}_heat.png")
            Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / f"{name}_overlay.png")
            np.save(out_dir / f"{name}_heat.npy", heat)


def collect_token_indices(tokenizer, prompt: str, target_tokens: Optional[Sequence[str]]) -> Dict[str, int]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
    if target_tokens:
        wanted = list(target_tokens)
    else:
        wanted = tokens

    token_map: Dict[str, int] = {}
    used = set()
    for want in wanted:
        want_norm = want.lower()
        idx = None
        for i, tok in enumerate(tokens):
            key = tok.lower()
            if key == want_norm or want_norm in key:
                if (want_norm, i) in used:
                    continue
                idx = i
                used.add((want_norm, i))
                break
        if idx is not None:
            token_map[want] = idx
    return token_map


class SDRecordingProcessor:
    """Wraps AttnProcessor2_0 to capture cross-attention."""

    def __init__(self, base_processor, accumulator: AttentionAccumulator):
        from diffusers.models.attention_processor import AttnProcessor2_0

        if not isinstance(base_processor, AttnProcessor2_0):
            raise TypeError("SDRecordingProcessor only supports AttnProcessor2_0.")
        self.base = base_processor
        self.acc = accumulator

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        import torch

        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        key_length = encoder_hidden_states.shape[1]

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        head_dim = attn.head_dim
        heads = attn.heads

        def reshape(x, seq_len):
            return x.view(batch_size, seq_len, heads, head_dim).permute(0, 2, 1, 3)

        q = reshape(q, sequence_length)
        k = reshape(k, key_length)
        v = reshape(v, key_length)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        self.acc.add_from_probs(probs, key_length, attn.__class__.__name__)

        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, heads * head_dim)
        hidden_states = context
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states + residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


class FluxAttnRecorderProcessor:
    """Adapted from Flux scripts: captures true image-query x text-key probabilities."""

    def __init__(self, accumulator: AttentionAccumulator):
        self.acc = accumulator

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        import torch
        from diffusers.models.embeddings import apply_rotary_emb

        B, N_img, _ = hidden_states.shape
        H = attn.heads
        Dh = attn.head_dim

        def proj_unflat(x, proj):
            y = proj(x)
            return y.unflatten(-1, (H, -1))

        q = proj_unflat(hidden_states, attn.to_q)
        k = proj_unflat(hidden_states, attn.to_k)
        v = proj_unflat(hidden_states, attn.to_v)

        has_enc = encoder_hidden_states is not None and attn.added_kv_proj_dim is not None
        if has_enc:
            q_enc = proj_unflat(encoder_hidden_states, attn.add_q_proj)
            k_enc = proj_unflat(encoder_hidden_states, attn.add_k_proj)
            v_enc = proj_unflat(encoder_hidden_states, attn.add_v_proj)

        q = attn.norm_q(q)
        k = attn.norm_k(k)
        if has_enc:
            q_enc = attn.norm_added_q(q_enc)
            k_enc = attn.norm_added_k(k_enc)

        if has_enc:
            q_all = torch.cat([q_enc, q], dim=1)
            k_all = torch.cat([k_enc, k], dim=1)
            v_all = torch.cat([v_enc, v], dim=1)
            N_txt = encoder_hidden_states.shape[1]
        else:
            q_all, k_all, v_all = q, k, v
            N_txt = 0

        if image_rotary_emb is not None:
            q_all = apply_rotary_emb(q_all, image_rotary_emb, sequence_dim=1)
            k_all = apply_rotary_emb(k_all, image_rotary_emb, sequence_dim=1)

        def to_bhnd(x):
            return x.permute(0, 2, 1, 3).contiguous()

        q_bhnd = to_bhnd(q_all)
        k_bhnd = to_bhnd(k_all)
        v_bhnd = to_bhnd(v_all)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q_bhnd, k_bhnd.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)

        if has_enc and N_txt > 0:
            img_start = N_txt
            img_probs = probs[:, :, img_start:, :N_txt]
            self.acc.add_from_probs(img_probs, N_txt, "flux-block")

        out = torch.matmul(probs, v_bhnd)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, q_all.shape[1], H * Dh)
        out = out.to(hidden_states.dtype)

        if has_enc:
            img_part = out[:, N_txt:]
            enc_part = out[:, :N_txt]
            img_out = attn.to_out[0](img_part)
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_part)
            return img_out, enc_out
        img_out = attn.to_out[0](out)
        img_out = attn.to_out[1](img_out)
        return img_out


def run_sd(pipe, args: argparse.Namespace, accumulator: AttentionAccumulator, image: Image.Image, out_dir: Path):
    import torch

    processors = {}
    for name, proc in pipe.unet.attn_processors.items():
        if proc is None:
            processors[name] = proc
            continue
        processors[name] = SDRecordingProcessor(proc, accumulator)
    pipe.unet.set_attn_processor(processors)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
    )
    result.images[0].save(out_dir / "generated.png")


def _iter_flux_attention_modules(transformer):
    """
    Supports both older Flux pipelines that expose ``inner_transformer.blocks``
    and newer versions where the blocks live directly under
    ``transformer.transformer_blocks`` (or ``blocks``).
    """

    candidate_attrs = [
        "inner_transformer",
        "transformer_blocks",
        "single_transformer_blocks",
        "blocks",
        "_repeated_blocks",
    ]
    blocks = None
    for attr in candidate_attrs:
        module = getattr(transformer, attr, None)
        if module is None:
            continue
        if hasattr(module, "blocks"):
            blocks = module.blocks
            break
        if isinstance(module, (list, tuple)):
            blocks = module
            break
        try:
            import torch.nn as nn
        except ImportError:
            nn = None
        if nn is not None and isinstance(module, nn.ModuleList):
            blocks = list(module)
            break
        if hasattr(module, "__iter__"):
            blocks = list(module)
            break
    if blocks is None:
        raise AttributeError("Unable to locate Flux transformer blocks for attention hooking.")

    for block in blocks:
        for name in ("attn", "attn1", "attn2"):
            attn = getattr(block, name, None)
            if attn is None or not hasattr(attn, "processor"):
                continue
            added_dim = getattr(attn, "added_kv_proj_dim", None)
            if added_dim is None or added_dim == 0:
                continue
            if hasattr(attn, "set_processor"):
                yield attn



def run_flux(pipe, args: argparse.Namespace, accumulator: AttentionAccumulator, image: Image.Image, out_dir: Path):
    import torch

    for attn in _iter_flux_attention_modules(pipe.transformer):
        attn.set_processor(FluxAttnRecorderProcessor(accumulator))

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
    )
    result.images[0].save(out_dir / "generated.png")


def main() -> None:
    args = parse_args()
    out_dir = args.output_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_image = Image.open(args.image).convert("RGB")

    import torch

    if args.backend == "sd":
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(args.device)
        tokenizer = pipe.tokenizer
    else:
        from diffusers import FluxImg2ImgPipeline

        pipe = FluxImg2ImgPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
        ).to(args.device)
        tokenizer = pipe.tokenizer

    token_map = collect_token_indices(tokenizer, args.prompt, args.tokens)
    if not token_map:
        raise RuntimeError("No matching tokens found for tracking.")

    accumulator = AttentionAccumulator(token_map)

    if args.backend == "sd":
        run_sd(pipe, args, accumulator, base_image, out_dir)
    else:
        run_flux(pipe, args, accumulator, base_image, out_dir)

    maps = accumulator.summary()
    accumulator.export(maps, base_image, out_dir / "attention")

    meta = {
        "backend": args.backend,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "tokens_tracked": token_map,
        "steps": args.steps,
        "guidance": args.guidance,
        "strength": args.strength,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
