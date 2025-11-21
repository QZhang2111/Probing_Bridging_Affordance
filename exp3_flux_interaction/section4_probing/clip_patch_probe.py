#!/usr/bin/env python3
"""
Patch-level CLIP probing utility.

Given an image and one or more prompts, this script loads the requested CLIP
checkpoint (default: hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K), encodes the
image into patch embeddings, encodes each prompt into a text embedding, and
computes cosine similarity per patch.  Each prompt produces:

- a Viridis heatmap (upsampled to the image size),
- an overlay on top of the RGB image, and
- the raw similarity map saved as .npy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from contextlib import contextmanager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=str, default="laion/CLIP-ViT-B-16-laion2B-s34B-b88K", help="Alias for reference; only used in metadata.")
    parser.add_argument("--arch", type=str, default="ViT-B-16", help="OpenCLIP architecture, e.g. ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="OpenCLIP pretrained tag, e.g. laion2b_s34b_b88k")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input RGB image.")
    parser.add_argument("--prompts", nargs="+", required=True, help="One or more text prompts to probe.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")
    parser.add_argument("--output-root", type=Path, default=Path("clip_patch_outputs"), help="Directory to save results.")
    parser.add_argument("--feat-source", choices=("token", "value"), default="value", help="Patch descriptor type: transformer output token or value vectors.")
    parser.add_argument("--layer-index", type=int, default=-1, help="Transformer block index to sample (supports negative indices).")
    parser.add_argument("--force-size", type=int, default=224, help="Resize image to this square resolution before padding (e.g. 224). Use 0/omit to keep original size.")
    return parser.parse_args()


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_model(arch: str, pretrained: str, device: str):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        arch,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(arch)
    if hasattr(model.visual, "output_tokens"):
        model.visual.output_tokens = True
    model.eval()
    return model, tokenizer


def _infer_patch_hw(model) -> tuple[int, int]:
    patch = getattr(model.visual, "patch_size", 16)
    if isinstance(patch, (tuple, list)):
        return (int(patch[0]), int(patch[1]))
    return (int(patch), int(patch))


def _pad_to_patch_multiple(image: Image.Image, patch_hw: tuple[int, int], source_size: tuple[int, int] | None = None) -> tuple[Image.Image, dict]:
    orig_w, orig_h = image.size
    if source_size is None:
        src_w, src_h = orig_w, orig_h
    else:
        src_w, src_h = source_size
    target_w = math.ceil(orig_w / patch_hw[1]) * patch_hw[1]
    target_h = math.ceil(orig_h / patch_hw[0]) * patch_hw[0]
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(image, (0, 0))
    meta = {
        "source_w": src_w,
        "source_h": src_h,
        "proc_w": orig_w,
        "proc_h": orig_h,
        "target_w": target_w,
        "target_h": target_h,
    }
    return canvas, meta


def _resize_positional_embedding(visual, new_grid: tuple[int, int]) -> torch.Tensor:
    pos_embed = visual.positional_embedding
    cls = pos_embed[:1]
    patch_pos = pos_embed[1:]
    base_tokens = patch_pos.shape[0]
    base_size = int(round(base_tokens ** 0.5))
    patch_pos = patch_pos.reshape(1, base_size, base_size, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=new_grid,
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], -1)
    resized = torch.cat([cls.unsqueeze(0), patch_pos], dim=1)
    return resized.squeeze(0)


@contextmanager
def patched_visual(visual, target_hw: tuple[int, int], patch_hw: tuple[int, int]):
    grid = (target_hw[0] // patch_hw[0], target_hw[1] // patch_hw[1])
    original_pos = visual.positional_embedding
    original_grid = getattr(visual, "grid_size", None)
    base_size = int(round((original_pos.shape[0] - 1) ** 0.5))
    base_grid = original_grid or (base_size, base_size)

    if grid == base_grid:
        yield
        return

    resized = _resize_positional_embedding(visual, grid).to(original_pos.device, dtype=original_pos.dtype)

    try:
        with torch.no_grad():
            visual.positional_embedding = torch.nn.Parameter(resized, requires_grad=False)
            visual.grid_size = grid
        yield
    finally:
        with torch.no_grad():
            visual.positional_embedding = original_pos
            visual.grid_size = original_grid or base_grid


def _extract_block_value(block, x: torch.Tensor) -> torch.Tensor:
    ln_in = block.ln_1(x)
    attn = block.attn
    embed_dim = ln_in.shape[-1]
    if getattr(attn, "_qkv_same_embed_dim", True):
        weight = attn.in_proj_weight
        bias = attn.in_proj_bias
        v_weight = weight[2 * embed_dim :, :]
        v_bias = bias[2 * embed_dim :] if bias is not None else None
    else:
        v_weight = attn.v_proj_weight
        v_bias = attn.v_proj_bias
    return F.linear(ln_in, v_weight, v_bias)[:, 1:, :]


def _resolve_layer_index(num_blocks: int, layer_index: int) -> int:
    idx = layer_index if layer_index >= 0 else num_blocks + layer_index
    if idx < 0 or idx >= num_blocks:
        raise ValueError(f"Layer index {layer_index} resolves to {idx}, outside [0, {num_blocks}).")
    return idx


def encode_image_patches(model, image: Image.Image, device: str, feat_source: str, layer_index: int, force_size: int | None):
    patch_hw = _infer_patch_hw(model)
    original_size = image.size
    working = image
    resized = None if force_size in (None, 0) else force_size
    if resized is not None:
        working = image.resize((resized, resized), Image.BICUBIC)
    padded, meta = _pad_to_patch_multiple(working, patch_hw, source_size=original_size)
    tensor = TF.to_tensor(padded).unsqueeze(0).to(device)
    tensor = TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)
    target_hw = (tensor.shape[-2], tensor.shape[-1])
    blocks = model.visual.transformer.resblocks
    target_layer = _resolve_layer_index(len(blocks), layer_index)

    with torch.no_grad():
        with patched_visual(model.visual, target_hw, patch_hw):
            proj = getattr(model.visual, "proj", None)
            if proj is None:
                raise AttributeError("visual.proj not found; cannot project patch tokens to embedding space.")

            x = model.visual._embeds(tensor)
            x = model.visual.patch_dropout(x)
            x = model.visual.ln_pre(x)
            patch_tokens = None
            for idx, block in enumerate(blocks):
                if feat_source == "value" and idx == target_layer:
                    patch_tokens = _extract_block_value(block, x)
                    break
                x = block(x)
                if feat_source == "token" and idx == target_layer:
                    patch_tokens = x[:, 1:, :]
                    break
            if patch_tokens is None:
                raise RuntimeError("Failed to capture transformer features; check layer index and source type.")

        patch_proj = torch.matmul(patch_tokens, proj)
        patch_norm = patch_proj / patch_proj.norm(dim=-1, keepdim=True)
    return patch_norm.squeeze(0), meta, patch_hw, target_layer


def encode_texts(model, tokenizer, prompts: List[str], device: str):
    import torch

    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds


def save_heatmaps(sim_grid: np.ndarray, prompts: List[str], base_image: Image.Image, pad_meta: dict, out_dir: Path):
    import matplotlib.cm as cm

    out_dir.mkdir(parents=True, exist_ok=True)
    base_np = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    proc_h = pad_meta.get("proc_h", pad_meta.get("orig_h"))
    proc_w = pad_meta.get("proc_w", pad_meta.get("orig_w"))
    source_h = pad_meta.get("source_h", proc_h)
    source_w = pad_meta.get("source_w", proc_w)
    target_h = pad_meta["target_h"]
    target_w = pad_meta["target_w"]

    for idx, name in enumerate(prompts):
        grid = sim_grid[idx]
        grid = grid - grid.min()
        denom = grid.max()
        norm = grid / denom if denom > 1e-12 else grid
        heat = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").resize((target_w, target_h), Image.BICUBIC)
        heat_np = np.asarray(heat, dtype=np.float32) / 255.0
        heat_np = heat_np[:proc_h, :proc_w]
        if (proc_h, proc_w) != (source_h, source_w):
            heat_np = np.asarray(
                Image.fromarray((heat_np * 255.0).astype(np.uint8), mode="L").resize((source_w, source_h), Image.BICUBIC)
            ) / 255.0
        colored = cm.get_cmap("jet")(heat_np)[..., :3]
        overlay = (0.6 * colored + 0.4 * base_np).clip(0.0, 1.0)

        slug = name.replace(" ", "_")[:32]
        Image.fromarray((colored * 255).astype(np.uint8)).save(out_dir / f"{slug}_heat.png")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / f"{slug}_overlay.png")
        np.save(out_dir / f"{slug}_heat.npy", heat_np.astype(np.float32))


def main() -> None:
    args = parse_args()
    out_dir = args.output_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.arch, args.pretrained, args.device)
    image = Image.open(args.image).convert("RGB")

    patch_embeds, pad_meta, patch_hw, used_layer = encode_image_patches(
        model,
        image,
        args.device,
        args.feat_source,
        args.layer_index,
        args.force_size,
    )
    text_embeds = encode_texts(model, tokenizer, args.prompts, args.device)

    import torch

    with torch.no_grad():
        sims = torch.matmul(text_embeds, patch_embeds.T)
        sims = sims.cpu().numpy()

    Hp = pad_meta["target_h"] // patch_hw[0]
    Wp = pad_meta["target_w"] // patch_hw[1]
    num_patches = Hp * Wp
    if patch_embeds.shape[0] != num_patches:
        raise ValueError(
            f"Patch count mismatch: embeddings {patch_embeds.shape[0]} vs grid {num_patches} (H={Hp}, W={Wp})."
        )
    sim_grids = sims.reshape(len(args.prompts), Hp, Wp)

    save_heatmaps(sim_grids, args.prompts, image, pad_meta, out_dir / "attention")

    image.save(out_dir / "source.png")
    meta = {
        "model_id": args.model_id,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "image": str(args.image),
        "prompts": args.prompts,
        "device": args.device,
        "grid_size": [Hp, Wp],
        "pad_meta": pad_meta,
        "patch_hw": patch_hw,
        "feat_source": args.feat_source,
        "layer_index": used_layer,
        "force_size": args.force_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
