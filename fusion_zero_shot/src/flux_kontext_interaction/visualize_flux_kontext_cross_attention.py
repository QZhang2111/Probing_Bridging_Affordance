#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_flux_kontext_cross_attention.py

Capture and visualize cross-attention in the FLUX.1 Kontext pipeline.
During execution this script will:
  - replace the FluxAttention processor in 19 dual-stream transformer blocks
  - record softmax probabilities from image queries to text keys
  - export per-token and per-layer heatmaps
  - save attention arrays and tokenizer outputs for downstream analysis

Example:
    python visualize_flux_kontext_cross_attention.py \
        --image_path ./axe_000692.jpg \
        --prompt "Maintain the original axe scene while hands firmly grip the handle preparing to chop." \
        --output_root ./outputs \
        --num_steps 20 --guidance 2.5 --seed 3
"""

# --- Torch<2.5 compatibility patch (kept consistent with other scripts) ---
import re
import torch
from torch.nn import functional as F


def _torch_version_lt_25(ver: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", ver)
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) < (2, 5)


if _torch_version_lt_25(torch.__version__):
    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_wrapper(*args, **kwargs):
        kwargs.pop("enable_gqa", None)  # Older versions do not support this argument.
        return _orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _sdpa_wrapper

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
# --- end patch ---

import os
import json
import math
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diffusers import FluxKontextPipeline  # noqa: E402
from diffusers.models.embeddings import apply_rotary_emb  # noqa: E402

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


# ========= Utility Functions =========
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def best_hw(n: int) -> Tuple[int, int]:
    root = int(math.sqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def pick_preferred_resolution(
    target_hw: Tuple[int, int],
    candidates: List[Tuple[int, int]] = PREFERED_KONTEXT_RESOLUTIONS,
) -> Tuple[int, int]:
    """
    Select the closest resolution from Kontext's recommended list.
    Prioritize aspect ratio first, then area.
    """
    tgt_h, tgt_w = target_hw
    if tgt_h <= 0 or tgt_w <= 0:
        raise ValueError(f"Invalid target size {target_hw}")
    tgt_ratio = tgt_w / tgt_h
    tgt_area = tgt_h * tgt_w

    def metric(hw: Tuple[int, int]) -> Tuple[float, float]:
        h, w = hw
        ratio = w / h
        area = h * w
        return (abs(ratio - tgt_ratio), abs(area - tgt_area))

    best = min(candidates, key=metric)
    return best


def sanitize_token(tok: str) -> str:
    t = tok.replace("/", "_").replace("\\", "_").replace(" ", "_")
    if len(t) > 24:
        t = t[:24]
    return t


def tokens_to_grid(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Map a 1D token sequence back to a 2D grid.
    coords can be (B, N, 3) or (N, 3); the last two channels are row/col.
    By convention, first channel == 0 means primary image tokens; other values
    (e.g., 1) indicate additional conditioning tokens.
    """
    if coords.ndim == 3:
        coords = coords[0]
    if coords.shape[-1] != 3:
        raise ValueError(f"Unexpected coord shape {coords.shape}")

    base_mask = coords[:, 0] == 0
    base_coords = coords[base_mask]
    if base_coords.shape[0] < values.shape[0]:
        logging.debug(
            "[coords] base tokens %d smaller than values %d, using prefix only",
            base_coords.shape[0],
            values.shape[0],
        )
    elif base_coords.shape[0] > values.shape[0]:
        base_coords = base_coords[: values.shape[0]]

    if base_coords.size == 0:
        logging.warning("Empty base_coords; fallback to 1D reshape.")
        side = int(math.sqrt(values.shape[0]))
        side = max(side, 1)
        return values.reshape(side, values.shape[0] // side if values.shape[0] % side == 0 else side)

    H = int(base_coords[:, 1].max()) + 1
    W = int(base_coords[:, 2].max()) + 1

    heat = np.zeros((H, W), dtype=np.float32)
    for idx in range(min(values.shape[0], base_coords.shape[0])):
        r = int(base_coords[idx, 1])
        c = int(base_coords[idx, 2])
        if r < H and c < W:
            heat[r, c] = values[idx]
    return heat


# ========= Global Recording Buffers =========
ATTN_LOG: List[Dict[str, Any]] = []
EXPECTED_LATENT_HW: Optional[Tuple[int, int]] = None
LAST_IMG_IDS: Optional[torch.Tensor] = None


# ========= Custom Processor =========
class FluxKontextAttnRecorderProcessor:
    """
    Override the official FluxAttnProcessor to:
      - capture attention probabilities from image queries to text keys
      - preserve the original return signature and pipeline behavior
    """

    _attention_backend = None  # Keep behavior aligned with the official processor.

    def __init__(self, tag: str = ""):
        self.tag = tag

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # hidden_states: (B, N_img, C)
        B, N_img, _ = hidden_states.shape
        H = attn.heads
        Dh = attn.head_dim

        image_seq_len_attr = getattr(attn, "image_seq_len", None)
        img_seq_len_attr = getattr(attn, "img_seq_len", None)

        if not hasattr(self, "_debug_logged"):
            logging.debug(
                "[debug] attn=%s hidden_states=%s encoder_hidden_states=%s image_seq_len=%s img_seq_len=%s kwargs=%s",
                attn.__class__.__name__,
                tuple(hidden_states.shape),
                tuple(encoder_hidden_states.shape) if encoder_hidden_states is not None else None,
                image_seq_len_attr,
                img_seq_len_attr,
                list(kwargs.keys()),
            )
            if encoder_hidden_states is not None:
                logging.debug(
                    "[debug] q_enc_len=%d q_img_len=%d",
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1],
                )
            self._debug_logged = True

        # ----- 1) Linear Projections -----
        def proj_unflat(x: torch.Tensor, proj_layer: torch.nn.Linear) -> torch.Tensor:
            y = proj_layer(x)
            return y.unflatten(-1, (H, -1))

        q = proj_unflat(hidden_states, attn.to_q)  # (B, N_img, H, Dh)
        k = proj_unflat(hidden_states, attn.to_k)
        v = proj_unflat(hidden_states, attn.to_v)

        has_enc = encoder_hidden_states is not None and attn.added_kv_proj_dim is not None
        if has_enc:
            q_enc = proj_unflat(encoder_hidden_states, attn.add_q_proj)
            k_enc = proj_unflat(encoder_hidden_states, attn.add_k_proj)
            v_enc = proj_unflat(encoder_hidden_states, attn.add_v_proj)

        # ----- 2) RMSNorm -----
        q = attn.norm_q(q)
        k = attn.norm_k(k)
        if has_enc:
            q_enc = attn.norm_added_q(q_enc)
            k_enc = attn.norm_added_k(k_enc)

        # ----- 3) Concatenate Text/Image Streams -----
        if has_enc:
            q_all = torch.cat([q_enc, q], dim=1)  # (B, N_txt + N_img, H, Dh)
            k_all = torch.cat([k_enc, k], dim=1)
            v_all = torch.cat([v_enc, v], dim=1)
            N_txt = q_enc.shape[1]
        else:
            q_all, k_all, v_all = q, k, v
            N_txt = 0

        # ----- 4) RoPE -----
        if image_rotary_emb is not None:
            q_all = apply_rotary_emb(q_all, image_rotary_emb, sequence_dim=1)
            k_all = apply_rotary_emb(k_all, image_rotary_emb, sequence_dim=1)

        # ----- 5) Attention and Output -----
        def to_bhnd(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 1, 3).contiguous()

        q_bhnd = to_bhnd(q_all)
        k_bhnd = to_bhnd(k_all)
        v_bhnd = to_bhnd(v_all)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q_bhnd, k_bhnd.transpose(-1, -2)) * scale  # (B, H, N_q, N_k)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1)  # (B, H, N_q, N_k)
        out_bhnd = torch.matmul(probs, v_bhnd)
        out_bn_hd = out_bhnd.permute(0, 2, 1, 3).contiguous().view(B, q_all.shape[1], H * Dh)
        out_bn_hd = out_bn_hd.to(hidden_states.dtype)

        if has_enc:
            enc_out_part = out_bn_hd[:, :N_txt]
            img_out_part = out_bn_hd[:, N_txt:]

            img_out = attn.to_out[0](img_out_part)
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out_part)

            probs_text = probs[:, :, N_txt:, :N_txt].detach().to(torch.float32).cpu()
            qsum_allkeys = probs[:, :, N_txt:, :].sum(-1).detach().to(torch.float32).cpu()

            valid_len = None
            img_seq_len_kw = kwargs.get("img_seq_len")
            if img_seq_len_kw is not None:
                valid_len = int(img_seq_len_kw)
            elif isinstance(img_seq_len_attr, int):
                valid_len = int(img_seq_len_attr)
            elif isinstance(image_seq_len_attr, int):
                valid_len = int(image_seq_len_attr)

            if valid_len is not None:
                while valid_len > probs_text.shape[2] and valid_len % 2 == 0:
                    valid_len //= 2
                if valid_len > probs_text.shape[2]:
                    valid_len = probs_text.shape[2]
                if valid_len < probs_text.shape[2]:
                    logging.debug(
                        "[trim] keeping first %d tokens out of %d for image branch",
                        valid_len,
                        probs_text.shape[2],
                    )
                probs_text = probs_text[:, :, :valid_len]
                qsum_allkeys = qsum_allkeys[:, :, :valid_len]

            ATTN_LOG.append(
                {
                    "tag": self.tag,
                    "shape": list(probs.shape),
                    "n_txt": int(N_txt),
                    "probs_text": probs_text,
                    "qsum_allkeys": qsum_allkeys,
                }
            )
            return img_out, enc_out

        return out_bn_hd


# ========= Hook Installation =========
def attach_recorder_to_dual_blocks(pipe: FluxKontextPipeline):
    for idx, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.attn.set_processor(FluxKontextAttnRecorderProcessor(tag=f"dual-{idx:02d}"))


# ========= Token Logging =========
def dump_tokens_t5(pipe: FluxKontextPipeline, prompt: str, out_dir: str) -> List[str]:
    enc = pipe.tokenizer_2(prompt, add_special_tokens=True, return_attention_mask=False)
    ids = enc["input_ids"]
    toks = pipe.tokenizer_2.convert_ids_to_tokens(ids)
    token_map = [{"id": i, "tid": tid, "tok": tok} for i, (tid, tok) in enumerate(zip(ids, toks))]
    with open(os.path.join(out_dir, "tokens_t5.json"), "w", encoding="utf-8") as f:
        json.dump(token_map, f, ensure_ascii=False, indent=2)
    for t in token_map:
        logging.info(f"[tokenizer_2] {t['id']:02d} | {t['tid']:5d} | {t['tok']}")
    return toks


# ========= Heatmap Generation =========
def _save_heatmaps(
    img: np.ndarray,
    P: torch.Tensor,
    tokens: List[str],
    out_dir: str,
    coords: Optional[np.ndarray] = None,
    ignore_tokens=None,
):
    if ignore_tokens is None:
        ignore_tokens = {"▁", "</s>"}

    os.makedirs(out_dir, exist_ok=True)

    N_img = P.shape[0]
    keep_idx = [i for i, tok in enumerate(tokens) if tok not in ignore_tokens]
    if not keep_idx:
        logging.warning("No valid tokens to visualize.")
        return

    P = P[:, keep_idx]
    tokens_kept = [tokens[i] for i in keep_idx]

    for t_idx, tok in enumerate(tokens_kept):
        values = P[:, t_idx].cpu().numpy()

        if coords is not None:
            heat = tokens_to_grid(values, coords)
        else:
            global EXPECTED_LATENT_HW
            if EXPECTED_LATENT_HW and EXPECTED_LATENT_HW[0] * EXPECTED_LATENT_HW[1] == N_img:
                H_lat, W_lat = EXPECTED_LATENT_HW
            else:
                H_lat, W_lat = best_hw(N_img)
                EXPECTED_LATENT_HW = (H_lat, W_lat)
            heat = values.reshape(H_lat, W_lat)
        mn, mx = float(heat.min()), float(heat.max())
        heat_n = np.zeros_like(heat) if mx - mn < 1e-12 else (heat - mn) / (mx - mn)

        heat_img = Image.fromarray((heat_n * 255).astype(np.uint8), mode="L")
        heat_up = np.array(heat_img.resize((img.shape[1], img.shape[0]), Image.BILINEAR)) / 255.0
        cmap = (plt.cm.jet(heat_up)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.6 * img + 0.4 * cmap).astype(np.uint8)

        tok_name = sanitize_token(tok)
        Image.fromarray((heat_up * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"heat_tok{t_idx:02d}_{tok_name}.png")
        )
        Image.fromarray(overlay).save(
            os.path.join(out_dir, f"overlay_tok{t_idx:02d}_{tok_name}.png")
        )


def make_token_maps_from_attnlog(
    img_path: str,
    tokens: List[str],
    out_dir: str,
    coords: Optional[np.ndarray] = None,
    avg_over: Tuple[str, ...] = ("head", "layer"),
    ignore_tokens=None,
):
    assert len(ATTN_LOG) > 0 and ATTN_LOG[0]["probs_text"].ndim == 4

    img = np.array(Image.open(img_path).convert("RGB"))

    P_mean, Q_mean = None, None
    total = len(ATTN_LOG)

    for rec in ATTN_LOG:
        P = rec["probs_text"].float()  # (B, H, N_img, N_txt)
        Q = rec["qsum_allkeys"].float()
        if "head" in avg_over:
            P = P.mean(dim=1)
            Q = Q.mean(dim=1)
        if P_mean is None:
            P_mean = P / total
            Q_mean = Q / total
        else:
            P_mean += P / total
            Q_mean += Q / total

    P = P_mean[0]
    Q = Q_mean[0]

    dev = (Q - 1.0).abs()
    logging.info(
        "[sanity] sum over all keys |sum-1| -> mean %.6f, max %.6f",
        float(dev.mean()),
        float(dev.max()),
    )

    os.makedirs(out_dir, exist_ok=True)
    _save_heatmaps(img, P, tokens, out_dir, coords=coords, ignore_tokens=ignore_tokens)

    np.savez_compressed(
        os.path.join(out_dir, "attn_avg_layers_imgtxt.npz"),
        attn_avg_imgtxt=P.cpu().numpy(),
        qsum_allkeys=Q.cpu().numpy(),
        tokens=np.array(tokens, dtype=object),
    )


def make_per_layer_stepavg_heatmaps(
    img_path: str,
    tokens: List[str],
    root_out_dir: str,
    coords: Optional[np.ndarray] = None,
    ignore_tokens=None,
):
    assert len(ATTN_LOG) > 0 and ATTN_LOG[0]["probs_text"].ndim == 4

    img = np.array(Image.open(img_path).convert("RGB"))
    per_layer_dir = os.path.join(root_out_dir, "per_layer")
    os.makedirs(per_layer_dir, exist_ok=True)

    agg_P: Dict[str, torch.Tensor] = {}
    agg_Q: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for rec in ATTN_LOG:
        tag = rec["tag"]
        P = rec["probs_text"].float().mean(dim=1)[0]
        Q = rec["qsum_allkeys"].float().mean(dim=1)[0]

        if tag not in counts:
            agg_P[tag] = P.clone()
            agg_Q[tag] = Q.clone()
            counts[tag] = 1
        else:
            n = counts[tag]
            agg_P[tag] += (P - agg_P[tag]) / (n + 1)
            agg_Q[tag] += (Q - agg_Q[tag]) / (n + 1)
            counts[tag] = n + 1

    for tag in sorted(agg_P.keys()):
        layer_dir = os.path.join(per_layer_dir, tag)
        _save_heatmaps(img, agg_P[tag], tokens, layer_dir, coords=coords, ignore_tokens=ignore_tokens)
        logging.info("[viz] layer %s averaged over %d steps -> %s", tag, counts[tag], layer_dir)


# ========= Main Pipeline =========
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Flux Kontext cross-attention visualizer")
    parser.add_argument("--model_dir", type=str, default="/home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev")
    parser.add_argument("--image_path", type=str, default="./wine_glass_003375.jpg")
    parser.add_argument(
        "--prompt",
        type=str,
        #  default=(
        #     "Maintain the original scene with the axe resting on the block, while a pair of hands firmly "
        #     "grip the handle preparing to chop."
        # )
        default=(
            "sip"
        ),
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="./kontext_attn_outputs")
    parser.add_argument("--save_per_layer", action="store_true", default=False)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--match_input_resolution",
        action="store_true",
        help="Automatically select the closest Kontext resolution to the input image size (used only when height/width are not explicitly set).",
    )
    args = parser.parse_args()

    assert os.path.exists(args.image_path), f"Image not found: {args.image_path}"

    set_seed(args.seed)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt)[:80]
    out_dir = os.path.join(args.output_root, f"{safe_prompt}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "log.txt"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Load FluxKontextPipeline from %s", args.model_dir)
    pipe = FluxKontextPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    pipe = pipe.to(args.device)

    global EXPECTED_LATENT_HW
    EXPECTED_LATENT_HW = None
    ATTN_LOG.clear()

    image = Image.open(args.image_path).convert("RGB")
    if args.match_input_resolution and (args.height is None or args.width is None):
        orig_w, orig_h = image.size
        matched_h, matched_w = pick_preferred_resolution((orig_h, orig_w))
        args.height = matched_h
        args.width = matched_w
        logging.info(
            "match_input_resolution=True -> use preferred size %dx%d (original %dx%d)",
            matched_h,
            matched_w,
            orig_h,
            orig_w,
        )

    tokens = dump_tokens_t5(pipe, args.prompt, out_dir)

    attach_recorder_to_dual_blocks(pipe)

    global LAST_IMG_IDS
    LAST_IMG_IDS = None
    orig_forward = pipe.transformer.forward

    def forward_wrapper(*args, **kwargs):
        global LAST_IMG_IDS
        if "img_ids" in kwargs and kwargs["img_ids"] is not None:
            LAST_IMG_IDS = kwargs["img_ids"].detach().to(torch.float32).cpu()
        return orig_forward(*args, **kwargs)

    pipe.transformer.forward = forward_wrapper

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    logging.info("Run pipeline ...")
    pipe_kwargs = dict(
        prompt=args.prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    if args.height is not None:
        pipe_kwargs["height"] = args.height
    if args.width is not None:
        pipe_kwargs["width"] = args.width

    try:
        output = pipe(**pipe_kwargs)
    finally:
        pipe.transformer.forward = orig_forward

    if LAST_IMG_IDS is None:
        logging.warning("Failed to capture img_ids; heatmaps will use heuristic reshape.")
        coords_np = None
    else:
        coords_np = LAST_IMG_IDS.numpy()
        logging.info("Captured img_ids shape %s", tuple(coords_np.shape))
        try:
            np.save(os.path.join(out_dir, "img_ids.npy"), coords_np)
            logging.info("Saved img_ids to %s", os.path.join(out_dir, "img_ids.npy"))
        except Exception:
            logging.exception("Failed to save img_ids array")

    result_image = output.images[0]
    result_path = os.path.join(out_dir, "gen.png")
    result_image.save(result_path)
    logging.info("Saved edited image to %s", result_path)

    per_token_dir = os.path.join(out_dir, "per_token")
    logging.info("Create per-token heatmaps...")
    make_token_maps_from_attnlog(result_path, tokens, per_token_dir, coords=coords_np)

    if args.save_per_layer:
        logging.info("Create per-layer heatmaps...")
        make_per_layer_stepavg_heatmaps(result_path, tokens, out_dir, coords=coords_np)

    logging.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
