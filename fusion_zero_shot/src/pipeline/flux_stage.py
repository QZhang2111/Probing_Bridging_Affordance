from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import ensure_dir, run_command, sanitize_token

# Paths will be supplied by caller; this module just exposes helpers.

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


def pick_preferred_resolution(
    target_hw: Tuple[int, int],
    candidates: Sequence[Tuple[int, int]] = PREFERED_KONTEXT_RESOLUTIONS,
) -> Tuple[int, int]:
    """
    Select the closest Kontext recommended resolution for the input size.
    Returns (height, width).
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

    return min(candidates, key=metric)


def run_flux_stage(
    python_exec: str,
    script_path: Path,
    image_path: Path,
    prompt: str,
    negative_prompt: str | None,
    output_root: Path,
    model_dir: Path,
    num_steps: int,
    guidance: float,
    seed: int,
    *,
    match_input_resolution: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Path:
    ensure_dir(output_root)
    existing = {p for p in output_root.glob("*") if p.is_dir()}

    if match_input_resolution and (height is None or width is None):
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
        matched_h, matched_w = pick_preferred_resolution((orig_h, orig_w))
        height = matched_h
        width = matched_w
        print(
            f"[flux] match_input_resolution -> use preferred size {matched_h}x{matched_w} "
            f"(original {orig_h}x{orig_w})"
        )

    cmd = [
        python_exec,
        str(script_path),
        "--model_dir",
        str(model_dir),
        "--image_path",
        str(image_path),
        "--prompt",
        prompt,
        "--output_root",
        str(output_root),
        "--num_steps",
        str(num_steps),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
    ]
    if height is not None:
        cmd += ["--height", str(height)]
    if width is not None:
        cmd += ["--width", str(width)]
    if negative_prompt:
        cmd.extend(
            [
                "--negative_prompt",
                negative_prompt,
            ]
        )
    run_command(cmd)

    new_dirs = [p for p in output_root.glob("*") if p.is_dir() and p not in existing]
    if not new_dirs:
        raise RuntimeError("Flux stage did not create a new output directory.")
    run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
    print(f"[flux] run_dir={run_dir}")
    return run_dir


def load_img_ids(run_dir: Path) -> Optional[np.ndarray]:
    path = run_dir / "img_ids.npy"
    if path.exists():
        return np.load(path)
    return None


def load_attn_npz(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    path = run_dir / "per_token" / "attn_avg_layers_imgtxt.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


def tokens_to_grid(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if coords.ndim == 3:
        coords = coords[0]
    coords = coords.astype(np.float32)
    num = min(coords.shape[0], values.shape[0])
    coords = coords[:num]
    values = values[:num]

    if coords.shape[1] < 3:
        raise ValueError("img_ids expected to have at least 3 columns")

    channel = coords[:, 0]
    unique = np.unique(channel.astype(int))
    if unique.size > 1:
        counts = [(np.sum(channel == u), u) for u in unique]
        primary = max(counts)[1]
        mask = channel == primary
        if mask.sum() > 0 and mask.sum() < values.shape[0]:
            coords = coords[mask]
            values = values[mask]

    rows = coords[:, 1]
    cols = coords[:, 2]
    H = int(rows.max()) + 1
    W = int(cols.max()) + 1
    heat = np.zeros((H, W), dtype=np.float32)
    for val, r, c in zip(values, rows, cols):
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            heat[rr, cc] = val
    return heat


def normalize_heat(heat: np.ndarray) -> np.ndarray:
    mn = float(heat.min())
    mx = float(heat.max())
    if mx - mn < 1e-12:
        return np.zeros_like(heat, dtype=np.float32)
    return (heat - mn) / (mx - mn)


def resize_to_generated(heat: np.ndarray, gen_size: Tuple[int, int]) -> np.ndarray:
    gen_w, gen_h = gen_size
    return cv2.resize(heat, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)


def letterbox_to_original(heat_gen: np.ndarray, gen_size: Tuple[int, int], orig_size: Tuple[int, int]) -> np.ndarray:
    gen_w, gen_h = gen_size
    orig_w, orig_h = orig_size
    scale = min(gen_w / orig_w, gen_h / orig_h)
    content_w = max(1, int(round(orig_w * scale)))
    content_h = max(1, int(round(orig_h * scale)))
    pad_x = int(round((gen_w - content_w) / 2.0))
    pad_y = int(round((gen_h - content_h) / 2.0))
    pad_x = max(0, min(gen_w - 1, pad_x))
    pad_y = max(0, min(gen_h - 1, pad_y))
    pad_x2 = max(pad_x + 1, min(gen_w, pad_x + content_w))
    pad_y2 = max(pad_y + 1, min(gen_h, pad_y + content_h))
    crop = heat_gen[pad_y:pad_y2, pad_x:pad_x2]
    if crop.size == 0:
        crop = heat_gen
    heat_orig = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return heat_orig.astype(np.float32)


def compute_direct_heatmaps(
    run_dir: Path,
    original_image_path: Path,
    target_tokens: Iterable[str],
) -> Dict[str, np.ndarray]:
    img_ids = load_img_ids(run_dir)
    attn_npz = load_attn_npz(run_dir)
    if img_ids is None or attn_npz is None:
        return {}

    attn = attn_npz["attn_avg_imgtxt"]
    txt_tokens = [str(t) for t in attn_npz["tokens"]]
    token_map = {tok: idx for idx, tok in enumerate(txt_tokens)}

    gen_img = Image.open(run_dir / "gen.png")
    gen_size = gen_img.size
    orig_img = Image.open(original_image_path)
    orig_size = orig_img.size

    result: Dict[str, np.ndarray] = {}
    attn_vals = attn
    for tok in target_tokens:
        if tok not in token_map:
            continue
        values = attn_vals[:, token_map[tok]]
        heat_lat = tokens_to_grid(values, img_ids)
        heat_norm = normalize_heat(heat_lat)
        heat_gen = resize_to_generated(heat_norm, gen_size)
        heat_orig = letterbox_to_original(heat_gen, gen_size, orig_size)
        result[tok] = np.clip(heat_orig, 0.0, 1.0)

    return result


def load_tokens(tokens_path: Path) -> Tuple[list[str], list[Dict[str, object]]]:
    with tokens_path.open("r", encoding="utf-8") as fh:
        token_map = json.load(fh)
    tokens = [entry["tok"] for entry in token_map]
    return tokens, token_map


def pick_token(tokens: Sequence[str], candidates: Iterable[str]) -> Tuple[int, str]:
    lowered = [(idx, tok, tok.lower()) for idx, tok in enumerate(tokens)]
    for candidate in candidates:
        cand_norm = candidate.lower()
        for idx, tok, tok_lower in lowered:
            if tok == candidate or tok_lower == cand_norm:
                return idx, tok
    # fallback: match by stripped sub-token belonging to candidate
    def _clean(s: str) -> str:
        return s.replace("▁", "").replace("_", "").lower()

    stripped_tokens = [
        (idx, tok, _clean(tok))
        for idx, tok in enumerate(tokens)
    ]
    for candidate in candidates:
        cand_clean = _clean(candidate)
        if not cand_clean:
            continue
        for idx, tok, tok_clean in stripped_tokens:
            if tok_clean and tok_clean in cand_clean:
                return idx, tok
    raise ValueError(f"Unable to find token among candidates {candidates}")


def locate_heatmap(per_token_dir: Path, idx: int, token: str) -> Path:
    name = f"heat_tok{idx:02d}_{sanitize_token(token)}.png"
    path = per_token_dir / name
    if path.exists():
        return path

    sanitized = sanitize_token(token)

    def parse_index(p: Path) -> int:
        prefix = p.name.split("_")[0]
        try:
            return int(prefix.replace("heat_tok", ""))
        except ValueError:
            return 10_000

    token_matches = sorted(
        [p for p in per_token_dir.glob("heat_tok*.png") if sanitized in p.stem or token.strip("_") in p.stem],
        key=lambda p: (abs(parse_index(p) - idx), p.name),
    )
    if token_matches:
        alt = token_matches[0]
        print(f"[warn] {path.name} missing; matched by token -> {alt.name}")
        return alt

    fallback = sorted(per_token_dir.glob(f"heat_tok{idx:02d}_*.png"))
    if fallback:
        alt = fallback[0]
        print(f"[warn] {path.name} missing; fallback -> {alt.name}")
        return alt

    if idx > 0:
        prev = idx - 1
        prev_candidates = sorted(per_token_dir.glob(f"heat_tok{prev:02d}_*.png"))
        for candidate in prev_candidates:
            if sanitize_token(token) in candidate.name or token.strip("_") in candidate.name:
                print(f"[warn] using neighbouring heatmap {candidate.name} for token {token}")
                return candidate
        if prev_candidates:
            print(f"[warn] fallback to neighbouring index {prev:02d} -> {prev_candidates[0].name}")
            return prev_candidates[0]

    raise FileNotFoundError(path)


def warp_heatmap(
    python_exec: str,
    script_path: Path,
    original: Path,
    edited: Path,
    heatmap: Path,
    out_dir: Path,
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    cmd = [
        python_exec,
        str(script_path),
        "--original",
        str(original),
        "--edited",
        str(edited),
        "--heatmap",
        str(heatmap),
        "--out_dir",
        str(out_dir),
    ]
    run_command(cmd)

    stem = heatmap.stem
    warped = out_dir / f"{stem}_on_original.png"
    overlay = out_dir / f"{stem}_overlay_on_original.png"
    affine = out_dir / f"{stem}_affine.npy"
    return {"heat": warped, "overlay": overlay, "affine": affine}
