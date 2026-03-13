#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
import sys
from PIL import Image

from prompt_templates import PROMPT_TEMPLATES
from utils.data_iter import iter_agd20k_samples, SampleEntry
from utils.logging_utils import append_csv, save_json, setup_logging
from metrics import cal_kl, cal_sim, cal_nss
from kontext_runner import run_kontext_generation
from heatmap_warper import warp_heatmap_cli


ROOT = Path(__file__).resolve().parent
INTERACTION_DIR = ROOT.parent / "flux_kontext_interaction"
VIS_SCRIPT = INTERACTION_DIR / "visualize_flux_kontext_cross_attention.py"
WARP_SCRIPT = INTERACTION_DIR / "warp_heatmap_to_original.py"

REPO_ROOT = ROOT.parent
ZERO_SHOT_ROOT = REPO_ROOT / "pipeline"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(ZERO_SHOT_ROOT))
sys.path.append(str(REPO_ROOT / "dino"))

from pipeline.roi_stage import build_roi_mask, compute_roi_tokens, restore_to_original
from pipeline.pca_stage import extract_dino_tokens, run_pca
from pipeline.geometry_stage import generate_geometry_mask, largest_component
from pipeline.utils import save_overlay, save_colormap

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
    candidates: List[Tuple[int, int]] = PREFERED_KONTEXT_RESOLUTIONS,
) -> Tuple[int, int]:
    """
    Select the closest resolution from Kontext's recommended resolution list.
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


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_object_name(obj: str) -> str:
    return obj.replace("_", " ")


def select_token(tokens: List[Dict], candidates: List[str]) -> Optional[Dict]:
    lowered = [(tok["id"], tok["tok"]) for tok in tokens]
    for cand in candidates:
        cand_lower = cand.lower()
        matches = [entry for entry in lowered if cand_lower in entry[1].lower()]
        if matches:
            idx, tok = matches[0]
            return {"index": idx, "token": tok}
    # fallback: choose first non-special token
    for idx, tok in lowered:
        if tok not in {"</s>", "▁"}:
            return {"index": idx, "token": tok}
    return None


def sanitize_token_name(token: str) -> str:
    name = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    if len(name) > 24:
        name = name[:24]
    return name


def get_heatmap_path(per_token_dir: Path, index: int, token: Optional[str] = None) -> Path:
    if token:
        sanitized = sanitize_token_name(token)
        token_matches = sorted(per_token_dir.glob(f"heat_tok*_{sanitized}.png"))
        if token_matches:
            return token_matches[0]
    pattern = f"heat_tok{index:02d}_*.png"
    matches = list(per_token_dir.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No heatmap found for index {index} (token={token}) in {per_token_dir}"
    )


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img.astype(np.float32)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_object_token_candidates(name: str) -> List[str]:
    raw = name.replace("-", " ").strip()
    variants = {raw, raw.replace("_", " "), raw.replace("_", ""), raw}
    variants.update(part.strip() for part in raw.replace("_", " ").split() if part.strip())

    candidates: List[str] = []
    seen = set()
    for variant in variants:
        if not variant:
            continue
        forms = {
            variant,
            variant.lower(),
            variant.capitalize(),
            variant.title(),
            variant.replace(" ", ""),
            variant.lower().replace(" ", ""),
        }
        for form in forms:
            form = form.strip()
            if not form:
                continue
            for cand in (form, f"▁{form}"):
                if cand not in seen:
                    candidates.append(cand)
                    seen.add(cand)
    return candidates


def load_heat_normalized(path: Path) -> np.ndarray:
    arr = load_gray(path)
    max_val = float(arr.max())
    if max_val > 0:
        arr = arr / max_val
    return arr.astype(np.float32)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def run_geom_pipeline(
    sample: SampleEntry,
    sample_dir: Path,
    geom_cfg: Dict,
    verb_heat_path: Path,
    object_heat_path: Path,
) -> Optional[Dict]:
    try:
        geom_root = ensure_dir(sample_dir / "geom_pipeline")
        roi_dir = ensure_dir(geom_root / "stage_roi")
        pca_dir = ensure_dir(geom_root / "stage_dino")
        geom_dir = ensure_dir(geom_root / "stage_geom")
        final_dir = ensure_dir(geom_root / "stage_final")

        pca_components = int(geom_cfg.get("pca_components", 3))
        disable_object_roi = bool(geom_cfg.get("disable_object_roi", False))
        geom_soft_fusion = bool(geom_cfg.get("geom_soft_fusion", False))
        soft_lambda = float(geom_cfg.get("geom_soft_lambda", 0.65))
        soft_gamma = float(geom_cfg.get("geom_soft_gamma", 0.7))
        soft_temperature_cfg = geom_cfg.get("geom_soft_temperature", 1.15)
        if isinstance(soft_temperature_cfg, (list, tuple)):
            soft_temperature_values = [float(t) for t in soft_temperature_cfg if t is not None]
        else:
            soft_temperature_values = [float(soft_temperature_cfg)]
        if not soft_temperature_values:
            soft_temperature_values = [1.15]
        soft_temperature_arg: float | List[float]
        if len(soft_temperature_values) == 1:
            soft_temperature_arg = soft_temperature_values[0]
        else:
            soft_temperature_arg = soft_temperature_values
        soft_dirichlet = float(geom_cfg.get("geom_soft_base", 0.008))
        soft_log1p = bool(geom_cfg.get("geom_soft_log1p", False))
        geom_sigma = float(geom_cfg.get("geom_sigma", 1.2))
        geom_threshold = float(geom_cfg.get("geom_threshold", 0.55))
        attention_use_nss = bool(geom_cfg.get("geom_sim_use_nss", False))
        attention_topk_percent = float(geom_cfg.get("geom_sim_topk_percent", 10.0))
        attention_nss_weight = float(geom_cfg.get("geom_sim_nss_weight", 1.0))
        attention_topk_weight = float(geom_cfg.get("geom_sim_topk_weight", 1.0))

        cache_root = geom_cfg.get("cache_root", "./cache")
        cache_path = Path(cache_root)
        if not cache_path.is_absolute():
            cache_path = (ROOT / cache_path).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        dino_backend = str(geom_cfg.get("dino_backend", "dinov3")).lower()
        dino_target_wh = geom_cfg.get("dino_target_wh")
        dino_patch_size = geom_cfg.get("dino_patch_size")
        dino_cache_only = bool(geom_cfg.get("dino_cache_only", False))
        if dino_target_wh is None:
            dino_target_wh = (1260, 980) if dino_backend == "dinov2" else (1280, 960)
        if dino_patch_size is None:
            dino_patch_size = 14 if dino_backend == "dinov2" else 16

        artifacts = extract_dino_tokens(
            sample.image_path,
            target_wh=tuple(dino_target_wh),
            patch_size=int(dino_patch_size),
            cache_root=cache_path,
            cache_only=dino_cache_only,
        )

        if disable_object_roi:
            meta = artifacts.meta
            roi_mask_orig = np.ones((meta.orig_h, meta.orig_w), dtype=np.float32)
            roi_mask_letterbox = np.ones(
                (artifacts.Hp * artifacts.patch, artifacts.Wp * artifacts.patch), dtype=np.float32
            )
            roi_indices = np.arange(artifacts.tokens.shape[0])
            token_mask = np.ones((artifacts.Hp, artifacts.Wp), dtype=np.uint8)
            roi_info = {
                "mode": "full",
                "threshold": None,
                "percentile": None,
                "token_fraction": 1.0,
                "token_count": int(roi_indices.size),
            }
        else:
            object_heat = load_heat_normalized(object_heat_path)
            roi_mask_orig, roi_mask_letterbox, roi_info = build_roi_mask(
                object_heat,
                artifacts.meta,
                percentile=geom_cfg.get("roi_percentile", 85.0),
            )
            roi_indices, token_mask = compute_roi_tokens(
                roi_mask_letterbox,
                artifacts.Hp,
                artifacts.Wp,
                artifacts.patch,
                threshold=geom_cfg.get("token_threshold", 0.1),
            )
            roi_info["token_fraction"] = float(roi_indices.size) / max(1.0, float(artifacts.tokens.shape[0]))
            roi_info["token_count"] = int(roi_indices.size)

        Image.fromarray((roi_mask_orig * 255).astype(np.uint8), mode="L").save(roi_dir / "object_roi_mask.png")
        Image.fromarray((roi_mask_letterbox * 255).astype(np.uint8), mode="L").save(roi_dir / "object_roi_letterbox.png")
        save_overlay(sample.image_path, roi_mask_orig, roi_dir / "object_roi_overlay.png", alpha=0.35 if disable_object_roi else 0.45)
        np.save(roi_dir / "object_roi_letterbox.npy", roi_mask_letterbox.astype(np.float32))

        token_mask_vis = cv2.resize(
            token_mask.astype(np.float32),
            (artifacts.Wp * artifacts.patch, artifacts.Hp * artifacts.patch),
            interpolation=cv2.INTER_NEAREST,
        )
        token_mask_orig = restore_to_original(token_mask_vis, artifacts.meta)
        save_overlay(sample.image_path, token_mask_orig, roi_dir / "token_roi_overlay.png", alpha=0.25 if disable_object_roi else 0.35)
        Image.fromarray((np.clip(token_mask_orig, 0, 1) * 255).astype(np.uint8), mode="L").save(roi_dir / "token_roi_mask.png")
        np.save(roi_dir / "token_mask.npy", token_mask.astype(np.float32))
        save_json(roi_info, roi_dir / "roi_info.json")

        pca_outputs = run_pca(artifacts, roi_indices, num_components=pca_components)
        letterbox_rgb = np.clip(pca_outputs["letterbox_rgb"], 0.0, 1.0)
        orig_rgb = np.clip(pca_outputs["orig_rgb"], 0.0, 1.0)
        orig_full = np.clip(pca_outputs["orig_full"], 0.0, 1.0)
        Image.fromarray((letterbox_rgb * 255).astype(np.uint8)).save(pca_dir / "pca_rgb_letterbox.png")
        Image.fromarray((orig_rgb * 255).astype(np.uint8)).save(pca_dir / "pca_rgb_original.png")
        for ch in range(pca_outputs["num_components"]):
            plane = orig_full[..., ch]
            save_colormap(plane, pca_dir / f"pc{ch + 1}_colormap.png")
        np.save(pca_dir / "pca_components.npy", pca_outputs["pca_components"])
        np.save(pca_dir / "explained_variance.npy", pca_outputs["explained_variance"])
        np.save(pca_dir / "explained_variance_ratio.npy", pca_outputs["explained_variance_ratio"])
        save_json({"bounds": pca_outputs["bounds"]}, pca_dir / "bounds.json")

        verb_heat = load_heat_normalized(verb_heat_path)
        geom_outputs = generate_geometry_mask(
            orig_full,
            smooth_sigma=geom_sigma,
            binary_threshold=geom_threshold,
            verb_map=verb_heat,
            enable_soft_fusion=geom_soft_fusion,
            soft_lambda=soft_lambda,
            soft_gamma=soft_gamma,
            soft_temperature=soft_temperature_arg,
            soft_dirichlet=soft_dirichlet,
            soft_use_log1p=soft_log1p,
            max_channels=pca_components,
            use_attention_fallback=attention_use_nss,
            attention_topk_percent=attention_topk_percent,
            attention_nss_weight=attention_nss_weight,
            attention_topk_weight=attention_topk_weight,
        )
        geom_energy = geom_outputs["energy"]
        geom_mask = geom_outputs["mask"]
        Image.fromarray((geom_energy * 255).astype(np.uint8), mode="L").save(geom_dir / "geom_energy.png")
        save_colormap(geom_energy, geom_dir / "geom_energy_colormap.png")
        np.save(geom_dir / "geom_energy.npy", geom_energy.astype(np.float32))
        save_binary_mask(geom_mask, geom_dir / "geom_mask.png")
        np.save(geom_dir / "geom_mask.npy", geom_mask.astype(np.float32))
        save_overlay(sample.image_path, geom_mask, geom_dir / "geom_mask_overlay.png", alpha=0.5)

        geom_metrics = compute_metrics(geom_dir / "geom_mask.png", sample.gt_path)
        soft_metrics = None
        soft_best_temperature = None
        soft_heat_path = ""
        soft_colormap_path = ""
        soft_overlay_path = ""
        soft_variants: List[Dict[str, object]] = []
        soft_candidates = geom_outputs.get("soft_fusion_multi")
        if soft_candidates is None and geom_outputs.get("soft_fusion") is not None:
            soft_candidates = [geom_outputs["soft_fusion"]]

        soft_best_map: Optional[np.ndarray] = None
        soft_best_metrics: Optional[Dict[str, float]] = None
        if soft_candidates:
            for candidate in soft_candidates:
                temp_val = candidate.get("temperature")
                if temp_val is None and candidate.get("params"):
                    temp_val = candidate["params"].get("temperature")
                temp_val = float(temp_val if temp_val is not None else soft_temperature_values[0])
                map_arr = np.clip(np.asarray(candidate["map"], dtype=np.float32), 0.0, None)
                suffix = f"t{temp_val:.2f}".replace(".", "")
                soft_heat_file = geom_dir / f"soft_fusion_heat_{suffix}.png"
                soft_colormap_file = geom_dir / f"soft_fusion_colormap_{suffix}.png"
                soft_overlay_file = geom_dir / f"soft_fusion_overlay_{suffix}.png"
                soft_npy_file = geom_dir / f"soft_fusion_heat_{suffix}.npy"
                max_soft = float(map_arr.max())
                soft_norm = map_arr / max_soft if max_soft > 0 else map_arr
                Image.fromarray((np.clip(soft_norm, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(soft_heat_file)
                save_colormap(soft_norm, soft_colormap_file)
                save_overlay(sample.image_path, soft_norm, soft_overlay_file, alpha=0.5)
                np.save(soft_npy_file, map_arr.astype(np.float32))
                metrics_soft = compute_metrics(soft_heat_file, sample.gt_path)
                record = {
                    "temperature": temp_val,
                    "heat": str(soft_heat_file),
                    "colormap": str(soft_colormap_file),
                    "overlay": str(soft_overlay_file),
                    "npy": str(soft_npy_file),
                    "mKLD": metrics_soft["mKLD"],
                    "mSIM": metrics_soft["mSIM"],
                    "mNSS": metrics_soft["mNSS"],
                }
                soft_variants.append(record)
                if soft_best_metrics is None or metrics_soft["mKLD"] < soft_best_metrics["mKLD"]:
                    soft_best_metrics = metrics_soft
                    soft_best_map = map_arr.copy()
                    soft_best_temperature = temp_val
                    soft_heat_path = str(soft_heat_file)
                    soft_colormap_path = str(soft_colormap_file)
                    soft_overlay_path = str(soft_overlay_file)
            if soft_best_map is not None:
                legacy_heat = geom_dir / "soft_fusion_heat.png"
                legacy_colormap = geom_dir / "soft_fusion_colormap.png"
                legacy_overlay = geom_dir / "soft_fusion_overlay.png"
                max_soft = float(soft_best_map.max())
                soft_norm = soft_best_map / max_soft if max_soft > 0 else soft_best_map
                Image.fromarray((np.clip(soft_norm, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(legacy_heat)
                save_colormap(soft_norm, legacy_colormap)
                save_overlay(sample.image_path, soft_norm, legacy_overlay, alpha=0.5)
                np.save(geom_dir / "soft_fusion_heat.npy", soft_best_map.astype(np.float32))
                soft_heat_path = str(legacy_heat)
                soft_colormap_path = str(legacy_colormap)
                soft_overlay_path = str(legacy_overlay)
                soft_metrics = soft_best_metrics

        fusion_thresholds = geom_cfg.get("fusion_thresholds")
        if fusion_thresholds is None:
            single_thr = geom_cfg.get("fusion_threshold")
            if single_thr is None:
                fusion_thresholds = [0.3]
            elif isinstance(single_thr, (list, tuple)):
                fusion_thresholds = list(single_thr)
            else:
                fusion_thresholds = [single_thr]
        elif isinstance(fusion_thresholds, (int, float)):
            fusion_thresholds = [fusion_thresholds]

        fusion_thresholds = [float(t) for t in fusion_thresholds]
        keep_largest = bool(geom_cfg.get("keep_largest", False))

        base_heat = verb_heat
        if soft_best_map is not None:
            base_heat = np.clip(soft_best_map, 0.0, None).astype(np.float32)

        score_map = np.zeros_like(base_heat, dtype=np.float32)
        vmax = float(base_heat.max())
        if vmax > 0:
            score_map = base_heat / vmax
        np.save(final_dir / "score_map.npy", score_map.astype(np.float32))
        save_colormap(score_map, final_dir / "score_map.png")

        final_infos = []
        geom_mask_clipped = np.clip(geom_mask, 0.0, 1.0).astype(np.float32)
        for idx, thr in enumerate(fusion_thresholds):
            mask = (score_map >= thr).astype(np.float32)
            mask *= geom_mask_clipped
            if keep_largest:
                mask = largest_component(mask)
            mask = (mask > 0).astype(np.float32)

            if idx == 0:
                mask_name = "final_mask.png"
                overlay_name = "final_overlay.png"
            else:
                suffix = f"{thr:.2f}".replace(".", "")
                mask_name = f"final_mask_t{suffix}.png"
                overlay_name = f"final_overlay_t{suffix}.png"

            mask_path = final_dir / mask_name
            overlay_path = final_dir / overlay_name
            save_binary_mask(mask, mask_path)
            save_overlay(sample.image_path, mask, overlay_path, alpha=0.5)
            metrics_final = compute_metrics(mask_path, sample.gt_path)
            final_infos.append(
                {
                    "threshold": thr,
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                    **metrics_final,
                }
            )

        best_final = None
        if final_infos:
            best_final = min(final_infos, key=lambda x: x["mKLD"])

        best_similarity = None
        if geom_outputs.get("similarity_scores"):
            valid_scores = [s for s in geom_outputs["similarity_scores"] if s is not None]
            if valid_scores:
                best_similarity = max(valid_scores)

        return {
            "roi_info": roi_info,
            "geom_outputs": geom_outputs,
            "geom_metrics": geom_metrics,
            "soft_heat_path": str(soft_heat_path) if soft_heat_path else "",
            "soft_best_temperature": soft_best_temperature,
            "soft_metrics": soft_metrics,
            "soft_variants": soft_variants,
            "final_masks": final_infos,
            "final_best": best_final,
            "geom_energy_path": str(geom_dir / "geom_energy.png"),
            "geom_mask_path": str(geom_dir / "geom_mask.png"),
            "soft_colormap_path": str(soft_colormap_path) if soft_colormap_path else "",
            "soft_overlay_path": str(soft_overlay_path) if soft_overlay_path else "",
            "best_similarity": best_similarity,
        }
    except Exception as exc:
        logging.exception("Geometry pipeline failed for %s/%s/%s: %s", sample.affordance, sample.object_name, sample.image_path.name, exc)
        return None


def compute_metrics(heat_path: Path, gt_path: Path) -> Dict[str, float]:
    heat = load_gray(heat_path)
    gt = load_gray(gt_path)
    if heat.shape != gt.shape:
        heat = cv2.resize(heat, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # normalize heat to 0-255
    mn, mx = float(heat.min()), float(heat.max())
    if mx - mn < 1e-12:
        heat_norm = np.zeros_like(heat)
    else:
        heat_norm = (heat - mn) / (mx - mn) * 255.0

    gt_norm = gt.copy()
    if gt_norm.max() <= 1.0:
        gt_norm = gt_norm * 255.0
    elif gt_norm.max() > 255.0:
        gt_norm = np.clip(gt_norm, 0, 255)

    return {
        "mKLD": cal_kl(heat_norm, gt_norm),
        "mSIM": cal_sim(heat_norm, gt_norm),
        "mNSS": cal_nss(heat_norm, gt_norm),
    }


def process_sample(
    sample: SampleEntry,
    cfg: Dict,
    exp_output_root: Path,
    summary_headers: List[str],
    geom_cfg: Dict,
) -> Optional[Dict]:
    affordance_cfg = PROMPT_TEMPLATES.get(sample.affordance)
    if affordance_cfg is None:
        logging.warning("Affordance %s not in prompt templates. Skipping.", sample.affordance)
        return None

    prompt = affordance_cfg["prompt"].format(object=format_object_name(sample.object_name))
    tokens_candidates = affordance_cfg["tokens"]

    sample_dir = (
        exp_output_root
        / sample.affordance
        / sample.object_name
        / sample.image_path.stem
    )
    kontext_dir = sample_dir / "kontext"

    geom_enabled = bool(geom_cfg.get("enable"))

    height = cfg.get("height")
    width = cfg.get("width")
    if cfg.get("match_input_resolution") and (height is None or width is None):
        with Image.open(sample.image_path) as img:
            orig_w, orig_h = img.size
        matched_h, matched_w = pick_preferred_resolution((orig_h, orig_w))
        height = matched_h
        width = matched_w
        logging.info(
            "match_input_resolution=True -> %s/%s use preferred size %dx%d (original %dx%d)",
            sample.affordance,
            sample.object_name,
            matched_h,
            matched_w,
            orig_h,
            orig_w,
        )

    result = run_kontext_generation(
        script_path=VIS_SCRIPT,
        model_dir=Path(cfg["model_dir"]),
        image_path=sample.image_path,
        prompt=prompt,
        output_root=kontext_dir,
        num_steps=cfg["num_inference_steps"],
        guidance=cfg["guidance_scale"],
        seed=cfg["seed"],
        height=height,
        width=width,
        negative_prompt=None,
    )

    token_info = select_token(result["tokens"], tokens_candidates)
    if token_info is None:
        logging.warning("No suitable token found for %s/%s/%s", sample.affordance, sample.object_name, sample.image_path.name)
        return None

    heat_path = get_heatmap_path(result["per_token_dir"], token_info["index"], token_info["token"])

    mapped_dir = sample_dir / "mapped"
    warped_heat = warp_heatmap_cli(
        script_path=WARP_SCRIPT,
        original=sample.image_path,
        edited=result["generated_image"],
        heatmap=heat_path,
        out_dir=mapped_dir,
        alpha=0.5,
    )

    object_token_info = None
    object_heatmap_path = None
    warped_object_heat = None
    if geom_enabled:
        object_candidates = build_object_token_candidates(sample.object_name)
        object_token_info = select_token(result["tokens"], object_candidates)
        if object_token_info is None:
            logging.warning(
                "Geom pipeline: no object token found for %s/%s/%s",
                sample.affordance,
                sample.object_name,
                sample.image_path.name,
            )
            geom_enabled = False
        else:
            try:
                object_heatmap_path = get_heatmap_path(
                    result["per_token_dir"],
                    object_token_info["index"],
                    object_token_info["token"],
                )
                warped_object_heat = warp_heatmap_cli(
                    script_path=WARP_SCRIPT,
                    original=sample.image_path,
                    edited=result["generated_image"],
                    heatmap=object_heatmap_path,
                    out_dir=mapped_dir,
                    alpha=0.5,
                )
            except Exception as exc:
                logging.exception(
                    "Geom pipeline: failed to obtain object heatmap for %s/%s/%s: %s",
                    sample.affordance,
                    sample.object_name,
                    sample.image_path.name,
                    exc,
                )
                geom_enabled = False
                object_heatmap_path = None
                warped_object_heat = None

    metrics = compute_metrics(warped_heat, sample.gt_path)

    overlay_path = mapped_dir / (heat_path.stem + "_overlay_on_original.png")
    detail = {
        "affordance": sample.affordance,
        "object": sample.object_name,
        "image": sample.image_path.name,
        "prompt": prompt,
        "token_index": token_info["index"],
        "token_str": token_info["token"],
        "generated_image": str(result["generated_image"]),
        "heatmap": str(heat_path),
        "warped_heatmap": str(warped_heat),
        "overlay": str(overlay_path) if overlay_path.exists() else "",
        "gt": str(sample.gt_path),
        **metrics,
    }

    geom_result = None
    if geom_enabled and object_heatmap_path and warped_object_heat:
        geom_result = run_geom_pipeline(
            sample,
            sample_dir,
            geom_cfg,
            warped_heat,
            warped_object_heat,
        )
        if geom_result is None:
            geom_enabled = False

    # populate optional fields for summary consistency
    detail["object_token_index"] = object_token_info["index"] if object_token_info else ""
    detail["object_token_str"] = object_token_info["token"] if object_token_info else ""
    detail["object_heatmap"] = str(object_heatmap_path) if object_heatmap_path else ""
    detail["object_warped_heatmap"] = str(warped_object_heat) if warped_object_heat else ""

    for key in [
        "geom_mask",
        "geom_energy",
        "geom_selected_pc",
        "geom_similarity_best",
        "geom_similarity_scores",
        "geom_mKLD",
        "geom_mSIM",
        "geom_mNSS",
        "soft_heatmap",
        "soft_best_temperature",
        "soft_mKLD",
        "soft_mSIM",
        "soft_mNSS",
        "soft_variants_detail",
        "final_mask",
        "final_threshold",
        "final_mKLD",
        "final_mSIM",
        "final_mNSS",
    ]:
        detail.setdefault(key, "")

    if geom_enabled and geom_result is not None:
        geom_outputs = geom_result["geom_outputs"]
        detail["geom_mask"] = geom_result["geom_mask_path"]
        detail["geom_energy"] = geom_result["geom_energy_path"]
        detail["geom_selected_pc"] = geom_outputs.get("selected_pc", "")
        if geom_result.get("best_similarity") is not None:
            detail["geom_similarity_best"] = geom_result["best_similarity"]
        if geom_outputs.get("similarity_scores"):
            detail["geom_similarity_scores"] = json.dumps(geom_outputs["similarity_scores"])
        detail["geom_mKLD"] = geom_result["geom_metrics"]["mKLD"]
        detail["geom_mSIM"] = geom_result["geom_metrics"]["mSIM"]
        detail["geom_mNSS"] = geom_result["geom_metrics"]["mNSS"]
        if geom_result.get("soft_heat_path"):
            detail["soft_heatmap"] = geom_result["soft_heat_path"]
        if geom_result.get("soft_best_temperature") is not None:
            detail["soft_best_temperature"] = geom_result["soft_best_temperature"]
        if geom_result.get("soft_metrics"):
            detail["soft_mKLD"] = geom_result["soft_metrics"]["mKLD"]
            detail["soft_mSIM"] = geom_result["soft_metrics"]["mSIM"]
            detail["soft_mNSS"] = geom_result["soft_metrics"]["mNSS"]
        detail["soft_variants_detail"] = json.dumps(geom_result.get("soft_variants", []))

        final_masks = geom_result.get("final_masks", [])
        best_final = geom_result.get("final_best")
        if best_final is None and final_masks:
            best_final = final_masks[0]
        if best_final is not None:
            detail["final_mask"] = best_final["mask"]
            detail["final_threshold"] = best_final["threshold"]
            detail["final_mKLD"] = best_final["mKLD"]
            detail["final_mSIM"] = best_final["mSIM"]
            detail["final_mNSS"] = best_final["mNSS"]
        detail["final_masks_detail"] = json.dumps(final_masks)
    else:
        detail["final_masks_detail"] = json.dumps([])
        detail["soft_variants_detail"] = json.dumps([])

    save_json(detail, sample_dir / "metrics.json")
    append_csv(exp_output_root / "summary.csv", summary_headers, detail)
    return detail


def main():
    parser = argparse.ArgumentParser(description="Run Flux Kontext evaluation over AGD20K")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom_cfg = cfg.get("geom_pipeline", {}) or {}
    geom_enabled = bool(geom_cfg.get("enable", False))

    dataset_root = Path(cfg["dataset_root"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()

    output_root = Path(cfg["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    cfg["model_dir"] = str(Path(cfg["model_dir"]).expanduser().resolve())

    exp_output_root: Optional[Path] = None
    if cfg.get("resume"):
        resume_dir = cfg.get("resume_run_dir")
        if resume_dir:
            resume_path = Path(resume_dir).expanduser()
            if not resume_path.is_absolute():
                resume_path = (output_root / resume_path).resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume directory not found: {resume_path}")
            exp_output_root = resume_path
        else:
            existing_runs = sorted(d for d in output_root.iterdir() if d.is_dir())
            if existing_runs:
                exp_output_root = existing_runs[-1]
    if exp_output_root is None:
        exp_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_output_root = output_root / exp_id
    exp_output_root.mkdir(parents=True, exist_ok=True)

    setup_logging(exp_output_root / "run.log")
    logging.info("Starting Flux Kontext evaluation, output: %s", exp_output_root)

    summary_headers = [
        "affordance",
        "object",
        "image",
        "prompt",
        "token_index",
        "token_str",
        "warped_heatmap",
        "gt",
        "mKLD",
        "mSIM",
        "mNSS",
        "object_token_index",
        "object_token_str",
        "object_heatmap",
        "object_warped_heatmap",
        "geom_mask",
        "geom_selected_pc",
        "geom_similarity_best",
        "geom_mKLD",
        "geom_mSIM",
        "geom_mNSS",
        "soft_heatmap",
        "soft_best_temperature",
        "soft_mKLD",
        "soft_mSIM",
        "soft_mNSS",
        "soft_variants_detail",
        "final_mask",
        "final_threshold",
        "final_mKLD",
        "final_mSIM",
        "final_mNSS",
        "geom_similarity_scores",
        "final_masks_detail",
    ]

    base_metric_keys = ["mKLD", "mSIM", "mNSS"]
    extra_metric_keys: List[str] = []
    if geom_enabled:
        extra_metric_keys.extend(["geom_mKLD", "geom_mSIM", "geom_mNSS"])
        if geom_cfg.get("geom_soft_fusion", False):
            extra_metric_keys.extend(["soft_mKLD", "soft_mSIM", "soft_mNSS"])
        extra_metric_keys.extend(["final_mKLD", "final_mSIM", "final_mNSS"])
    metric_keys = base_metric_keys + extra_metric_keys

    def new_metric_accumulator() -> Dict[str, float]:
        data = {k: 0.0 for k in metric_keys}
        data["count"] = 0.0
        return data

    total = 0
    processed = 0
    skipped_samples: List[Dict[str, str]] = []
    failed_samples: List[Dict[str, str]] = []
    metric_sums = {k: 0.0 for k in metric_keys}
    per_affordance_sums = defaultdict(new_metric_accumulator)
    processed_keys = set()
    resume_reused = 0

    summary_path = exp_output_root / "summary.csv"
    if cfg.get("resume") and summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["affordance"], row["object"], row["image"])
                processed_keys.add(key)
                processed += 1
                aff_metrics = per_affordance_sums[row["affordance"]]
                aff_metrics["count"] += 1
                for key_metric in metric_keys:
                    val_str = row.get(key_metric, "")
                    if not val_str:
                        continue
                    try:
                        val = float(val_str)
                    except (TypeError, ValueError):
                        continue
                    metric_sums[key_metric] += val
                    aff_metrics[key_metric] += val
        if processed:
            logging.info("Resume enabled: preloaded %d entries from %s", processed, summary_path)
    for sample in iter_agd20k_samples(
        dataset_root,
        affordances=cfg.get("affordances"),
        max_per_object=cfg.get("max_images_per_object"),
    ):
        total += 1
        key = (sample.affordance, sample.object_name, sample.image_path.name)
        if cfg.get("resume") and key in processed_keys:
            resume_reused += 1
            continue
        try:
            detail = process_sample(sample, cfg, exp_output_root, summary_headers, geom_cfg)
            if detail:
                processed += 1
                processed_keys.add(key)
                aff_metrics = per_affordance_sums[detail["affordance"]]
                aff_metrics["count"] += 1
                for key_metric in metric_keys:
                    val_str = detail.get(key_metric, "")
                    if val_str in ("", None):
                        continue
                    try:
                        val = float(val_str)
                    except (TypeError, ValueError):
                        continue
                    metric_sums[key_metric] += val
                    aff_metrics[key_metric] += val
            else:
                skipped_samples.append(
                    {
                        "affordance": sample.affordance,
                        "object": sample.object_name,
                        "image": sample.image_path.name,
                        "reason": "detail_not_returned",
                    }
                )
        except Exception as exc:
            logging.exception(
                "Failed processing %s/%s/%s: %s",
                sample.affordance,
                sample.object_name,
                sample.image_path.name,
                exc,
            )
            failed_samples.append(
                {
                    "affordance": sample.affordance,
                    "object": sample.object_name,
                    "image": sample.image_path.name,
                    "error": str(exc),
                }
            )

    overall_metrics = {}
    if processed:
        for key_metric in metric_keys:
            overall_metrics[f"{key_metric}_mean"] = metric_sums[key_metric] / processed
        logging.info(
            "Global metrics (processed %d samples): KLD=%.3f, SIM=%.3f, NSS=%.3f",
            processed,
            overall_metrics.get("mKLD_mean", float("nan")),
            overall_metrics.get("mSIM_mean", float("nan")),
            overall_metrics.get("mNSS_mean", float("nan")),
        )
    else:
        logging.warning("No samples processed successfully; global metrics not computed.")

    per_affordance_metrics = {}
    for aff, values in per_affordance_sums.items():
        count = values["count"]
        if not count:
            continue
        metrics_entry = {"count": count}
        for key_metric in metric_keys:
            metrics_entry[f"{key_metric}_mean"] = values[key_metric] / count
        per_affordance_metrics[aff] = metrics_entry

    global_metrics = {
        "total_samples": total,
        "processed_samples": processed,
        "resume_reused": resume_reused,
        "skipped_samples": len(skipped_samples),
        "failed_samples": len(failed_samples),
        "overall_metrics": overall_metrics if processed else None,
        "per_affordance_metrics": per_affordance_metrics,
        "skipped_detail": skipped_samples,
        "failed_detail": failed_samples,
    }
    if geom_enabled:
        global_metrics["geom_pipeline_config"] = geom_cfg
    save_json(global_metrics, exp_output_root / "global_metrics.json")

    logging.info(
        "Completed. Processed %d/%d samples. Reused(resume): %d | Skipped: %d | Failed: %d",
        processed,
        total,
        resume_reused,
        len(skipped_samples),
        len(failed_samples),
    )


if __name__ == "__main__":
    main()
