#!/usr/bin/env python3
"""
End-to-end affordance pipeline for a single image.

Stages
------
1. 调用 Flux Kontext 捕获 cross-attention 热图并写出 per-token 可视化；
2. 将动词/名词热图 warp 回原图，依据 object 热图生成 bbox ROI；
3. 在 ROI token 上运行 DINOv3 PCA，输出主成分响应；
4. （可选）融合 pc1~3 得到几何掩码；
5. 将 verb attention 与几何/pc1 结合生成最终 affordance 掩码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

# 项目根路径
REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO_SHOT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ZERO_SHOT_ROOT / "cache"

FLUX_SCRIPT = REPO_ROOT / "FLUX" / "Flux_Kontext_Interaction" / "visualize_flux_kontext_cross_attention.py"
WARP_SCRIPT = REPO_ROOT / "FLUX" / "Flux_Kontext_Interaction" / "warp_heatmap_to_original.py"
MODEL_DIR_DEFAULT = Path("/home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev")

# 追加 Python Path，复用 dino/Section2 模块
sys.path.append(str(REPO_ROOT / "dino"))
sys.path.append(str(REPO_ROOT / "Section2_exp"))

from pipeline import flux_stage  # noqa: E402
from pipeline.geometry_stage import generate_geometry_mask, largest_component  # noqa: E402
from pipeline.pca_stage import DINOArtifacts, extract_dino_tokens, run_pca  # noqa: E402
from pipeline.roi_stage import build_roi_mask, compute_roi_tokens, restore_to_original  # noqa: E402
from pipeline.utils import ensure_dir, save_colormap, save_overlay, save_colormap_overlay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux + DINO zero-shot affordance pipeline")
    parser.add_argument("--image", type=Path, default=ZERO_SHOT_ROOT / "knife_01_00000001_rgb.jpg", help="输入 UMD RGB 图像路径")
    parser.add_argument("--prompt", type=str, default="Grasp knife", help="Kontext prompt")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional Kontext negative prompt")
    parser.add_argument("--flux-model", type=Path, default=MODEL_DIR_DEFAULT, help="Flux Kontext 模型目录")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-input-resolution",
        dest="match_input_resolution",
        action="store_true",
        help="自动匹配原图尺寸对应的 Kontext 推荐分辨率（默认开启）",
    )
    parser.add_argument(
        "--no-match-input-resolution",
        dest="match_input_resolution",
        action="store_false",
        help="关闭自动匹配，使用模型默认尺寸",
    )
    parser.set_defaults(match_input_resolution=True)
    parser.add_argument("--output-root", type=Path, default=ZERO_SHOT_ROOT / "outputs")
    parser.add_argument("--roi-percentile", type=float, default=80.0, help="hotmap fallback 分位阈值")
    parser.add_argument("--token-threshold", type=float, default=0.1, help="token 覆盖率阈值")
    parser.add_argument("--fusion-threshold", type=float, default=0.1, help="最终掩码默认阈值")
    parser.add_argument(
        "--fusion-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="选填：同时输出多个阈值的最终掩码，例：--fusion-thresholds 0.35 0.55",
    )
    parser.add_argument(
        "--verb-tokens",
        type=str,
        default="brush",
        help="逗号分隔的动词词形（不带 T5 ▁ 前缀），自动扩展常见变体",
    )
    parser.add_argument(
        "--object-tokens",
        type=str,
        default="toothbrush",
        help="逗号分隔的名词词形（不带 ▁），自动扩展候选",
    )
    parser.add_argument("--geom-mask", type=bool, default=True, help="启用或禁用 PCA 通道几何掩码")
    parser.add_argument("--geom-sigma", type=float, default=1.2, help="几何通道平滑 σ")
    parser.add_argument("--geom-threshold", type=float, default=0.70, help="几何能量二值化阈值 (0~1)")
    parser.add_argument(
        "--geom-soft-fusion",
        type=bool, default=True, 
        help="启用 verb 热图与选中 PCA 通道的 soft 融合热图",
    )
    parser.add_argument("--geom-soft-lambda", type=float, default=0.65, help="soft 融合中 verb 权重 λ (0~1)")
    parser.add_argument("--geom-soft-gamma", type=float, default=0.7, help="几何能量幅度压缩系数 γ (>0)")
    parser.add_argument("--geom-soft-temperature", type=float, default=0.5, help="softmax 温度 τ (>0，越大越平滑)")
    parser.add_argument("--geom-soft-base", type=float, default=0, help="Dirichlet 薄底座占比 η (0~1)")
    parser.add_argument(
        "--geom-soft-log1p",
        action="store_true",
        help="softmax 前对热图做 log1p，压缩峰值动态范围",
    )
    parser.add_argument("--pca-components", type=int, default=10, help="PCA 主成分数量 (>=1)")
    parser.add_argument(
        "--geom-pc-index",
        type=int,
        default=None,
        help="指定用于几何阶段的 PCA 通道 (1-based, ≤ pca-components)。若不设置则自动按相似度挑选",
    )
    parser.add_argument(
        "--disable-object-roi",
        action="store_true",
        help="禁用基于 object heat 的 ROI，使用全图 token 执行 PCA",
    )
    parser.add_argument("--keep-largest",  type=bool, default=True, help="最终掩码仅保留最大连通区域")
    parser.add_argument(
        "--direct-mapping",
        action="store_true",
        help="启用基于 img_ids 的直接热图映射（默认关闭，使用仿射 warp）",
    )
    parser.add_argument(
        "--geom-sim-use-nss",
        action="store_true",
        help="启用 NSS+top-k 能量差选择 PCA 通道（失败时回退到余弦相似度）",
    )
    parser.add_argument(
        "--geom-sim-topk-percent",
        type=float,
        default=5.0,
        help="使用 attention fallback 时，verb heat 前多少百分比像素作为 top-k 掩码",
    )
    parser.add_argument(
        "--geom-sim-nss-weight",
        type=float,
        default=1.0,
        help="attention fallback 中 NSS 得分的权重",
    )
    parser.add_argument(
        "--geom-sim-topk-weight",
        type=float,
        default=0.0,
        help="attention fallback 中 top-k 能量差的权重",
    )
    return parser.parse_args()


DEFAULT_VERB_BASE = ["hold", "cut"]
DEFAULT_OBJECT_BASE = ["knife", "axe"]


def parse_token_bases(raw: str | None, default: List[str]) -> List[str]:
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or list(default)


def expand_token_candidates(base_tokens: List[str], *, include_inflections: bool) -> List[str]:
    seen: List[str] = []
    added = set()
    for base in base_tokens:
        if not base:
            continue
        base = base.strip()
        if not base:
            continue
        clean = base.lstrip("▁").strip()
        if not clean:
            continue
        variants = set()
        for stem in {base, clean, clean.lower(), clean.capitalize(), clean.upper()}:
            if not stem:
                continue
            variants.add(stem)
        for stem in list(variants):
            if not stem.startswith("▁"):
                variants.add("▁" + stem)
            variants.add(stem.lower())
            variants.add("_" + stem.lower())
            variants.add("_" + stem)
        if include_inflections:
            for stem in {clean, clean.lower()}:
                if not stem:
                    continue
                variants.update(
                    {
                        stem + "ing",
                        stem + "ed",
                        stem + "s",
                        "▁" + stem + "ing",
                        "▁" + stem + "ed",
                        "▁" + stem + "s",
                        "_" + stem + "ing",
                        "_" + stem + "ed",
                        "_" + stem + "s",
                    }
                )
        for cand in variants:
            cand = cand.strip()
            if not cand:
                continue
            if cand not in added:
                seen.append(cand)
                added.add(cand)
    return seen


def stage_flux(
    args: argparse.Namespace,
    flux_root: Path,
    use_direct: bool,
) -> dict:
    flux_run_dir = flux_stage.run_flux_stage(
        sys.executable,
        FLUX_SCRIPT,
        args.image,
        args.prompt,
        args.negative_prompt,
        flux_root,
        args.flux_model,
        args.num_steps,
        args.guidance,
        args.seed,
        match_input_resolution=args.match_input_resolution,
    )

    tokens, token_map = flux_stage.load_tokens(flux_run_dir / "tokens_t5.json")

    verb_base = parse_token_bases(args.verb_tokens, DEFAULT_VERB_BASE)
    object_base = parse_token_bases(args.object_tokens, DEFAULT_OBJECT_BASE)
    verb_candidates = expand_token_candidates(verb_base, include_inflections=True)
    object_candidates = expand_token_candidates(object_base, include_inflections=False)

    verb_idx, verb_token = flux_stage.pick_token(tokens, verb_candidates)
    obj_idx, obj_token = flux_stage.pick_token(tokens, object_candidates)
    print(f"[token] verb={verb_token} idx={verb_idx}; object={obj_token} idx={obj_idx}")

    if use_direct:
        direct_maps = flux_stage.compute_direct_heatmaps(flux_run_dir, args.image, [verb_token, obj_token])
    else:
        direct_maps = {}

    verb_array = direct_maps.get(verb_token)
    obj_array = direct_maps.get(obj_token)

    fallback_info = {"verb": None, "object": None}
    if verb_array is None or obj_array is None:
        per_token_dir = flux_run_dir / "per_token"
        verb_heat = flux_stage.locate_heatmap(per_token_dir, verb_idx, verb_token)
        obj_heat = flux_stage.locate_heatmap(per_token_dir, obj_idx, obj_token)
        mapped_dir = ensure_dir(flux_run_dir / "mapped")
        verb_warp = flux_stage.warp_heatmap(sys.executable, WARP_SCRIPT, args.image, flux_run_dir / "gen.png", verb_heat, mapped_dir)
        obj_warp = flux_stage.warp_heatmap(sys.executable, WARP_SCRIPT, args.image, flux_run_dir / "gen.png", obj_heat, mapped_dir)
        fallback_info["verb"] = verb_warp
        fallback_info["object"] = obj_warp
        if verb_array is None:
            verb_img = cv2.imread(str(verb_warp["heat"]), cv2.IMREAD_GRAYSCALE)
            verb_array = verb_img.astype(np.float32) / 255.0 if verb_img is not None else None
        if obj_array is None:
            obj_img = cv2.imread(str(obj_warp["heat"]), cv2.IMREAD_GRAYSCALE)
            obj_array = obj_img.astype(np.float32) / 255.0 if obj_img is not None else None

    if verb_array is None or obj_array is None:
        raise RuntimeError("Failed to obtain valid attention heatmaps for verb/object tokens")

    return {
        "run_dir": flux_run_dir,
        "token_map": token_map,
        "direct_mapping": use_direct,
        "verb": {
            "index": verb_idx,
            "token": verb_token,
            "heat_array": verb_array,
            "fallback": fallback_info["verb"],
            "base": verb_base,
            "candidates": verb_candidates,
            "direct": bool(use_direct and fallback_info["verb"] is None),
        },
        "object": {
            "index": obj_idx,
            "token": obj_token,
            "heat_array": obj_array,
            "fallback": fallback_info["object"],
            "base": object_base,
            "candidates": object_candidates,
            "direct": bool(use_direct and fallback_info["object"] is None),
        },
    }


def stage_roi(
    args: argparse.Namespace,
    run_root: Path,
    artifacts: DINOArtifacts,
    object_heat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    stage_dir = run_root / "stage_roi"
    if args.disable_object_roi or object_heat is None:
        meta = artifacts.meta
        roi_mask_orig = np.ones((meta.orig_h, meta.orig_w), dtype=np.float32)
        roi_mask_letterbox = np.ones((artifacts.Hp * artifacts.patch, artifacts.Wp * artifacts.patch), dtype=np.float32)
        token_mask = np.ones((artifacts.Hp, artifacts.Wp), dtype=np.uint8)
        roi_indices = np.arange(artifacts.tokens.shape[0])
        roi_info = {
            "mode": "full",
            "threshold": None,
            "percentile": None,
            "token_fraction": 1.0,
            "token_count": int(roi_indices.size),
        }
    else:
        roi_mask_orig, roi_mask_letterbox, roi_info = build_roi_mask(
            object_heat,
            artifacts.meta,
            percentile=args.roi_percentile,
        )
        roi_indices, token_mask = compute_roi_tokens(
            roi_mask_letterbox, artifacts.Hp, artifacts.Wp, artifacts.patch, args.token_threshold
        )
        roi_info["token_fraction"] = float(roi_indices.size) / float(artifacts.tokens.shape[0])
        roi_info["token_count"] = int(roi_indices.size)

    Image.fromarray((roi_mask_orig * 255).astype(np.uint8), mode="L").save(stage_dir / "object_roi_mask.png")
    Image.fromarray((roi_mask_letterbox * 255).astype(np.uint8), mode="L").save(stage_dir / "object_roi_letterbox.png")
    save_overlay(args.image, roi_mask_orig, stage_dir / "object_roi_overlay.png", alpha=0.35 if args.disable_object_roi else 0.45)
    np.save(stage_dir / "object_roi_letterbox.npy", roi_mask_letterbox)

    token_mask_vis = cv2.resize(token_mask.astype(np.float32), (artifacts.Wp * artifacts.patch, artifacts.Hp * artifacts.patch), interpolation=cv2.INTER_NEAREST)
    token_mask_orig = restore_to_original(token_mask_vis, artifacts.meta)
    save_overlay(args.image, token_mask_orig, stage_dir / "token_roi_overlay.png", alpha=0.25 if args.disable_object_roi else 0.35)
    np.save(stage_dir / "token_mask.npy", token_mask)
    Image.fromarray((np.clip(token_mask_orig, 0, 1) * 255).astype(np.uint8), mode="L").save(stage_dir / "token_roi_mask.png")
    with (stage_dir / "roi_info.json").open("w", encoding="utf-8") as fh:
        json.dump(roi_info, fh, indent=2)

    return roi_mask_orig, token_mask, roi_indices, roi_info


def stage_dino(
    run_root: Path,
    args: argparse.Namespace,
    artifacts: DINOArtifacts,
    roi_indices: np.ndarray,
) -> dict:
    pca_outputs = run_pca(artifacts, roi_indices, num_components=args.pca_components)
    letterbox_rgb = np.clip(pca_outputs["letterbox_rgb"], 0.0, 1.0)
    orig_rgb = np.clip(pca_outputs["orig_rgb"], 0.0, 1.0)
    orig_full = np.clip(pca_outputs["orig_full"], 0.0, 1.0)

    def to_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            return np.repeat(arr[..., None], 3, axis=2)
        channels = arr.shape[2]
        if channels == 3:
            return arr
        if channels == 1:
            return np.repeat(arr, 3, axis=2)
        if channels == 2:
            third = arr[..., :1]
            return np.concatenate([arr, third], axis=2)
        return arr[..., :3]

    Image.fromarray((to_rgb(letterbox_rgb) * 255).astype(np.uint8)).save(run_root / "stage_dino" / "pca_rgb_letterbox.png")
    Image.fromarray((to_rgb(orig_rgb) * 255).astype(np.uint8)).save(run_root / "stage_dino" / "pca_rgb_original.png")

    num_components = pca_outputs["num_components"]
    for ch in range(num_components):
        plane_orig = orig_full[..., ch]
        save_colormap(plane_orig, run_root / "stage_dino" / f"pc{ch + 1}_colormap.png")

    np.save(run_root / "stage_dino" / "pca_components.npy", pca_outputs["pca_components"])
    np.save(run_root / "stage_dino" / "explained_variance.npy", pca_outputs["explained_variance"])
    np.save(run_root / "stage_dino" / "explained_variance_ratio.npy", pca_outputs["explained_variance_ratio"])
    with (run_root / "stage_dino" / "bounds.json").open("w", encoding="utf-8") as fh:
        json.dump({"bounds": pca_outputs["bounds"]}, fh, indent=2)

    return {"orig_rgb": orig_rgb, "orig_full": orig_full, "pca": pca_outputs}


def stage_geometry(
    args: argparse.Namespace,
    run_root: Path,
    pcs_full: np.ndarray,
    verb_heat: np.ndarray,
) -> dict | None:
    if not args.geom_mask and not args.geom_soft_fusion:
        return None

    geom_dir = ensure_dir(run_root / "stage_geom")
    geom_outputs = generate_geometry_mask(
        pcs_full,
        smooth_sigma=args.geom_sigma,
        binary_threshold=args.geom_threshold,
        verb_map=verb_heat,
        enable_soft_fusion=args.geom_soft_fusion,
        soft_lambda=args.geom_soft_lambda,
        soft_gamma=args.geom_soft_gamma,
        soft_temperature=args.geom_soft_temperature,
        soft_dirichlet=args.geom_soft_base,
        soft_use_log1p=args.geom_soft_log1p,
        max_channels=args.pca_components,
        forced_pc_index=args.geom_pc_index,
        use_attention_fallback=args.geom_sim_use_nss,
        attention_topk_percent=args.geom_sim_topk_percent,
        attention_nss_weight=args.geom_sim_nss_weight,
        attention_topk_weight=args.geom_sim_topk_weight,
    )
    geom_energy = geom_outputs["energy"]
    geom_mask = geom_outputs["mask"]

    Image.fromarray((np.clip(geom_energy, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(geom_dir / "geom_energy.png")
    save_colormap(geom_energy, geom_dir / "geom_energy_colormap.png")
    np.save(geom_dir / "geom_energy.npy", geom_energy.astype(np.float32))

    if args.geom_mask:
        Image.fromarray((geom_mask * 255).astype(np.uint8), mode="L").save(geom_dir / "geom_mask.png")
        np.save(geom_dir / "geom_mask.npy", geom_mask.astype(np.float32))
        save_overlay(args.image, geom_mask, geom_dir / "geom_mask_overlay.png", alpha=0.5)

    meta_geom = {
        "weights": geom_outputs["weights"].tolist(),
        "thresholds": geom_outputs["thresholds"],
        "selected_pc": geom_outputs.get("selected_pc"),
        "pc_index": geom_outputs.get("pc_index"),
        "similarity_scores": geom_outputs.get("similarity_scores"),
        "similarity_method": geom_outputs.get("similarity_method"),
        "mask_enabled": bool(args.geom_mask),
        "soft_enabled": bool(args.geom_soft_fusion),
        "channels_used": int(geom_outputs.get("channels_used", pcs_full.shape[2] if pcs_full.ndim == 3 else 1)),
        "pc_labels": geom_outputs.get("pc_labels"),
    }
    if geom_outputs.get("attention_meta") is not None:
        meta_geom["attention_meta"] = geom_outputs["attention_meta"]
    if args.geom_soft_fusion:
        meta_geom["soft_params"] = {
            "lambda": args.geom_soft_lambda,
            "gamma": args.geom_soft_gamma,
            "temperature": args.geom_soft_temperature,
            "dirichlet": args.geom_soft_base,
            "log1p": bool(args.geom_soft_log1p),
        }
    soft_fusion = geom_outputs.get("soft_fusion")
    if soft_fusion is not None:
        soft_map = soft_fusion["map"]
        max_soft = float(np.max(soft_map))
        soft_norm = soft_map
        if max_soft > 0.0:
            soft_norm = soft_map / max_soft
        Image.fromarray((np.clip(soft_norm, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(geom_dir / "soft_fusion_heat.png")
        save_colormap(soft_norm, geom_dir / "soft_fusion_colormap.png")
        save_overlay(args.image, soft_norm, geom_dir / "soft_fusion_overlay.png", alpha=0.5)
        np.save(geom_dir / "soft_fusion_heat.npy", soft_map.astype(np.float32))
        meta_geom["soft_fusion"] = {
            "params": soft_fusion.get("params", {}),
            "heat": str(geom_dir / "soft_fusion_heat.png"),
            "overlay": str(geom_dir / "soft_fusion_overlay.png"),
        }
    with (geom_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_geom, fh, indent=2)

    return geom_outputs


def stage_final(
    args: argparse.Namespace,
    run_root: Path,
    base_heat: np.ndarray,
    geom_outputs: dict | None,
    fusion_thresholds: List[float],
) -> List[dict]:
    heat_arr = np.clip(base_heat, 0.0, None).astype(np.float32)
    geom_mask = None
    if geom_outputs is not None and args.geom_mask:
        mask_candidate = geom_outputs.get("mask")
        if mask_candidate is not None:
            geom_mask = np.clip(mask_candidate, 0.0, 1.0)

    min_val = float(np.min(heat_arr))
    max_val = float(np.max(heat_arr))
    denom = max_val - min_val
    if denom > 1e-9:
        score_norm = (heat_arr - min_val) / denom
    elif max_val > 0:
        score_norm = heat_arr / max_val
    else:
        score_norm = np.zeros_like(heat_arr, dtype=np.float32)

    save_colormap(score_norm, run_root / "stage_final" / "score_map.png")
    np.save(run_root / "stage_final" / "score_map.npy", score_norm)

    final_masks_info = []
    if args.keep_largest:
        print("[info] keep-largest enabled: final masks constrained to biggest connected component.")

    for idx_thr, thr in enumerate(fusion_thresholds):
        mask = (score_norm >= thr).astype(np.float32)
        if geom_mask is not None:
            mask *= geom_mask
        if args.keep_largest:
            mask = largest_component(mask)

        if idx_thr == 0:
            mask_name = "final_mask.png"
            overlay_name = "final_overlay.png"
        else:
            suff = f"{thr:.2f}".replace(".", "")
            mask_name = f"final_mask_t{suff}.png"
            overlay_name = f"final_overlay_t{suff}.png"

        mask_path = run_root / "stage_final" / mask_name
        overlay_path = run_root / "stage_final" / overlay_name

        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
        save_overlay(args.image, mask, overlay_path, alpha=0.5)

        final_masks_info.append(
            {
                "threshold": thr,
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "area": float(mask.sum()),
            }
        )

    return final_masks_info


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    fusion_thresholds = list(args.fusion_thresholds) if args.fusion_thresholds else [args.fusion_threshold]

    # Stage directory scaffold
    flux_root = ensure_dir(args.output_root / "flux_runs")
    flux_info = stage_flux(args, flux_root, args.direct_mapping)

    run_id = flux_info["run_dir"].name
    run_root = ensure_dir(args.output_root / run_id)
    for sub in ("stage_flux", "stage_roi", "stage_dino", "stage_final"):
        ensure_dir(run_root / sub)

    # 保存 stage_flux 信息
    stage_flux_info = {
        "flux_run_dir": str(flux_info["run_dir"]),
        "verb_token": flux_info["verb"]["token"],
        "verb_index": flux_info["verb"]["index"],
        "object_token": flux_info["object"]["token"],
        "object_index": flux_info["object"]["index"],
        "verb_base": flux_info["verb"].get("base", []),
        "object_base": flux_info["object"].get("base", []),
        "verb_candidates": flux_info["verb"].get("candidates", []),
        "object_candidates": flux_info["object"].get("candidates", []),
        "direct_mapping": bool(flux_info.get("direct_mapping")),
        "verb_direct": flux_info["verb"].get("direct", False),
        "object_direct": flux_info["object"].get("direct", False),
    }
    if flux_info["verb"].get("fallback") is not None:
        stage_flux_info["verb_fallback"] = {k: str(v) for k, v in flux_info["verb"]["fallback"].items()}
    if flux_info["object"].get("fallback") is not None:
        stage_flux_info["object_fallback"] = {k: str(v) for k, v in flux_info["object"]["fallback"].items()}

    heat_paths: Dict[str, str] = {}
    stage_flux_dir = run_root / "stage_flux"
    for label, info in [
        ("verb", flux_info["verb"]),
        ("object", flux_info["object"]),
    ]:
        heat_arr = info.get("heat_array")
        if heat_arr is not None:
            heat_img_path = stage_flux_dir / f"{label}_heat.png"
            Image.fromarray((np.clip(heat_arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(heat_img_path)
            save_colormap(heat_arr, stage_flux_dir / f"{label}_heat_colormap.png")
            save_overlay(args.image, heat_arr, stage_flux_dir / f"{label}_overlay.png", alpha=0.5)
            save_colormap_overlay(args.image, heat_arr, stage_flux_dir / f"{label}_overlay_colormap.png", alpha=0.6)
            heat_paths[label] = str(heat_img_path)
        elif info.get("fallback") is not None:
            warp = info["fallback"]
            overlay_path = Path(warp["overlay"])
            if overlay_path.exists():
                (stage_flux_dir / f"{label}_overlay.png").write_bytes(overlay_path.read_bytes())
                heat_paths[label] = str(overlay_path)
            heat_path = Path(warp["heat"])
            if heat_path.exists():
                (stage_flux_dir / f"{label}_heat_on_original.png").write_bytes(heat_path.read_bytes())
    stage_flux_info["saved_heats"] = heat_paths

    with (stage_flux_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(stage_flux_info, fh, indent=2)

    # DINO tokens
    artifacts = extract_dino_tokens(args.image, cache_root=CACHE_ROOT)
    # ROI 阶段
    roi_mask_orig, token_mask, roi_indices, roi_info = stage_roi(
        args,
        run_root,
        artifacts,
        flux_info["object"]["heat_array"],
    )

    # PCA / 主成分响应
    pca_stage_info = stage_dino(run_root, args, artifacts, roi_indices)

    # 几何掩码（可选）
    geom_outputs = stage_geometry(
        args,
        run_root,
        pca_stage_info["orig_full"],
        flux_info["verb"]["heat_array"],
    )

    base_heat = flux_info["verb"]["heat_array"]
    if geom_outputs is not None and args.geom_soft_fusion:
        soft_fusion = geom_outputs.get("soft_fusion")
        if soft_fusion is not None:
            base_heat = soft_fusion["map"]

    # 最终融合
    final_masks_info = stage_final(
        args,
        run_root,
        base_heat,
        geom_outputs,
        fusion_thresholds,
    )

    outputs_dict = {
        "pca_rgb": str(run_root / "stage_dino" / "pca_rgb_original.png"),
        "final_overlay": final_masks_info[0]["overlay"],
    }
    if "verb" in heat_paths:
        outputs_dict["verb_heat"] = heat_paths["verb"]
    if "object" in heat_paths:
        outputs_dict["object_heat"] = heat_paths["object"]

    summary = {
        "run_id": run_id,
        "flux_prompt": args.prompt,
        "flux_negative_prompt": args.negative_prompt,
        "roi_percentile": args.roi_percentile,
        "token_threshold": args.token_threshold,
        "fusion_threshold": args.fusion_threshold,
        "fusion_thresholds": fusion_thresholds,
        "num_roi_tokens": int(roi_indices.size),
        "total_tokens": int(artifacts.tokens.shape[0]),
        "flux_run_dir": str(flux_info["run_dir"]),
        "direct_mapping": bool(flux_info.get("direct_mapping")),
        "verb_direct": flux_info["verb"].get("direct", False),
        "object_direct": flux_info["object"].get("direct", False),
        "geom_threshold": float(args.geom_threshold),
        "outputs": outputs_dict,
        "final_masks": final_masks_info,
        "roi_info": roi_info,
        "dino_cache": {
            "root": str(CACHE_ROOT),
            "cache_hit": bool(artifacts.cache_hit),
        },
        "object_roi_enabled": not args.disable_object_roi,
        "pca_components_requested": int(args.pca_components),
    }
    if geom_outputs is not None:
        summary["geom_selected_pc"] = geom_outputs.get("selected_pc")
        summary["geom_pc_index"] = geom_outputs.get("pc_index")
        summary["geom_similarity_scores"] = geom_outputs.get("similarity_scores")
        summary["geom_similarity_method"] = geom_outputs.get("similarity_method")
        summary["geom_attention_meta"] = geom_outputs.get("attention_meta")
        geom_dir = run_root / "stage_geom"
        geometry_block: Dict[str, object] = {
            "energy_map": str(geom_dir / "geom_energy.png"),
            "weights": geom_outputs["weights"].tolist(),
            "thresholds": geom_outputs["thresholds"],
            "params": {
                "sigma": args.geom_sigma,
                "binary": args.geom_threshold,
            },
            "selected_pc": geom_outputs.get("selected_pc"),
            "pc_index": geom_outputs.get("pc_index"),
            "similarity_scores": geom_outputs.get("similarity_scores"),
            "similarity_method": geom_outputs.get("similarity_method"),
            "mask_enabled": bool(args.geom_mask),
            "soft_enabled": bool(args.geom_soft_fusion),
            "channels_used": int(geom_outputs.get("channels_used", pca_stage_info["pca"]["num_components"])),
            "pc_labels": geom_outputs.get("pc_labels"),
        }
        if geom_outputs.get("attention_meta") is not None:
            geometry_block["attention_meta"] = geom_outputs["attention_meta"]
        if args.geom_mask:
            geometry_block["mask"] = str(geom_dir / "geom_mask.png")
        if args.geom_soft_fusion:
            geometry_block["soft_params"] = {
                "lambda": args.geom_soft_lambda,
                "gamma": args.geom_soft_gamma,
                "temperature": args.geom_soft_temperature,
                "dirichlet": args.geom_soft_base,
                "log1p": bool(args.geom_soft_log1p),
            }
            soft_fusion = geom_outputs.get("soft_fusion")
            if soft_fusion is not None:
                geometry_block["soft_fusion"] = {
                    "heat": str(geom_dir / "soft_fusion_heat.png"),
                    "overlay": str(geom_dir / "soft_fusion_overlay.png"),
                    "params": soft_fusion.get("params", {}),
                }
        summary["geometry"] = geometry_block
    if pca_stage_info is not None:
        bounds_serializable = [[float(b[0]), float(b[1])] for b in pca_stage_info["pca"]["bounds"]]
        pca_meta = {
            "num_components": pca_stage_info["pca"]["num_components"],
            "explained_variance": pca_stage_info["pca"]["explained_variance"].tolist(),
            "explained_variance_ratio": pca_stage_info["pca"]["explained_variance_ratio"].tolist(),
            "bounds": bounds_serializable,
        }
        summary["pca_stats"] = pca_meta
    with (run_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[done] Pipeline finished.")
    print(f"  Flux run dir : {flux_info['run_dir']}")
    print(f"  Result dir   : {run_root}")
    print(f"  Final overlay: {final_masks_info[0]['overlay']}")


if __name__ == "__main__":
    main()
