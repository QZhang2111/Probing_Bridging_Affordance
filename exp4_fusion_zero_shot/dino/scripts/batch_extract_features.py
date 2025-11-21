#!/usr/bin/env python3
"""Batch extract DINOv3 dense tokens for the affordance datasets."""

from __future__ import annotations

import csv
import gc
from pathlib import Path
from typing import Dict

from PIL import Image
from tqdm import tqdm

from dino.src.feature_extraction import (
    append_manifest,
    ensure_dir,
    extract_last_tokens,
    load_dinov3,
    pick_target_by_orientation,
    resize_letterbox_to,
    sanitize,
    save_tokens_npz,
    sweep_cuda,
    to_tensor_norm,
)
from dino.src.settings import get_settings

PATCH_SIZE = 16
TARGET_UMD = (1280, 960)
TARGET_LAND_1920 = (1920, 1440)
TARGET_PORT_1920 = (1440, 1920)
MODEL_NAME = "dinov3_vit7b16"
SKIP_IF_EXISTS = True


def _prepare_paths() -> Dict[str, Path]:
    settings = get_settings()
    lists_dir = settings.paths.get("metadata_lists")
    if lists_dir is None:
        raise RuntimeError("metadata_lists path missing from configuration")

    cache_root = settings.paths.get("cache_root")
    if cache_root is None:
        raise RuntimeError("cache_root path missing from configuration")

    out_root = cache_root / "tokens"
    ensure_dir(out_root)

    csv_paths = {
        "umd": lists_dir / "umd_index.csv",
        "fun3du": lists_dir / "fun3du_index.csv",
        "kitchen": lists_dir / "kitchen_views.csv",
    }

    outputs = {ds: out_root / ds for ds in csv_paths}
    for path in outputs.values():
        ensure_dir(path)

    manifest = out_root / "manifest.jsonl"
    return {
        "manifest": manifest,
        "out_root": out_root,
        **{f"csv_{ds}": path for ds, path in csv_paths.items()},
        **{f"out_{ds}": path for ds, path in outputs.items()},
    }


def _extract_for_row(img_path: Path, target_size, model, meta_extra: Dict) -> Dict:
    if not img_path.exists():
        print(f"[WARN] missing file, skip: {img_path}")
        return {}

    img = Image.open(img_path).convert("RGB")
    img_pad, pad_meta = resize_letterbox_to(img, target_size, PATCH_SIZE)
    tokens, Hp, Wp = extract_last_tokens(model, to_tensor_norm(img_pad))
    meta = dict(
        H_patches=Hp,
        W_patches=Wp,
        patch_size=PATCH_SIZE,
        resized_h=pad_meta.final_h,
        resized_w=pad_meta.final_w,
        model=MODEL_NAME,
        preprocess="imagenet_meanstd",
        source_path=str(img_path),
        **pad_meta.as_dict(),
        **meta_extra,
    )
    meta["tokens_shape"] = tuple(tokens.shape)
    return {"tokens": tokens, "meta": meta}


def _handle_umd(row: Dict, model, outputs: Dict[str, Path], manifest: Path):
    img_path = Path(row["image_path"])
    cls_ = sanitize(row["class"])
    iid = sanitize(row["instance_id"])
    out_name = f"{cls_}_{iid}.vit7b16.{TARGET_UMD[0]}x{TARGET_UMD[1]}.last.npz"
    out_path = outputs["umd"] / out_name
    if SKIP_IF_EXISTS and out_path.exists():
        print(f"[skip] {out_path}")
        return

    result = _extract_for_row(img_path, TARGET_UMD, model, {"dataset": "UMD"})
    if not result:
        return
    save_tokens_npz(result["tokens"], result["meta"], out_path)
    append_manifest({"dataset": "UMD", "save_path": str(out_path), **result["meta"]}, manifest)
    print(f"[ok] UMD -> {out_path}  shape={result['tokens'].shape}")


def _handle_fun3du(row: Dict, model, outputs: Dict[str, Path], manifest: Path):
    img_path = Path(row["image_path"])
    scene = sanitize(row["scene"])
    sub = sanitize(row["subscene"])
    view = sanitize(row["view_id"])

    with Image.open(img_path) as img:
        target = pick_target_by_orientation(img, TARGET_LAND_1920, TARGET_PORT_1920)
    out_name = f"{scene}_{sub}_{view}.vit7b16.{target[0]}x{target[1]}.last.npz"
    out_path = outputs["fun3du"] / out_name
    if SKIP_IF_EXISTS and out_path.exists():
        print(f"[skip] {out_path}")
        return

    result = _extract_for_row(img_path, target, model, {"dataset": "Fun3DU"})
    if not result:
        return
    save_tokens_npz(result["tokens"], result["meta"], out_path)
    append_manifest({"dataset": "Fun3DU", "save_path": str(out_path), **result["meta"]}, manifest)
    print(f"[ok] Fun3DU -> {out_path}  shape={result['tokens'].shape}")


def _handle_kitchen(row: Dict, model, outputs: Dict[str, Path], manifest: Path):
    img_path = Path(row["image_path"])
    scene = sanitize(row["scene_id"])
    view = sanitize(row["view_id"])

    with Image.open(img_path) as img:
        target = pick_target_by_orientation(img, TARGET_LAND_1920, TARGET_PORT_1920)
    out_name = f"{scene}_{view}.vit7b16.{target[0]}x{target[1]}.last.npz"
    out_path = outputs["kitchen"] / out_name
    if SKIP_IF_EXISTS and out_path.exists():
        print(f"[skip] {out_path}")
        return

    result = _extract_for_row(img_path, target, model, {"dataset": "Kitchen"})
    if not result:
        return
    save_tokens_npz(result["tokens"], result["meta"], out_path)
    append_manifest({"dataset": "Kitchen", "save_path": str(out_path), **result["meta"]}, manifest)
    print(f"[ok] Kitchen -> {out_path}  shape={result['tokens'].shape}")


def run() -> None:
    paths = _prepare_paths()

    csv_map = {
        "umd": paths["csv_umd"],
        "fun3du": paths["csv_fun3du"],
        "kitchen": paths["csv_kitchen"],
    }
    outputs = {
        "umd": paths["out_umd"],
        "fun3du": paths["out_fun3du"],
        "kitchen": paths["out_kitchen"],
    }
    manifest = paths["manifest"]

    model = load_dinov3(MODEL_NAME)

    handlers = {
        "umd": _handle_umd,
        "fun3du": _handle_fun3du,
        "kitchen": _handle_kitchen,
    }

    for dataset, csv_path in csv_map.items():
        if not csv_path.exists():
            print(f"[warn] csv not found, skip: {csv_path}")
            continue
        with csv_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"[start] {dataset}: total {len(rows)} images")
        for row in tqdm(rows, desc=dataset, ncols=80):
            try:
                handlers[dataset](row, model, outputs, manifest)
            except Exception as exc:
                print(f"[ERR] {dataset} row failed: {row}\n  -> {exc!r}")
            sweep_cuda()
            gc.collect()
        print(f"[DONE] {dataset}: processed={len(rows)}")


if __name__ == "__main__":
    run()
