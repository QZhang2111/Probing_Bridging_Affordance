#!/usr/bin/env python3
"""
Compute KLD / SIM / NSS against AGD20K ground-truth maps in a training-free manner.

Example
-------
python eval_agd20k_metrics.py \
    --pred /path/to/heatmap.png \
    --gt /home/li325/qing_workspace/dataset/AGD20K/AGD20K/Seen/testset/GT/hold/toothbrush/toothbrush_001764.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "FLUX" / "AGD20K_Flux_unseen"
sys.path.append(str(METRICS_DIR))

from metrics import cal_kl, cal_sim, cal_nss  # type: ignore  # noqa: E402


def _load_heatmap(path: Path) -> np.ndarray:
    """Load an image/heatmap as float32 grayscale."""

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(path)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array = array.astype(np.float32)
    if np.max(array) <= 1.0 + 1e-6:
        array *= 255.0
    array = np.clip(array, 0.0, None)
    return array


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.clip(pred, 0.0, None)
    gt = np.clip(gt, 0.0, None)
    return {
        "kld": cal_kl(pred, gt),
        "sim": cal_sim(pred, gt),
        "nss": cal_nss(pred, gt),
    }


def parse_args() -> argparse.Namespace:
    default_gt = (
        Path("/home/li325/qing_workspace/dataset/AGD20K/AGD20K/Seen/testset/GT/hold/toothbrush/toothbrush_001764.png")
    )
    parser = argparse.ArgumentParser(description="Evaluate AGD20K metrics for a predicted heatmap.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted heatmap or saliency map path.")
    parser.add_argument("--gt", type=Path, default=default_gt, help="Ground-truth heatmap path.")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about shapes and normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = args.pred.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()

    pred = _load_heatmap(pred_path)
    gt = _load_heatmap(gt_path)

    if pred.shape != gt.shape:
        if args.verbose:
            print(f"[warn] pred shape {pred.shape} != gt shape {gt.shape}; resizing prediction to GT.")
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    metrics = compute_metrics(pred, gt)

    print("[metrics]")
    for name, value in metrics.items():
        print(f"  {name.upper():>3} = {value:.6f}")

    if args.save_json is not None:
        output_path = args.save_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump({"prediction": str(pred_path), "gt": str(gt_path), "metrics": metrics}, fh, indent=2)
        if args.verbose:
            print(f"[info] metrics written to {output_path}")


if __name__ == "__main__":
    main()
