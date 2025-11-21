#!/usr/bin/env python
"""Create training curves and qualitative galleries from saved artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.visualization.plots import (
    plot_final_metrics,
    plot_training_curves,
    plot_step_curves,
    save_prediction_gallery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing summary.json and saved examples")
    parser.add_argument(
        "--defaults",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the default configuration (for palette/class names).",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=PROJECT_ROOT / "configs" / "local.yaml",
        help="Optional override configuration file.",
    )
    return parser.parse_args()


STEP_PATTERNS = {
    "train": re.compile(r"train step\s+(\d+)/(\d+)\s+\|\s+loss=([0-9.eE+-]+)\s+\|\s+mIoU=([0-9.eE+-]+)"),
    "val": re.compile(r"val step\s+(\d+)/(\d+)\s+\|\s+loss=([0-9.eE+-]+)\s+\|\s+mIoU=([0-9.eE+-]+)"),
    "test": re.compile(r"test step\s+(\d+)/(\d+)\s+\|\s+loss=([0-9.eE+-]+)\s+\|\s+mIoU=([0-9.eE+-]+)"),
}


def _extract_step_curves(log_path: Path, splits: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    data = {split: {"steps": [], "loss": [], "miou": []} for split in splits}
    patterns = {split: STEP_PATTERNS[split] for split in splits if split in STEP_PATTERNS}
    if not patterns:
        return data

    prev_step = {split: 0 for split in patterns}
    offsets = {split: 0 for split in patterns}

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for split, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(3))
                    miou = float(match.group(4))

                    if step < prev_step[split]:
                        offsets[split] += prev_step[split]

                    global_step = offsets[split] + step

                    data[split]["steps"].append(global_step)
                    data[split]["loss"].append(loss)
                    data[split]["miou"].append(miou)

                    prev_step[split] = step
                    break
    return data


def main() -> None:
    args = parse_args()
    local_path = args.local if args.local.exists() else None
    config = load_config(args.defaults, local_path)
    vis_cfg = config.get("visualization", {})
    class_names = config["dataset"].get("class_names")
    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(config["dataset"]["num_classes"])]

    summary = None
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

    history = None
    history_path = args.run_dir / "training_history.json"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)

    test_metrics = summary.get("test_metrics") if summary else None

    if history:
        plot_training_curves(history, args.run_dir, test_metrics=test_metrics)
        train_tail = history[-1]
        train_metrics = {
            "loss": train_tail.get("train_loss"),
            "miou": train_tail.get("train_miou"),
        }
        val_metrics = summary.get("best_val") if summary else {
            "loss": train_tail.get("val_loss"),
            "miou": train_tail.get("val_miou"),
        }
    else:
        train_metrics = None
        val_metrics = summary.get("best_val") if summary else None

    plot_final_metrics(train_metrics, val_metrics, test_metrics, args.run_dir)

    log_path = None
    if summary:
        hyper = summary.get("best_hyperparams")
        if hyper:
            lr = hyper.get("lr")
            wd = hyper.get("weight_decay")
            if lr is not None and wd is not None:
                candidate = args.run_dir / f"lr{lr:.0e}_wd{wd:.0e}" / "train.log"
                if candidate.exists():
                    log_path = candidate
    if log_path is None:
        for candidate in sorted(args.run_dir.glob("lr*_wd*/train.log")):
            if candidate.exists():
                log_path = candidate
                break

    train_curves = None
    val_curves = None
    test_curves = None

    if log_path is not None:
        curves = _extract_step_curves(log_path, ("train", "val"))
        train_curves = curves.get("train")
        val_curves = curves.get("val")

        test_log = log_path.parent / "eval_test.log"
        if test_log.exists():
            test_curves = _extract_step_curves(test_log, ("test",)).get("test")

    def _sanitize(curves: dict | None) -> dict | None:
        if not curves:
            return None
        if not curves.get("steps"):
            return None
        return curves

    train_curves = _sanitize(train_curves)
    val_curves = _sanitize(val_curves)
    test_curves = _sanitize(test_curves)

    plot_step_curves(train_curves, val_curves, test_curves, args.run_dir)

    alpha = vis_cfg.get("overlay_alpha", 0.5)
    val_examples_path = args.run_dir / "val_examples.pt"
    if val_examples_path.exists():
        examples = torch.load(val_examples_path)
        save_prediction_gallery(
            examples,
            args.run_dir / "val_visuals",
            class_names,
            alpha=alpha,
        )

    test_examples_path = args.run_dir / "test_examples.pt"
    if test_examples_path.exists():
        examples = torch.load(test_examples_path)
        save_prediction_gallery(
            examples,
            args.run_dir / "test_visuals",
            class_names,
            alpha=alpha,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
