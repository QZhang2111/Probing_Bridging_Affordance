#!/usr/bin/env python
"""Evaluate a saved linear probe checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.eval import evaluate_linear_probe
from src.engine.trainer import LinearProbeExperiment
from src.utils.config import load_config
from src.utils.logging import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the saved checkpoint (.pth)")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the default configuration file.",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=PROJECT_ROOT / "configs" / "local.yaml",
        help="Optional override configuration file.",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="If set, dumps qualitative examples next to the checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = args.local if args.local.exists() else None
    config = load_config(args.defaults, local_path)
    experiment = LinearProbeExperiment(config)

    eval_logger = create_logger(
        args.checkpoint.parent,
        name=f"linear_probe_eval.{args.split}",
        filename=f"eval_{args.split}.log",
    )

    head = experiment._build_head().to(experiment.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    head.load_state_dict(state_dict)

    loader = experiment.val_loader if args.split == "val" else experiment.test_loader

    metrics, examples = evaluate_linear_probe(
        experiment.backbone,
        head,
        loader,
        experiment.device,
        precision=experiment.training_cfg.get("precision", "bf16"),
        num_classes=experiment.num_classes,
        ignore_index=experiment.ignore_index,
        criterion=torch.nn.CrossEntropyLoss(ignore_index=experiment.ignore_index),
        target_layer=experiment.target_layer,
        max_examples=experiment.cfg.get("visualization", {}).get("num_samples", 0),
        logger=eval_logger,
        log_interval=experiment.val_log_interval,
        ignore_indices=experiment.metric_ignore_indices,
        split=args.split,
        use_multi_head=experiment.use_multi_head,
    )

    eval_logger.info("%s metrics: %s", args.split, metrics)

    metrics_path = args.checkpoint.with_suffix(f".{args.split}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))

    if args.save_examples and examples:
        out_path = args.checkpoint.with_suffix(".examples.pt")
        torch.save(examples, out_path)
        print(f"Saved qualitative examples to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
