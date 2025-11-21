#!/usr/bin/env python
"""Generate train/val/test split JSON for the UMD affordance dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import parse_category_split, train_val_test_split, save_split_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the UMD dataset root (folder containing `tools/`).",
    )
    parser.add_argument(
        "--category-split",
        type=Path,
        required=True,
        help="Path to category_split.txt",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Portion of training tools used as validation (default: 0.1).",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=42,
        help="Random seed for selecting validation tools (default: 42).",
    )
    parser.add_argument(
        "--ensure-val-all-classes",
        action="store_true",
        help="Ensure the validation split covers all affordance classes.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Total number of affordance classes (required if ensuring class coverage).",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Ignore index used in label masks (default: 255).",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude background label (0) from the coverage requirement.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1024,
        help="Maximum attempts when searching for a validation subset that meets coverage constraints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to store the resulting split JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = parse_category_split(args.category_split)
    mapping = train_val_test_split(
        category_entries=entries,
        dataset_root=args.dataset_root,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed,
        ensure_val_all_classes=args.ensure_val_all_classes,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        exclude_background=args.exclude_background,
        max_attempts=args.max_attempts,
    )
    save_split_mapping(mapping, args.output)
    print(
        "Saved split mapping with "
        f"{len(mapping['train'])} train / {len(mapping['val'])} val / {len(mapping['test'])} test samples "
        f"to {args.output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
