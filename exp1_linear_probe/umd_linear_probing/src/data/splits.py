"""Utilities for parsing the UMD category split and producing train/val/test partitions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import scipy.io as sio

__all__ = [
    "CategorySplitEntry",
    "parse_category_split",
    "build_instance_index",
    "train_val_test_split",
    "save_split_mapping",
]


@dataclass(frozen=True)
class CategorySplitEntry:
    """Represents a single line in ``category_split.txt``."""

    split_id: int
    tool_name: str

    @classmethod
    def from_line(cls, line: str) -> "CategorySplitEntry":
        raw = line.strip().split()
        if len(raw) != 2:
            raise ValueError(f"Malformed line in category split: {line!r}")
        split_id = int(raw[0])
        tool_name = raw[1]
        return cls(split_id=split_id, tool_name=tool_name)


def parse_category_split(path: Path) -> List[CategorySplitEntry]:
    entries: List[CategorySplitEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(CategorySplitEntry.from_line(line))
    return entries


def build_instance_index(dataset_root: Path, tools: Sequence[str]) -> Dict[str, List[str]]:
    """Collect all frame identifiers (``tool_xx_########``) for the provided tools."""

    index: Dict[str, List[str]] = {}
    for tool in tools:
        tool_dir = dataset_root / "tools" / tool
        if not tool_dir.is_dir():
            raise FileNotFoundError(f"Missing tool directory: {tool_dir}")
        frame_ids: List[str] = []
        for rgb_path in sorted(tool_dir.glob("*_rgb.jpg")):
            frame_ids.append(rgb_path.stem.replace("_rgb", ""))
        if not frame_ids:
            raise RuntimeError(f"No RGB frames found for {tool_dir}")
        index[tool] = frame_ids
    return index


def train_val_test_split(
    category_entries: Sequence[CategorySplitEntry],
    dataset_root: Path,
    val_ratio: float = 0.1,
    val_seed: int = 42,
    *,
    ensure_val_all_classes: bool = False,
    num_classes: int | None = None,
    ignore_index: int = 255,
    exclude_background: bool = False,
    max_attempts: int = 1024,
) -> Dict[str, List[Dict[str, str]]]:
    """Split instances into train/val/test following the UMD protocol.

    ``category_split.txt`` marks ``split_id == 1`` as the training pool and ``2`` as the
    test pool. We further carve out a validation slice from the training pool by drawing
    complete tool instances with a fixed random seed for reproducibility.
    """

    if not (0.0 < val_ratio < 0.5):
        raise ValueError("Validation ratio should be within (0, 0.5) for stability")

    train_tools = [e.tool_name for e in category_entries if e.split_id == 1]
    test_tools = [e.tool_name for e in category_entries if e.split_id == 2]

    train_index = build_instance_index(dataset_root, train_tools)
    test_index = build_instance_index(dataset_root, test_tools)

    target_classes: Set[int] | None = None
    tool_class_map: Dict[str, Set[int]] = {}
    if ensure_val_all_classes:
        if num_classes is None:
            raise ValueError("num_classes must be provided when ensure_val_all_classes is enabled.")
        target_classes = set(range(num_classes))
        if exclude_background:
            target_classes.discard(0)
        if not target_classes:
            raise ValueError("Target classes empty after applying exclude_background filter.")
        tool_class_map = _compute_tool_class_map(
            dataset_root,
            train_index,
            ignore_index=ignore_index,
            target_classes=target_classes,
        )
        observed_union = set().union(*tool_class_map.values())
        if not target_classes.issubset(observed_union):
            missing = sorted(target_classes.difference(observed_union))
            raise RuntimeError(
                f"Validation coverage requirement impossible; missing classes {missing} in training pool."
            )

    rng = random.Random(val_seed)
    val_tool_count = max(1, int(round(len(train_tools) * val_ratio)))
    attempts = 0
    while True:
        if attempts >= max_attempts:
            raise RuntimeError(
                "Unable to find validation subset covering required classes within max attempts."
            )
        attempts += 1
        val_tools = rng.sample(train_tools, val_tool_count)
        if not ensure_val_all_classes:
            break
        observed: Set[int] = set()
        for tool in val_tools:
            observed.update(tool_class_map.get(tool, set()))
            if target_classes.issubset(observed):
                break
        if target_classes.issubset(observed):
            break

    train_tools_eff = [t for t in train_tools if t not in val_tools]

    def expand_index(selected_tools: Iterable[str], index: Dict[str, List[str]]) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        for tool in sorted(selected_tools):
            for frame_id in index[tool]:
                samples.append(
                    {
                        "tool": tool,
                        "frame_id": frame_id,
                        "rgb": f"tools/{tool}/{frame_id}_rgb.jpg",
                        "depth": f"tools/{tool}/{frame_id}_depth.png",
                        "label_mat": f"tools/{tool}/{frame_id}_label.mat",
                    }
                )
        return samples

    split_mapping = {
        "train": expand_index(train_tools_eff, train_index),
        "val": expand_index(val_tools, train_index),
        "test": expand_index(test_tools, test_index),
    }
    return split_mapping


def save_split_mapping(mapping: Dict[str, List[Dict[str, str]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)


def _compute_tool_class_map(
    dataset_root: Path,
    tool_index: Dict[str, List[str]],
    *,
    ignore_index: int,
    target_classes: Set[int],
) -> Dict[str, Set[int]]:
    """Pre-compute observed affordance classes for each tool."""

    if not target_classes:
        return {tool: set() for tool in tool_index}

    target_min = min(target_classes)
    target_max = max(target_classes)
    tool_classes: Dict[str, Set[int]] = {}
    for tool, frames in tool_index.items():
        observed: Set[int] = set()
        for frame_id in frames:
            label_path = dataset_root / "tools" / tool / f"{frame_id}_label.mat"
            mat = sio.loadmat(label_path)
            mask = np.asarray(mat["gt_label"], dtype=np.int64)
            valid = mask != ignore_index
            classes = np.unique(mask[valid])
            for cls in classes:
                value = int(cls)
                if target_min <= value <= target_max and value in target_classes:
                    observed.add(value)
            if target_classes.issubset(observed):
                break
        tool_classes[tool] = observed
    return tool_classes
