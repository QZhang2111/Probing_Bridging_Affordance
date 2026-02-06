from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


@dataclass
class SampleEntry:
    affordance: str
    object_name: str
    image_path: Path
    gt_path: Path


def iter_agd20k_samples(
    dataset_root: Path,
    affordances: Optional[Iterable[str]] = None,
    max_per_object: Optional[int] = None,
) -> Iterator[SampleEntry]:
    """
    遍历 AGD20K Unseen/testset 结构。

    dataset_root 目录结构:
      - egocentric/<affordance>/<object>/<image>
      - GT/<affordance>/<object>/<mask>
    """
    egocentric_root = dataset_root / "egocentric"
    gt_root = dataset_root / "GT"

    if not egocentric_root.exists():
        raise FileNotFoundError(f"Egocentric root not found: {egocentric_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    affordance_list = sorted(p.name for p in egocentric_root.iterdir() if p.is_dir())
    if affordances is not None:
        allow = set(affordances)
        affordance_list = [a for a in affordance_list if a in allow]

    for aff in affordance_list:
        aff_dir = egocentric_root / aff
        for obj_dir in sorted(p for p in aff_dir.iterdir() if p.is_dir()):
            obj = obj_dir.name
            gt_obj_dir = gt_root / aff / obj
            if not gt_obj_dir.exists():
                continue

            images = sorted(
                [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in obj_dir.glob(ext)]
            )
            if max_per_object is not None:
                images = images[:max_per_object]

            for img_path in images:
                gt_candidates = [
                    gt_obj_dir / (img_path.stem + ext)
                    for ext in (".png", ".jpg", ".jpeg")
                ]
                gt_path = next((p for p in gt_candidates if p.exists()), None)
                if gt_path is None:
                    continue
                yield SampleEntry(
                    affordance=aff,
                    object_name=obj,
                    image_path=img_path,
                    gt_path=gt_path,
                )
