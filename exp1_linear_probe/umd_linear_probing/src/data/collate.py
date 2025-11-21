"""Custom collate functions."""

from __future__ import annotations

from typing import Any, Dict, List

from torch.utils.data._utils.collate import default_collate

__all__ = ["collate_with_meta"]


def collate_with_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = [item.pop("meta") for item in batch]
    collated = {key: default_collate([item[key] for item in batch]) for key in batch[0].keys()}
    collated["meta"] = meta
    return collated
