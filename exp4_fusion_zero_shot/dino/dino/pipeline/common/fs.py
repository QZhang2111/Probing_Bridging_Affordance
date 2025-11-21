"""Filesystem utilities shared across experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path | str) -> None:
    """Create a directory (and parents) if it does not already exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def append_jsonl(entries: Iterable[dict[str, Any]] | dict[str, Any], manifest_path: Path | str) -> None:
    """Append one or more JSON entries to a manifest file."""

    if isinstance(entries, dict):
        iterable = [entries]
    else:
        iterable = entries

    path = Path(manifest_path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        for entry in iterable:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
