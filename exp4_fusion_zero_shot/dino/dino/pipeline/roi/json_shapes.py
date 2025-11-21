"""Helpers for parsing ROI annotations stored as JSON polygons/bboxes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ROIShape:
    name: str
    shape: str
    coords: List[float]


def load_roi_shapes(path: Path) -> List[ROIShape]:
    data = json.loads(path.read_text())
    shapes = data.get("shapes", data)
    rois: List[ROIShape] = []
    for entry in shapes:
        rois.append(
            ROIShape(
                name=str(entry.get("name", f"roi_{len(rois)}")),
                shape=str(entry.get("shape", entry.get("type", "rect"))).lower(),
                coords=list(entry.get("coords", entry.get("points", entry.get("bbox", [])))),
            )
        )
    return rois
