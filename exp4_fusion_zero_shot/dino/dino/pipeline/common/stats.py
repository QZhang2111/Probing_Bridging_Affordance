"""Statistical helpers used across visualizations."""

from __future__ import annotations

import numpy as np


def percentile_stretch(x: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float, float]:
    """Normalize values to [0,1] using percentile-based min/max."""

    lo = np.percentile(x, low)
    hi = np.percentile(x, high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32), float(lo), float(hi)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y.astype(np.float32), float(lo), float(hi)
