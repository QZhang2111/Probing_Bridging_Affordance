"""PCA helpers re-exported from shared common utilities."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.pca import SubspaceModel, apply_percentile_bounds, project_tokens, upsample_components

__all__ = [
    "SubspaceModel",
    "apply_percentile_bounds",
    "project_tokens",
    "upsample_components",
]
