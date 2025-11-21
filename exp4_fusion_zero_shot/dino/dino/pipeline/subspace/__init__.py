"""PCA subspace utilities for training and projection."""

from .trainer import fit_weighted_pca, SubspaceModel  # noqa: F401
from .projector import (
    apply_percentile_bounds,
    embed_roi_tokens,
    project_tokens,
    rgb_from_components,
    scale_by_percentiles,
)  # noqa: F401
