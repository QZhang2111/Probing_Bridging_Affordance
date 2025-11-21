"""Lightweight utilities for the Section2 affordance experiment pipeline."""

from .config import ExperimentConfig, ModelSpec, OverlayConfig  # noqa: F401
from .feature import ResizeMeta, create_extractor, letterbox_image, restore_original_resolution  # noqa: F401
from .io import (  # noqa: F401
    load_tokens_npz,
    save_image_colormap,
    save_image_gray,
    save_image_rgb,
    save_tokens_npz,
)
from .pca import (
    SubspaceModel,
    apply_percentile_bounds,
    embed_roi_tokens,
    fit_weighted_pca,
    project_tokens,
    scale_by_percentiles,
)  # noqa: F401
from .roi import ROISelection, letterbox_mask, mask_from_alpha, mask_to_roi  # noqa: F401
from .similarity import cosine_similarity, save_similarity_overlay  # noqa: F401
