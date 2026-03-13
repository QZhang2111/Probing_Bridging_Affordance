"""Helpers for exp2 auxiliary analysis."""

from .config import ExperimentConfig, ModelSpec, OverlayConfig
from .io import ensure_dir, load_tokens_npz, save_image_colormap, save_image_gray, save_image_rgb
from .pca import SubspaceModel, apply_percentile_bounds, project_tokens, upsample_components
from .resize import ResizeMeta, restore_original_resolution
from .similarity import cosine_similarity, save_similarity_overlay

__all__ = [
    "ExperimentConfig",
    "ModelSpec",
    "OverlayConfig",
    "ensure_dir",
    "load_tokens_npz",
    "save_image_colormap",
    "save_image_gray",
    "save_image_rgb",
    "SubspaceModel",
    "apply_percentile_bounds",
    "project_tokens",
    "upsample_components",
    "ResizeMeta",
    "restore_original_resolution",
    "cosine_similarity",
    "save_similarity_overlay",
]
