"""ROI normalization utilities bridging dataset annotations and token grids."""

from .structures import ROI, ROISelection, ROIBatch, SelectionStrategy  # noqa: F401
from .utils import (
    dilate_patch_mask,
    mask_to_token_ratios,
    ratios_to_indices,
    compute_soft_weights,
)  # noqa: F401
from .umd import load_umd_affordance_mask  # noqa: F401
from .json_shapes import load_roi_shapes  # noqa: F401
from .selection import (  # noqa: F401
    selection_from_ratios,
    selection_from_token_mask,
    selection_from_mask_tokens,
)
from .visualize import (  # noqa: F401
    overlay_mask_on_image,
    render_mask_overlay_from_npz,
    save_mask_overlays_for_first_instances,
)
