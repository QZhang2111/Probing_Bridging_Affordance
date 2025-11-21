"""Backbone namespace for linear probing experiments.

Avoid importing heavyweight/optional dependencies at module import time.
Stable Diffusion depends on diffusers/torch features that may be unavailable
in some environments; import it lazily/optionally so other backbones work.
"""

from .dino import DINOBackbone, DINOv3Backbone
from .dinov2 import DINOv2Backbone
from .flux import FluxBackbone
from .openclip import OpenCLIPBackbone
from .siglip2 import SigLIP2Backbone
from .sam import SAMBackbone
from .linear_head import MultiLayerLinearHead

__all__ = [
    "DINOBackbone",
    "DINOv3Backbone",
    "DINOv2Backbone",
    "FluxBackbone",
    "OpenCLIPBackbone",
    "SigLIP2Backbone",
    "SAMBackbone",
    "MultiLayerLinearHead",
]

# Optional: Stable Diffusion backbone
try:  # pragma: no cover - optional dependency
    from .stable_diffusion import StableDiffusionBackbone  # type: ignore
except Exception:
    StableDiffusionBackbone = None  # type: ignore
else:
    __all__.append("StableDiffusionBackbone")
