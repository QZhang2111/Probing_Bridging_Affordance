"""Backbone namespace for linear probing experiments.

Avoid importing heavyweight/optional dependencies at module import time.
"""

from .dino import DINOBackbone, DINOv3Backbone
from .dinov2 import DINOv2Backbone
from .openclip import OpenCLIPBackbone
from .sam import SAMBackbone
from .linear_head import MultiLayerLinearHead

__all__ = [
    "DINOBackbone",
    "DINOv3Backbone",
    "DINOv2Backbone",
    "OpenCLIPBackbone",
    "SAMBackbone",
    "MultiLayerLinearHead",
]

# Optional: SigLIP2 backbone (depends on transformers)
try:  # pragma: no cover - optional dependency
    from .siglip2 import SigLIP2Backbone  # type: ignore
except Exception:
    SigLIP2Backbone = None  # type: ignore
else:
    __all__.append("SigLIP2Backbone")

# Optional: Flux backbone (depends on transformers + flash_attn)
try:  # pragma: no cover - optional dependency
    from .flux import FluxBackbone  # type: ignore
except Exception:
    FluxBackbone = None  # type: ignore
else:
    __all__.append("FluxBackbone")

# Optional: Stable Diffusion backbone
try:  # pragma: no cover - optional dependency
    from .stable_diffusion import StableDiffusionBackbone  # type: ignore
except Exception:
    StableDiffusionBackbone = None  # type: ignore
else:
    __all__.append("StableDiffusionBackbone")
