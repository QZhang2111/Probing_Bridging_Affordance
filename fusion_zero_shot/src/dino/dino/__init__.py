"""Top-level package exposing pipeline utilities and legacy modules."""

from importlib import import_module
import sys

__all__ = ["pipeline", "src"]

# Ensure legacy imports ``dino.src`` continue to resolve.
_src_module = import_module("src")
sys.modules.setdefault(__name__ + ".src", _src_module)
src = _src_module

# Explicitly import pipeline so ``dino.pipeline`` is available.
from . import pipeline  # noqa: E402,F401
