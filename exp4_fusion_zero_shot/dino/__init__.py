"""DINO affordance experiment toolkit."""

from importlib import import_module
import sys


__all__ = []

# Bridge legacy ``dino.src`` imports to the top-level ``src`` package.
_src_module = import_module("src")
sys.modules.setdefault(__name__ + ".src", _src_module)
src = _src_module

__all__.append("src")
