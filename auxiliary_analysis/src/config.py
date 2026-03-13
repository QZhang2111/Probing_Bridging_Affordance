"""Lightweight configuration loader for auxiliary analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class OverlayConfig:
    cmap: str
    alpha: float
    low_pct: float
    high_pct: float


@dataclass(frozen=True)
class ModelSpec:
    key: str


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path
    anchor_pixel: Tuple[int, int]
    images: Dict[str, Path]
    overlay: OverlayConfig
    models: Dict[str, ModelSpec]

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentConfig":
        cfg_path = path.resolve()
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("settings YAML must be a mapping")

        paths_cfg = _expect_mapping(data.get("paths", {}), "paths")
        output_root = _resolve_path(cfg_path, paths_cfg.get("output_root"))

        general_cfg = _expect_mapping(data.get("general", {}), "general")
        anchor = tuple(
            int(v) for v in _expect_sequence(general_cfg.get("anchor_pixel"), "general.anchor_pixel", length=2)
        )

        images_cfg = _expect_mapping(general_cfg.get("images", {}), "general.images")
        images: Dict[str, Path] = {}
        for name, value in images_cfg.items():
            images[name] = _resolve_path(cfg_path, value)

        overlay_cfg = _expect_mapping(general_cfg.get("overlay", {}), "general.overlay")
        overlay = OverlayConfig(
            cmap=str(overlay_cfg.get("cmap", "viridis")),
            alpha=float(overlay_cfg.get("alpha", 0.6)),
            low_pct=float(overlay_cfg.get("low_pct", 1.0)),
            high_pct=float(overlay_cfg.get("high_pct", 99.0)),
        )

        model_list = _expect_sequence(data.get("models", []), "models")
        models: Dict[str, ModelSpec] = {}
        for item in model_list:
            model_cfg = _expect_mapping(item, "models[]")
            key = str(model_cfg.get("key", "")).strip()
            if not key:
                raise ValueError("Each model entry must include non-empty key")
            models[key] = ModelSpec(key=key)

        return cls(
            output_root=output_root,
            anchor_pixel=(int(anchor[0]), int(anchor[1])),
            images=images,
            overlay=overlay,
            models=models,
        )

    def get_model(self, key: str) -> ModelSpec:
        try:
            return self.models[key]
        except KeyError as exc:
            keys = ", ".join(self.models.keys())
            raise KeyError(f"Unknown model key '{key}'. Available: {keys}") from exc

    def iter_models(self, keys: Optional[Iterable[str]] = None) -> Iterable[ModelSpec]:
        if keys is None:
            return self.models.values()
        return (self.get_model(k) for k in keys)


def _expect_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _expect_sequence(value: Any, name: str, *, length: Optional[int] = None) -> List[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    seq = list(value)
    if length is not None and len(seq) != length:
        raise ValueError(f"{name} must have length {length}")
    return seq


def _resolve_path(cfg_path: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("required path is missing in configuration")
    path = Path(str(value))
    if not path.is_absolute():
        path = (cfg_path.parent / path).resolve()
    return path
