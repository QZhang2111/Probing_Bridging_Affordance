"""Configuration loader supporting multiple feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Normalization:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclass(frozen=True)
class OverlayConfig:
    cmap: str
    alpha: float
    low_pct: float
    high_pct: float


@dataclass(frozen=True)
class ModelSpec:
    key: str
    type: str
    target_size: Tuple[int, int]
    patch_size: int
    device: str
    normalization: Normalization
    keep_full_resolution: bool
    use_letterbox: bool = False
    pad_to_patch_multiple: bool = True
    images_override: Optional[Dict[str, Path]] = None
    repo: Optional[Path] = None
    checkpoint: Optional[Path] = None
    hub_model: Optional[str] = None
    model_id: Optional[str] = None
    prompt: Optional[str] = None
    time_step: Optional[int] = None
    primary_layer: Optional[str] = None
    layers_to_hook: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path
    mask_path: Optional[Path]
    use_mask: bool
    anchor_pixel: Tuple[int, int]
    images: Dict[str, Path]
    overlay: OverlayConfig
    device: str
    models: Dict[str, ModelSpec]

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentConfig":
        cfg_path = path.resolve()
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("settings YAML must be a mapping")

        paths_cfg = _expect_mapping(data.get("paths", {}), "paths")
        output_root = _resolve_path(cfg_path, paths_cfg.get("output_root", "../cache"))

        general_cfg = _expect_mapping(data.get("general", {}), "general")
        device = str(general_cfg.get("device", "cuda"))
        use_mask = bool(general_cfg.get("use_mask", False))
        mask_path = _resolve_optional_path(cfg_path, general_cfg.get("mask_path"))
        if use_mask and mask_path is None:
            raise ValueError("general.mask_path must be provided when use_mask is true")
        anchor = tuple(int(v) for v in _expect_sequence(general_cfg.get("anchor_pixel"), "general.anchor_pixel", length=2))

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

        def _parse_layers(cfg_value: Any, key_name: str) -> Optional[Tuple[int, ...]]:
            if cfg_value is None:
                return None
            seq = _expect_sequence(cfg_value, key_name)
            return tuple(int(v) for v in seq)

        def _clone_kwargs(base: Dict[str, Any]) -> Dict[str, Any]:
            cloned = dict(base)
            images_override = base.get("images_override")
            if images_override is not None:
                cloned["images_override"] = dict(images_override)
            return cloned

        for item in model_list:
            model_cfg = _expect_mapping(item, "models[]")
            key = str(model_cfg.get("key"))
            if not key:
                raise ValueError("Each model entry must include a non-empty 'key'")
            mtype = str(model_cfg.get("type", "dinov3"))
            target_size = tuple(int(v) for v in _expect_sequence(model_cfg.get("target_size"), f"models[{key}].target_size", length=2))
            patch_size = int(model_cfg.get("patch_size", 16))
            normalization_cfg = model_cfg.get("normalization")
            if normalization_cfg is None:
                normalization = Normalization(mean=DEFAULT_MEAN, std=DEFAULT_STD)
            else:
                norm_map = _expect_mapping(normalization_cfg, f"models[{key}].normalization")
                mean = tuple(float(v) for v in _expect_sequence(norm_map.get("mean"), f"models[{key}].normalization.mean", length=3))
                std = tuple(float(v) for v in _expect_sequence(norm_map.get("std"), f"models[{key}].normalization.std", length=3))
                normalization = Normalization(mean=mean, std=std)

            repo = model_cfg.get("repo")
            checkpoint = model_cfg.get("checkpoint")
            hub_model = model_cfg.get("hub_model")
            model_id = model_cfg.get("model_id")
            keep_full = bool(model_cfg.get("keep_full_resolution", False))
            use_letterbox = bool(model_cfg.get("use_letterbox", False))
            pad_to_patch = bool(model_cfg.get("pad_to_patch_multiple", True))
            layers_to_hook = _parse_layers(model_cfg.get("layers_to_hook"), f"models[{key}].layers_to_hook")

            images_override_cfg = model_cfg.get("images_override")
            images_override: Optional[Dict[str, Path]] = None
            if images_override_cfg is not None:
                override_map = _expect_mapping(images_override_cfg, f"models[{key}].images_override")
                images_override = {}
                for name, value in override_map.items():
                    images_override[name] = _resolve_path(cfg_path, value)
            prompt = model_cfg.get("prompt")
            time_step = model_cfg.get("time_step")
            primary_layer = model_cfg.get("primary_layer")

            base_kwargs: Dict[str, Any] = {
                "key": key,
                "type": mtype,
                "target_size": target_size,
                "patch_size": patch_size,
                "device": str(model_cfg.get("device", device)),
                "normalization": normalization,
                "keep_full_resolution": keep_full,
                "use_letterbox": use_letterbox,
                "pad_to_patch_multiple": pad_to_patch,
                "images_override": images_override,
                "repo": _resolve_optional_path(cfg_path, repo),
                "checkpoint": _resolve_optional_path(cfg_path, checkpoint),
                "hub_model": str(hub_model) if hub_model is not None else None,
                "model_id": str(model_id) if model_id is not None else None,
                "prompt": str(prompt) if prompt is not None else None,
                "time_step": int(time_step) if time_step is not None else None,
                "primary_layer": str(primary_layer) if primary_layer is not None else None,
                "layers_to_hook": layers_to_hook,
            }

            models[key] = ModelSpec(**base_kwargs)

            layer_variants = model_cfg.get("layer_variants")
            if layer_variants is not None:
                variants = _expect_sequence(layer_variants, f"models[{key}].layer_variants")
                for entry in variants:
                    var_map = _expect_mapping(entry, "layer_variants[]")
                    suffix = str(var_map.get("suffix", "")).strip()
                    if not suffix:
                        raise ValueError(f"layer_variants for '{key}' must include a non-empty 'suffix'")
                    new_key = f"{key}{suffix}"
                    variant_kwargs = _clone_kwargs(base_kwargs)
                    variant_kwargs["key"] = new_key
                    if "primary_layer" in var_map:
                        variant_kwargs["primary_layer"] = str(var_map["primary_layer"])
                    if "prompt" in var_map:
                        variant_kwargs["prompt"] = str(var_map["prompt"])
                    if "time_step" in var_map:
                        variant_kwargs["time_step"] = int(var_map["time_step"])
                    if "layers_to_hook" in var_map:
                        variant_kwargs["layers_to_hook"] = _parse_layers(var_map["layers_to_hook"], f"models[{key}].layer_variants[{suffix}].layers_to_hook")
                    models[new_key] = ModelSpec(**variant_kwargs)

        return cls(
            output_root=output_root,
            mask_path=mask_path,
            use_mask=use_mask,
            anchor_pixel=(int(anchor[0]), int(anchor[1])),
            images=images,
            overlay=overlay,
            device=device,
            models=models,
        )

    def list_model_keys(self) -> List[str]:
        return list(self.models.keys())

    def get_model(self, key: str) -> ModelSpec:
        try:
            return self.models[key]
        except KeyError as exc:
            raise KeyError(f"Unknown model key '{key}'. Available: {', '.join(self.models.keys())}") from exc

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
    return _resolve_optional_path(cfg_path, value)


def _resolve_optional_path(cfg_path: Path, value: Any) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (cfg_path.parent / path).resolve()
    return path
