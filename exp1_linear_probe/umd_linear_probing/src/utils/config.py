"""Lightweight configuration loader with optional local overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["Config", "load_config"]


@dataclass(frozen=True)
class Config:
    data: Dict[str, Any]

    def __getitem__(self, item: str) -> Any:  # Enables ``config["dataset"]``.
        return self.data[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration file {path} must contain a mapping at the top level")
    return data


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(base_dir: Path, candidate: Optional[str]) -> Optional[Path]:
    if candidate in (None, ""):
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    dataset_cfg = config.get("dataset") or {}
    if "root" in dataset_cfg:
        dataset_cfg["root"] = _resolve_path(base_dir, dataset_cfg["root"])
    if "split_path" in dataset_cfg:
        dataset_cfg["split_path"] = _resolve_path(base_dir, dataset_cfg["split_path"])

    model_cfg = config.get("model") or {}
    params = model_cfg.get("params") if isinstance(model_cfg, dict) else None
    if isinstance(params, dict):
        if "repo_path" in params:
            params["repo_path"] = _resolve_path(base_dir, params["repo_path"])
        if "checkpoint_path" in params:
            params["checkpoint_path"] = _resolve_path(base_dir, params["checkpoint_path"])
        if "config_path" in params:
            params["config_path"] = _resolve_path(base_dir, params["config_path"])
        if "model_dir" in params:
            params["model_dir"] = _resolve_path(base_dir, params["model_dir"])

    train_cfg = config.get("training") or {}
    if "output_dir" in train_cfg:
        train_cfg["output_dir"] = _resolve_path(base_dir, train_cfg["output_dir"])
    if "output_root" in train_cfg:
        train_cfg["output_root"] = _resolve_path(base_dir, train_cfg["output_root"])

    return config


def load_config(
    default_path: Path,
    local_override: Optional[Path] = None,
) -> Config:
    base_dir = default_path.parent.parent  # project root (configs/..)
    config = _read_yaml(default_path)
    if local_override is not None:
        overrides = _read_yaml(local_override)
        config = _deep_update(config, overrides)

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        section = dict(model_cfg)
        target = section.pop("target", None) or section.pop("name", None)
        config_path_value = section.pop("config_path", None)
        overrides = section.pop("overrides", {})
        user_params = section.pop("params", None)

        template: Dict[str, Any] = {}
        if config_path_value is not None:
            resolved_model_path = _resolve_path(base_dir, config_path_value)
            if resolved_model_path is None:
                raise FileNotFoundError(f"Model config path {config_path_value!r} could not be resolved")
            template = _read_yaml(resolved_model_path)

        # Prefer explicit 'params' sub-mapping when provided; otherwise treat the remaining
        # keys in the model section as params (backward-compatible with template style).
        if isinstance(user_params, dict):
            params = _deep_update(dict(template), user_params)
            # Any remaining keys in 'section' are treated as additional param overrides
            if section:
                params = _deep_update(params, section)
        else:
            params = _deep_update(dict(template), section)
        if overrides:
            params = _deep_update(params, overrides)

        if target is None:
            target = params.get("name")
        if target is None:
            raise ValueError("Model configuration must specify a 'target' or include a name in the template.")

        config["model"] = {"target": target, "params": params}

    config = _resolve_paths(config, base_dir)
    return Config(data=config)
