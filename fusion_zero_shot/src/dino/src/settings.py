"""Configuration loader for DINO experiments.

Loads a default YAML config and optionally merges overrides from
``configs/local.yaml``.  Paths declared in the config are resolved relative to
this ``dino`` package directory unless already absolute.  Users can provide a
local override file to point to custom checkpoint locations or datasets
without modifying the tracked defaults.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "defaults.yaml"
LOCAL_CONFIG_PATH = BASE_DIR / "configs" / "local.yaml"
ENV_CHECKPOINT = "DINO_CHECKPOINT_PATH"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(candidate: Any) -> Path | None:
    if candidate in (None, ""):
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


@dataclass(frozen=True)
class Settings:
    """Immutable wrapper that exposes resolved repository paths."""

    paths: Dict[str, Path]
    model_repo: Path
    model_checkpoint: Path | None

    def require_checkpoint(self) -> Path:
        if self.model_checkpoint is None:
            raise RuntimeError(
                "Model checkpoint path is not configured. Set it in"
                " configs/local.yaml or via the DINO_CHECKPOINT_PATH"
                " environment variable."
            )
        return self.model_checkpoint


_cached_settings: Settings | None = None


def load_settings(force_reload: bool = False) -> Settings:
    global _cached_settings
    if _cached_settings is not None and not force_reload:
        return _cached_settings

    config = _read_yaml(DEFAULT_CONFIG_PATH)
    overrides = _read_yaml(LOCAL_CONFIG_PATH)
    config = _deep_update(config, overrides)

    paths_cfg = config.get("paths") or {}
    if not isinstance(paths_cfg, dict):
        warnings.warn("settings.paths override must be a mapping; ignoring non-mapping value", RuntimeWarning)
        paths_cfg = {}
    resolved_paths = {k: _resolve_path(v) for k, v in paths_cfg.items()}

    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        warnings.warn("settings.model override must be a mapping; ignoring non-mapping value", RuntimeWarning)
        model_cfg = {}
    checkpoint = model_cfg.get("checkpoint_path") or os.getenv(ENV_CHECKPOINT)
    settings = Settings(
        paths={k: v for k, v in resolved_paths.items() if v is not None},
        model_repo=_resolve_path(model_cfg.get("repo_dir")) or BASE_DIR,
        model_checkpoint=_resolve_path(checkpoint),
    )
    _cached_settings = settings
    return settings


def get_settings() -> Settings:
    """Convenience alias used by callers expecting a cached configuration."""

    return load_settings()
