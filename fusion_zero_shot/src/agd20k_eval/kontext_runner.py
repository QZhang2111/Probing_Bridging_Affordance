from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_VIS_MODULE = None
_WORKERS: Dict[str, object] = {}


def _load_visualizer_module(script_path: Path):
    global _VIS_MODULE
    if _VIS_MODULE is not None:
        return _VIS_MODULE

    module_name = "fusion_zero_shot_flux_kontext_visualizer"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load visualizer module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _VIS_MODULE = module
    return module


def _get_worker(script_path: Path, model_dir: Path):
    module = _load_visualizer_module(script_path)
    key = str(model_dir.resolve())
    worker = _WORKERS.get(key)
    if worker is None:
        logging.info("Creating persistent Kontext worker for %s", key)
        worker = module.KontextWorker(model_dir=key, device="cuda")
        _WORKERS[key] = worker
    return module, worker


def close_kontext_workers() -> None:
    for worker in _WORKERS.values():
        close = getattr(worker, "close", None)
        if callable(close):
            close()
    _WORKERS.clear()


def run_kontext_generation(
    script_path: Path,
    model_dir: Path,
    image_path: Path,
    prompt: str,
    output_root: Path,
    num_steps: int,
    guidance: float,
    seed: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run Kontext generation and return key output paths.

    The default path uses a persistent in-process worker so the FLUX pipeline
    stays resident on GPU across samples.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    module, worker = _get_worker(script_path, model_dir)
    logging.info(
        "Running Kontext generation in persistent worker: model=%s image=%s prompt=%r",
        model_dir,
        image_path,
        prompt,
    )
    result = module.run_visualization(
        model_dir=str(model_dir),
        image_path=str(image_path),
        prompt=prompt,
        output_root=str(output_root),
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        device="cuda",
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        worker=worker,
    )
    return result
