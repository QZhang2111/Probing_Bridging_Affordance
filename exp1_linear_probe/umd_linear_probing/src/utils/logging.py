"""Structured logging setup for experiments."""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["create_logger"]


def _has_console_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return True
    return False


def create_logger(output_dir: Path, name: str = "linear_probe", filename: str = "train.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not _has_console_handler(logger):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    logfile = output_dir / filename
    if not _has_file_handler(logger, logfile):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
