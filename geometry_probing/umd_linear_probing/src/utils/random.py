"""Deterministic seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
