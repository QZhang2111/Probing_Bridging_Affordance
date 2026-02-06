from __future__ import annotations

import numpy as np


def cal_kl(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    return float(np.sum(map2 * np.log(map2 / (map1 + eps) + eps)))


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)
    return float(np.sum(intersection))


def cal_nss(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    if std < eps:
        return 0.0
    smap = (pred - np.mean(pred)) / std
    if np.max(gt) - np.min(gt) < eps:
        return 0.0
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + eps)
    fixation_map = (fixation_map > 0.1).astype(np.float32)
    denom = fixation_map.sum()
    if denom < eps:
        return 0.0
    return float(np.sum(smap * fixation_map) / (denom + eps))
