"""Segmentation metrics used for monitoring training."""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["ConfusionMatrix", "update_confusion_matrix", "compute_iou"]

ConfusionMatrix = torch.Tensor


def update_confusion_matrix(
    confmat: ConfusionMatrix,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> ConfusionMatrix:
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    if preds.numel() == 0:
        return confmat
    indices = targets * num_classes + preds
    hist = torch.bincount(indices, minlength=num_classes ** 2)
    confmat = confmat + hist.reshape(num_classes, num_classes)
    return confmat


def compute_iou(confmat: ConfusionMatrix) -> Dict[str, torch.Tensor]:
    true_pos = torch.diag(confmat)
    false_pos = confmat.sum(dim=0) - true_pos
    false_neg = confmat.sum(dim=1) - true_pos
    denom = true_pos + false_pos + false_neg
    iou = true_pos / torch.clamp(denom, min=1)
    miou = torch.nanmean(iou)
    return {"per_class": iou, "miou": miou}
