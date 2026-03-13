"""Evaluation utilities for the linear probe."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.metrics import compute_iou, update_confusion_matrix

__all__ = ["evaluate_linear_probe"]


def evaluate_linear_probe(
    backbone,
    head,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
    num_classes: int,
    ignore_index: int,
    criterion,
    target_layer: int,
    max_examples: int = 0,
    *,
    logger = None,
    log_interval: int = 0,
    ignore_indices: Iterable[int] | None = None,
    split: str = "val",
    use_multi_head: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, torch.Tensor]]]:
    backbone.eval()
    head.eval()

    total_loss = 0.0
    total_images = 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    examples: List[Dict[str, torch.Tensor]] = []

    total_steps = len(dataloader)

    def _mask_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not ignore_indices:
            return metrics
        per_class = metrics["per_class"].clone()
        for idx in ignore_indices:
            if 0 <= idx < per_class.numel():
                per_class[idx] = torch.nan
        metrics["per_class"] = per_class
        metrics["miou"] = torch.nanmean(per_class)
        return metrics

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            patch_targets = batch["patch_mask"].to(device, non_blocking=True)
            pixel_targets = batch["pixel_mask"].to(device, non_blocking=True)

            features = backbone(images, autocast_precision=precision)
            # Attach optional geometry features from batch
            for gk in ("geom_depth", "geom_normal"):
                if gk in batch and isinstance(batch[gk], torch.Tensor):
                    features[gk] = batch[gk].to(device, non_blocking=True)
            if use_multi_head:
                # If head expects geometry keys that are missing, fill zero placeholders (with batch dim)
                if hasattr(head, 'feature_keys') and hasattr(head, 'primary_key'):
                    primary = head.primary_key
                    ref = features.get(primary, next(iter(features.values())))
                    B = ref.shape[0] if ref.dim() == 4 else 1
                    grid = ref.shape[-2:]
                    for gk, ch in (("geom_depth", 1), ("geom_normal", 3)):
                        if gk in getattr(head, 'feature_keys', []) and gk not in features:
                            features[gk] = torch.zeros((B, ch, *grid), dtype=torch.float32, device=device)
                logits = head(features)
            else:
                feats = features[target_layer]
                logits = head(feats)
            loss = criterion(logits, patch_targets)

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

            upsampled = F.interpolate(
                logits,
                size=pixel_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = upsampled.argmax(dim=1).cpu()
            targets = pixel_targets.cpu()
            confmat = update_confusion_matrix(
                confmat,
                preds.view(-1),
                targets.view(-1),
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

            if max_examples and len(examples) < max_examples:
                for idx in range(min(max_examples - len(examples), images.size(0))):
                    examples.append(
                        {
                            "image": batch["image"][idx].cpu(),
                            "prediction": preds[idx],
                            "target": targets[idx],
                            "patch_logits": logits[idx].cpu(),
                            "meta": batch["meta"][idx],
                        }
                    )

            if logger and log_interval and (step % log_interval == 0 or step == total_steps):
                metrics_so_far = compute_iou(confmat)
                metrics_so_far = _mask_metrics(metrics_so_far)
                avg_loss_so_far = total_loss / max(total_images, 1)
                logger.info(
                    "%s step %05d/%05d | loss=%.4f | mIoU=%.3f",
                    split,
                    step,
                    total_steps,
                    avg_loss_so_far,
                    float(metrics_so_far["miou"].item()),
                )

    metrics = compute_iou(confmat)
    metrics = _mask_metrics(metrics)

    average_loss = total_loss / max(total_images, 1)
    results = {
        "loss": float(average_loss),
        "miou": float(metrics["miou"].item()),
        "per_class_iou": metrics["per_class"].tolist(),
    }
    return results, examples
