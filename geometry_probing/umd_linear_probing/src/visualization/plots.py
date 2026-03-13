"""Visualization helpers for training diagnostics and qualitative results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

DINOV3_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINOV3_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_PALETTE = np.array(
    [
        [68, 68, 68],  # background
        [255, 99, 71],  # grasp
        [65, 105, 225],  # cut
        [34, 139, 34],  # scoop
        [255, 215, 0],  # contain
        [218, 112, 214],  # pound
        [255, 165, 0],  # support
        [0, 206, 209],  # wrap-grasp
    ],
    dtype=np.float32,
) / 255.0

SPLIT_COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}

__all__ = ["plot_training_curves", "plot_final_metrics", "plot_step_curves", "save_prediction_gallery"]


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = array.transpose(1, 2, 0)
    array = (array * DINOV3_STD) + DINOV3_MEAN
    array = np.clip(array, 0.0, 1.0)
    return array


def _mask_to_color(mask: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    mask_np = np.clip(mask_np, 0, palette.shape[0] - 1)
    return palette[mask_np]


def plot_training_curves(history: List[dict], output_dir: Path, *, test_metrics: dict | None = None) -> None:
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_miou = [record["train_miou"] for record in history]
    val_miou = [record["val_miou"] for record in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    if test_metrics is not None:
        test_epoch = epochs[-1] + 0.5 if epochs else 1
        axes[0].scatter([test_epoch], [test_metrics["loss"]], label="test", marker="x", color="black")
        axes[0].axhline(test_metrics["loss"], linestyle="--", color="black", alpha=0.3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_title("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_miou, label="train")
    axes[1].plot(epochs, val_miou, label="val")
    if test_metrics is not None:
        test_epoch = epochs[-1] + 0.5 if epochs else 1
        axes[1].scatter([test_epoch], [test_metrics["miou"]], label="test", marker="x", color="black")
        axes[1].axhline(test_metrics["miou"], linestyle="--", color="black", alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("mIoU")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_final_metrics(
    train_metrics: dict | None,
    val_metrics: dict | None,
    test_metrics: dict | None,
    output_dir: Path,
) -> None:
    splits: List[str] = []
    losses: List[float] = []
    mious: List[float] = []

    def add_split(name: str, metrics: dict | None) -> None:
        if metrics is None:
            return
        if "loss" not in metrics or "miou" not in metrics:
            return
        splits.append(name)
        losses.append(float(metrics["loss"]))
        mious.append(float(metrics["miou"]))

    add_split("train", train_metrics)
    add_split("val", val_metrics)
    add_split("test", test_metrics)

    if not splits:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(splits))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, losses, color="#6baed6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_ylabel("loss")
    axes[0].set_title("Final loss by split")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, mious, color="#74c476")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("mIoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Final mIoU by split")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    for axis, values in zip(axes, [losses, mious]):
        for xpos, val in zip(x, values):
            axis.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "final_metrics.png", dpi=150)
    plt.close(fig)


def plot_step_curves(
    train_data: dict | None,
    val_data: dict | None,
    test_data: dict | None,
    output_dir: Path,
) -> None:
    series: List[tuple[str, dict]] = []
    for name, data in (("train", train_data), ("val", val_data), ("test", test_data)):
        if not data:
            continue
        steps = data.get("steps", [])
        losses = data.get("loss", [])
        mious = data.get("miou", [])
        if steps and losses and mious:
            series.append((name, data))

    if not series:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(series)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)

    for row, (split, data) in enumerate(series):
        color = SPLIT_COLORS.get(split, "#444444")
        steps = data["steps"]
        losses = data["loss"]
        mious = data["miou"]

        ax_loss = axes[row, 0]
        ax_loss.plot(steps, losses, color=color, linewidth=1.5)
        ax_loss.set_title(f"{split} loss")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, linestyle="--", alpha=0.3)

        ax_miou = axes[row, 1]
        ax_miou.plot(steps, mious, color=color, linewidth=1.5)
        ax_miou.set_title(f"{split} mIoU")
        ax_miou.set_xlabel("step")
        ax_miou.set_ylabel("mIoU")
        ax_miou.set_ylim(0, 1)
        ax_miou.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "step_curves.png", dpi=150)
    plt.close(fig)


def save_prediction_gallery(
    examples: Sequence[dict],
    output_dir: Path,
    class_names: Sequence[str],
    alpha: float = 0.5,
    palette: np.ndarray = DEFAULT_PALETTE,
) -> None:
    if not examples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    num_examples = len(examples)
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, example in enumerate(examples):
        image = _to_numpy_image(example["image"])
        pred = _mask_to_color(example["prediction"], palette)
        target = _mask_to_color(example["target"], palette)

        overlay_pred = (1 - alpha) * image + alpha * pred
        overlay_target = (1 - alpha) * image + alpha * target

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"RGB | {example['meta']['tool']}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_pred)
        axes[row, 1].set_title("Prediction")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_target)
        axes[row, 2].set_title("Ground truth")
        axes[row, 2].axis("off")

    legend_fig = plt.figure(figsize=(4, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    handles = []
    for idx, name in enumerate(class_names):
        color = palette[idx]
        handles.append(
            plt.Line2D([0], [0], marker="s", color=color, label=name, linestyle="", markersize=10)
        )
    legend_ax.legend(handles=handles, title="Affordance classes", loc="center", ncol=2)

    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_examples.png", dpi=150)
    legend_fig.savefig(output_dir / "palette_legend.png", dpi=150)
    plt.close(fig)
    plt.close(legend_fig)
