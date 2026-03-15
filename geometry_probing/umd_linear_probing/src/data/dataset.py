"""Data loading primitives for the UMD affordance dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import math

import numpy as np
import scipy.io as sio
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["UMDAffordanceDataset", "downsample_affordance_mask"]


def downsample_affordance_mask(
    mask: np.ndarray,
    patch_size: int,
    num_classes: int,
    ignore_index: int = 255,
    min_coverage: float = 0.55,
) -> np.ndarray:
    """Down-sample pixel-wise affordance masks onto the patch grid.

    Ignore pixels labeled ``ignore_index`` when voting. Patches with insufficient
    coverage receive ``ignore_index``.
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    height, width = mask.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Mask with shape {mask.shape} is not divisible by patch size {patch_size}"
        )

    h_tokens = height // patch_size
    w_tokens = width // patch_size
    patch_mask = np.full((h_tokens, w_tokens), ignore_index, dtype=np.int64)

    for h in range(h_tokens):
        for w in range(w_tokens):
            patch = mask[
                h * patch_size : (h + 1) * patch_size,
                w * patch_size : (w + 1) * patch_size,
            ]
            valid = patch != ignore_index
            if not np.any(valid):
                continue
            values, counts = np.unique(patch[valid], return_counts=True)
            best_idx = int(np.argmax(counts))
            coverage = counts[best_idx] / valid.sum()
            if coverage >= min_coverage:
                label = int(values[best_idx])
                if label < 0 or label >= num_classes:
                    raise ValueError(f"Label {label} outside [0, {num_classes})")
                patch_mask[h, w] = label
    return patch_mask


def _per_image_depth_normalize(depth: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Robust per-image normalization for depth to [-1, 1].

    Maps depth via percentiles: d' = clip((d - p_low)/(p_high - p_low), 0, 1), then to [-1,1].
    Handles degenerate ranges by returning zeros.
    """
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=np.float32)
    pl = np.percentile(d[finite], low)
    ph = np.percentile(d[finite], high)
    if ph <= pl + 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    x = (d - pl) / (ph - pl)
    x = np.clip(x, 0.0, 1.0)
    return (x * 2.0 - 1.0).astype(np.float32)


class UMDAffordanceDataset(Dataset):
    """PyTorch dataset returning images and down-sampled affordance masks."""

    def __init__(
        self,
        dataset_root: Path,
        split_records: List[Dict[str, str]],
        image_transform: Optional[Callable] = None,
        patch_size: int = 16,
        num_classes: int = 8,
        ignore_index: int = 255,
        min_patch_coverage: float = 0.55,
        exclude_background: bool = False,
        pad_to_patch_multiple: bool = False,
        geometry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split_records = split_records
        self.image_transform = image_transform
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.min_patch_coverage = min_patch_coverage
        self.exclude_background = exclude_background
        self.pad_to_patch_multiple = pad_to_patch_multiple
        # Geometry config
        self.geom_cfg = geometry if isinstance(geometry, dict) else None
        self.geom_index: Dict[str, Dict[str, str]] = {}
        if self.geom_cfg and 'manifest_path' in self.geom_cfg:
            import json
            manifest_path = Path(self.geom_cfg['manifest_path'])
            if manifest_path.is_file():
                with manifest_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                # Expecting {'train': [{frame_id, pred_depth_npy, pred_normal_npy, ...}, ...]} or flat list
                entries = []
                if isinstance(data, dict):
                    for k in ['train', 'val', 'test', 'data']:
                        if k in data and isinstance(data[k], list):
                            entries.extend(data[k])
                elif isinstance(data, list):
                    entries = data
                for item in entries:
                    fid = item.get('frame_id')
                    if isinstance(fid, str):
                        self.geom_index[fid] = {
                            'depth': item.get('pred_depth_npy'),
                            'normal': item.get('pred_normal_npy'),
                        }

    def _resolve_geom_asset(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return self.dataset_root / path

    def __len__(self) -> int:
        return len(self.split_records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.split_records[index]
        rgb_path = self.dataset_root / record["rgb"]
        label_path = self.dataset_root / record["label_mat"]

        with Image.open(rgb_path) as img:
            img = img.convert("RGB")
            image_array = np.array(img)

        mat = sio.loadmat(label_path)
        mask = np.asarray(mat["gt_label"], dtype=np.int64)
        if self.exclude_background:
            remapped = np.full_like(mask, fill_value=self.ignore_index)
            foreground = mask > 0
            remapped[foreground] = mask[foreground] - 1
        else:
            remapped = mask

        if self.pad_to_patch_multiple:
            image_array, remapped = _pad_to_patch_multiple(
                image_array,
                remapped,
                patch_size=self.patch_size,
                ignore_index=self.ignore_index,
            )

        if self.image_transform is not None:
            img_for_transform = Image.fromarray(image_array)
            image_tensor = self.image_transform(img_for_transform)
        else:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32) / 255.0

        patch_mask = downsample_affordance_mask(
            remapped,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            min_coverage=self.min_patch_coverage,
        )

        pixel_mask = torch.from_numpy(remapped.astype(np.int64))
        patch_mask_tensor = torch.from_numpy(patch_mask.astype(np.int64))

        sample: Dict[str, Any] = {
            "image": image_tensor,
            "pixel_mask": pixel_mask,
            "patch_mask": patch_mask_tensor,
            "meta": {
                "tool": record["tool"],
                "frame_id": record["frame_id"],
            },
        }
        # Attach geometry features if configured
        if self.geom_cfg:
            use_depth = bool(self.geom_cfg.get('use_depth', False))
            use_normal = bool(self.geom_cfg.get('use_normal', False))
            entry = self.geom_index.get(record["frame_id"], {})
            H, W = image_array.shape[:2]
            # Optional pooling parameters
            pool = self.geom_cfg.get('pool', {}) if isinstance(self.geom_cfg.get('pool', {}), dict) else {}
            k = int(pool.get('kernel', self.patch_size))
            s = int(pool.get('stride', self.patch_size))

            if use_depth and entry.get('depth'):
                try:
                    depth_path = self._resolve_geom_asset(entry['depth'])
                    if depth_path is None:
                        raise FileNotFoundError("Depth path is missing from geometry manifest.")
                    depth = np.load(depth_path).astype(np.float32)  # HxW
                    # resize to current image size
                    depth_rs = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
                    # normalize per-image percentiles
                    depth_norm = _per_image_depth_normalize(depth_rs)
                    depth_t = torch.from_numpy(depth_norm)[None, ...]  # 1xH xW
                    depth_grid = torch.nn.functional.avg_pool2d(depth_t[None, ...], kernel_size=k, stride=s).squeeze(0)
                    sample['geom_depth'] = depth_grid.to(torch.float32)
                except Exception:
                    pass

            if use_normal and entry.get('normal'):
                try:
                    normal_path = self._resolve_geom_asset(entry['normal'])
                    if normal_path is None:
                        raise FileNotFoundError("Normal path is missing from geometry manifest.")
                    normal = np.load(normal_path).astype(np.float32)  # HxWx3, in [-1,1]
                    # resize to current image size
                    normal_rs = cv2.resize(normal, (W, H), interpolation=cv2.INTER_LINEAR)
                    # renormalize per-pixel
                    eps = 1e-6
                    nrm = normal_rs.reshape(-1, 3)
                    denom = np.linalg.norm(nrm, axis=1, keepdims=True)
                    nrm = nrm / np.maximum(denom, eps)
                    normal_rs = nrm.reshape(H, W, 3)
                    normal_t = torch.from_numpy(normal_rs).permute(2, 0, 1)  # 3xH xW
                    normal_grid = torch.nn.functional.avg_pool2d(normal_t[None, ...], kernel_size=k, stride=s).squeeze(0)
                    # small renorm after pooling to keep within [-1,1]
                    g = normal_grid.permute(1, 2, 0)
                    g = g / torch.clamp(torch.linalg.norm(g, dim=-1, keepdim=True), min=1e-6)
                    sample['geom_normal'] = g.permute(2, 0, 1).to(torch.float32)
                except Exception:
                    pass
        return sample


def _pad_to_patch_multiple(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    patch_size: int,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    target_h = int(math.ceil(height / patch_size) * patch_size)
    target_w = int(math.ceil(width / patch_size) * patch_size)

    pad_h = target_h - height
    pad_w = target_w - width
    if pad_h == 0 and pad_w == 0:
        return image, mask

    image_pad = ((0, pad_h), (0, pad_w), (0, 0))
    mask_pad = ((0, pad_h), (0, pad_w))

    padded_image = np.pad(image, image_pad, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, mask_pad, mode="constant", constant_values=ignore_index)
    return padded_image, padded_mask
