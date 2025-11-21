"""Training loop for dense linear probing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.data.collate import collate_with_meta
from src.data.dataset import UMDAffordanceDataset
from src.data.transforms import get_default_image_transform, get_sam_image_transform
from src.engine.eval import evaluate_linear_probe
from src.models import (
    DINOBackbone,
    DINOv3Backbone,
    DINOv2Backbone,
    FluxBackbone,
    SigLIP2Backbone,
    SAMBackbone,
    StableDiffusionBackbone,
)
from src.models.head import LinearProbeHead
from src.models.linear_head import MultiLayerLinearHead
from src.utils.config import Config
from src.utils.logging import create_logger
from src.utils.metrics import compute_iou, update_confusion_matrix
from src.utils.random import set_seed


def _state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _serialize_config(value):
    if isinstance(value, dict):
        return {key: _serialize_config(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

__all__ = ["LinearProbeExperiment"]


@dataclass
class TrainingRecord:
    epoch: int
    lr: float
    weight_decay: float
    train_loss: float
    train_miou: float
    val_loss: float
    val_miou: float
    val_per_class: List[float]


class LinearProbeExperiment:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.training_cfg = config["training"]
        self.dataset_cfg = config["dataset"]
        self.model_cfg = config["model"]
        self.model_target = self.model_cfg["target"]
        self.model_params = dict(self.model_cfg.get("params", {}))
        self.output_dir_name = self.model_params.pop("output_dir_name", None)
        raw_head_cfg = self.model_cfg.get("head", {})
        self.head_cfg = dict(raw_head_cfg) if isinstance(raw_head_cfg, dict) else {}
        if not self.head_cfg and "head" in self.model_params:
            head_from_params = self.model_params.pop("head")
            if isinstance(head_from_params, dict):
                self.head_cfg = dict(head_from_params)
        head_type = str(self.head_cfg.get("type", "linear")).lower()
        self.use_multi_head = head_type in {"multi", "multi_linear", "multi_layer_linear"}
        self.head_feature_keys: List[object] | None = None
        self.head_feature_channels: Dict[object, int] = {}
        self.primary_feature_key: object | None = None

        output_root = self.training_cfg.get("output_root")
        base_name = None
        if isinstance(self.output_dir_name, str):
            stripped = self.output_dir_name.strip()
            base_name = stripped if stripped else None

        if output_root is not None:
            base_dir = Path(output_root) / (base_name or self.model_target)
        else:
            output_dir_setting = self.training_cfg.get("output_dir")
            if output_dir_setting is None:
                raise KeyError(
                    "Training configuration must provide either 'output_root' or 'output_dir'."
                )
            base_dir = Path(output_dir_setting)
            if base_name:
                if base_dir.name:
                    base_dir = base_dir.parent / base_name
                else:
                    base_dir = base_dir / base_name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_timestamp = timestamp
        if base_dir.name:
            self.output_dir = base_dir.parent / f"{base_dir.name}_{timestamp}"
        else:
            self.output_dir = base_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_logger = create_logger(
            self.output_dir, name="linear_probe_master", filename="master.log"
        )
        self.logger = self.base_logger
        self.log_interval = self.training_cfg.get("log_interval", 100)
        self.val_log_interval = self.training_cfg.get("val_log_interval", self.log_interval)

        self.device = torch.device(self.training_cfg.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available; falling back to CPU")
            self.device = torch.device("cpu")

        set_seed(self.training_cfg.get("seed", 1337))
        self.logger.info("Loading dataset splits from %s", self.dataset_cfg["split_path"])
        with Path(self.dataset_cfg["split_path"]).open("r", encoding="utf-8") as handle:
            self.split_mapping = json.load(handle)

        self.ignore_index = self.dataset_cfg.get("ignore_index", 255)
        self.num_classes = self.dataset_cfg.get("num_classes", 8)
        self.patch_size = self.dataset_cfg.get("patch_size", 16)
        self.min_patch_coverage = self.dataset_cfg.get("min_patch_coverage", 0.55)
        self.metric_ignore_indices = set(self.dataset_cfg.get("metric_ignore_indices", []))

        if self.model_target == "dinov3":
            self.logger.info("Instantiating DINOv3 backbone %s", self.model_params["name"])
            self.transform = get_default_image_transform()
            self.backbone = DINOv3Backbone(
                model_name=self.model_params["name"],
                repo_path=Path(self.model_params["repo_path"]),
                checkpoint_path=Path(self.model_params["checkpoint_path"]),
                layers_to_hook=self.model_params.get("layers_to_hook", [-1]),
                device=str(self.device),
            )
            self.target_layer = self.model_params.get("primary_layer")
            if self.target_layer is None:
                # default to last requested layer
                self.target_layer = self.backbone.layer_order[-1]
        elif self.model_target == "dinov2":
            self.logger.info("Instantiating DINOv2 backbone %s", self.model_params["name"])
            self.transform = get_default_image_transform()
            self.backbone = DINOv2Backbone(
                model_name=self.model_params["name"],
                repo_path=Path(self.model_params["repo_path"]),
                checkpoint_path=Path(self.model_params["checkpoint_path"]),
                layers_to_hook=self.model_params.get("layers_to_hook", [-1]),
                device=str(self.device),
            )
            self.target_layer = self.model_params.get("primary_layer")
            if self.target_layer is None:
                self.target_layer = self.backbone.layer_order[-1]
            self.patch_size = self.model_params.get("patch_size", self.patch_size)
        elif self.model_target == "dino":
            self.logger.info("Instantiating DINO backbone %s", self.model_params["name"])
            self.transform = get_default_image_transform()
            hub_kwargs = self.model_params.get("hub_kwargs")
            if hub_kwargs is not None and not isinstance(hub_kwargs, dict):
                raise TypeError("hub_kwargs must be a mapping when provided.")
            self.backbone = DINOBackbone(
                model_name=self.model_params["name"],
                repo_path=Path(self.model_params["repo_path"]),
                checkpoint_path=Path(self.model_params["checkpoint_path"]),
                layers_to_hook=self.model_params.get("layers_to_hook", [-1]),
                device=str(self.device),
                checkpoint_key=self.model_params.get("checkpoint_key"),
                hub_kwargs=dict(hub_kwargs) if isinstance(hub_kwargs, dict) else None,
            )
            self.target_layer = self.model_params.get("primary_layer")
            if self.target_layer is None:
                self.target_layer = self.backbone.layer_order[-1]
            self.patch_size = self.model_params.get("patch_size", self.patch_size)
        elif self.model_target == "open_clip":
            from src.models.openclip import OpenCLIPBackbone

            model_id = self.model_params.get("model_id")
            if model_id is None:
                raise ValueError("OpenCLIP configuration must provide 'model_id'.")
            self.logger.info("Instantiating OpenCLIP backbone %s", model_id)
            backbone_precision = self.model_params.get(
                "precision", self.training_cfg.get("precision", "fp32")
            )
            self.backbone = OpenCLIPBackbone(
                model_id=model_id,
                device=str(self.device),
                precision=backbone_precision,
                repo_path=self.model_params.get("repo_path"),
                layers_to_hook=self.model_params.get("layers_to_hook"),
            )
            self.transform = self.backbone.transform
            self.target_layer = self.model_params.get("primary_layer", self.backbone.default_layer)
            self.patch_size = self.model_params.get("patch_size", self.patch_size)
        elif self.model_target == "flux":
            self.logger.info("Instantiating Flux backbone from %s", self.model_params.get("model_dir"))
            self.transform = get_default_image_transform()
            flux_params = dict(self.model_params)
            flux_params.setdefault("device", str(self.device))
            self.backbone = FluxBackbone(**flux_params)
            self.target_layer = self.model_params.get("primary_layer", self.backbone.primary_layer)
            self.patch_size = self.model_params.get("patch_size", 16)
        elif self.model_target == "stable_diffusion":
            self.logger.info("Instantiating Stable Diffusion backbone %s", self.model_params.get("model_id"))
            self.transform = get_default_image_transform()
            sd_params = dict(self.model_params)
            sd_params.setdefault("device", str(self.device))
            patch_size = sd_params.pop("patch_size", 16)
            # SD is an optional dependency; give a clear error if it's unavailable in this env
            if 'StableDiffusionBackbone' not in globals() or StableDiffusionBackbone is None:
                raise RuntimeError(
                    "StableDiffusionBackbone is unavailable in this environment. "
                    "Please install a compatible diffusers + torch stack (e.g., torch>=2.1 and diffusers>=0.27), "
                    "or downgrade diffusers to a version that works with your torch version (e.g., diffusers==0.27.* for torch 2.0.x)."
                )
            self.backbone = StableDiffusionBackbone(**sd_params)
            target_layer_cfg = self.model_params.get("primary_layer")
            if target_layer_cfg is None:
                self.target_layer = self.backbone.primary_layer
            else:
                self.target_layer = self.backbone._resolve_single(target_layer_cfg)  # type: ignore[attr-defined]
            self.patch_size = patch_size
        elif self.model_target == "siglip2":
            self.logger.info("Instantiating SigLIP2 backbone %s", self.model_params.get("model_id"))
            self.transform = get_default_image_transform()
            siglip_params = dict(self.model_params)
            siglip_params.setdefault("device", str(self.device))
            patch_size = siglip_params.pop("patch_size", 16)
            self.backbone = SigLIP2Backbone(**siglip_params)
            self.target_layer = self.backbone.default_layer
            self.patch_size = patch_size
        elif self.model_target == "sam":
            self.logger.info("Instantiating SAM backbone %s", self.model_params.get("arch", "vit_h"))
            # Use SAM-specific transform (ToTensor only). SAMBackbone applies
            # its own mean/std normalization internally; using the default
            # ImageNet normalization here would cause double-normalization.
            self.transform = get_sam_image_transform()
            sam_params = dict(self.model_params)
            sam_params.setdefault("device", str(self.device))
            patch_size = sam_params.pop("patch_size", 16)
            self.backbone = SAMBackbone(**sam_params)
            self.target_layer = self.backbone.layer_key
            self.patch_size = patch_size
        else:
            raise ValueError(f"Unsupported model target: {self.model_target}")

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        self.logger.info("Training linear probe on feature key %s", self.target_layer)

        self.embed_dim, self.patch_hw = self._infer_feature_shape()
        self.logger.info(
            "Feature map for layer %s has embed_dim=%d, grid=%s",
            self.target_layer,
            self.embed_dim,
            self.patch_hw,
        )

    def _build_dataset(self, split: str) -> UMDAffordanceDataset:
        return UMDAffordanceDataset(
            dataset_root=Path(self.dataset_cfg["root"]),
            split_records=self.split_mapping[split],
            image_transform=self.transform,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            min_patch_coverage=self.min_patch_coverage,
            exclude_background=self.dataset_cfg.get("exclude_background", False),
            pad_to_patch_multiple=self.dataset_cfg.get("pad_to_patch_multiple", False),
            geometry=self.dataset_cfg.get("geometry"),
        )

    def _build_dataloaders(self):
        batch_size = self.training_cfg["batch_size"]
        num_workers = self.training_cfg.get("num_workers", 4)

        train_dataset = self._build_dataset("train")
        val_dataset = self._build_dataset("val")
        test_dataset = self._build_dataset("test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_with_meta,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_with_meta,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_with_meta,
        )
        return train_loader, val_loader, test_loader

    def _apply_metric_mask(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.metric_ignore_indices:
            return metrics

        per_class = metrics["per_class"].clone()
        for idx in self.metric_ignore_indices:
            if 0 <= idx < per_class.numel():
                per_class[idx] = torch.nan

        metrics["per_class"] = per_class
        metrics["miou"] = torch.nanmean(per_class)
        return metrics

    def _filter_metrics_for_summary(self, metrics: Dict[str, object] | None) -> Dict[str, object]:
        if not metrics:
            return {}
        filtered = dict(metrics)
        per_class = filtered.get("per_class_iou")
        if isinstance(per_class, list) and self.metric_ignore_indices:
            filtered["per_class_iou"] = [
                value
                for idx, value in enumerate(per_class)
                if idx not in self.metric_ignore_indices
            ]
        return filtered

    def _infer_feature_shape(self):
        batch = next(iter(self.train_loader))
        images = batch["image"]
        if images.dim() != 4 or images.shape[1:] != (3, 480, 640):
            height, width = images.shape[-2:]
            if not self.dataset_cfg.get("pad_to_patch_multiple", False):
                raise ValueError(
                    f"Expected input images with shape (B, 3, 480, 640); received {tuple(images.shape)}"
                )
            if images.shape[1] != 3 or height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    "Padded images must have 3 channels and spatial dims divisible by patch size "
                    f"{self.patch_size}; received {tuple(images.shape)}"
                )

        precision = self.training_cfg.get("precision", "bf16")
        with torch.no_grad():
            feats = self.backbone(
                images[:1].to(self.device, non_blocking=True), autocast_precision=precision
            )

        if not isinstance(feats, dict):
            raise TypeError("Backbone must return a mapping from feature keys to tensors.")

        available_keys = list(feats.keys())
        # Add optional geometry keys from batch if present
        for gk in ("geom_depth", "geom_normal"):
            if gk in batch and isinstance(batch[gk], torch.Tensor):
                if gk not in available_keys:
                    available_keys.append(gk)

        def resolve_key(key: object) -> object:
            if key in feats:
                return key
            if isinstance(key, str) and key in {"geom_depth", "geom_normal"} and key in available_keys:
                return key
            if isinstance(key, str):
                try:
                    numeric = int(key)
                except ValueError:
                    pass
                else:
                    if numeric in feats:
                        return numeric
            if isinstance(key, int):
                candidate = str(key)
                if candidate in feats:
                    return candidate
            raise KeyError(f"Feature key {key!r} is not available from the backbone/geometry. Available keys: {available_keys}")

        if self.use_multi_head:
            requested_keys = self.head_cfg.get("feature_keys")
            if requested_keys is None:
                resolved_keys = list(available_keys)
            else:
                resolved_keys = [resolve_key(key) for key in requested_keys]
            # Auto-include geometry features when enabled in dataset config
            geom_cfg = self.dataset_cfg.get("geometry") or {}
            if isinstance(geom_cfg, dict):
                if bool(geom_cfg.get("use_depth", False)) and "geom_depth" in available_keys and "geom_depth" not in resolved_keys:
                    resolved_keys.append("geom_depth")
                if bool(geom_cfg.get("use_normal", False)) and "geom_normal" in available_keys and "geom_normal" not in resolved_keys:
                    resolved_keys.append("geom_normal")
            if not resolved_keys:
                raise ValueError("feature_keys must resolve to at least one feature map.")
            primary_candidate = self.head_cfg.get("primary_key", resolved_keys[-1])
            primary_key = resolve_key(primary_candidate)
        else:
            resolved_keys = [resolve_key(self.target_layer)]
            primary_key = resolved_keys[-1]

        def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 4:
                return tensor
            if tensor.dim() == 2:
                return tensor[:, :, None, None]
            raise ValueError(f"Backbone output must be 2D or 4D; received shape {tuple(tensor.shape)}")

        selected_keys = set(resolved_keys)
        selected_keys.add(primary_key)
        feature_channels: Dict[object, int] = {}
        normalized_feats: Dict[object, torch.Tensor] = {}
        # Prepare normalized feature map shapes for both backbone & geometry entries
        for key in selected_keys:
            if key in feats:
                tensor = ensure_4d(feats[key])
            else:
                # geometry from batch
                geom = batch.get(str(key)) if isinstance(key, str) else None
                if geom is None:
                    raise KeyError(f"Feature {key!r} requested but not found in batch geometry.")
                tensor = ensure_4d(geom[:1])  # use first sample to infer shape
            feature_channels[key] = tensor.shape[1]
            normalized_feats[key] = tensor

        primary_tensor = normalized_feats[primary_key]
        grid = primary_tensor.shape[2:]
        patch_mask = batch["patch_mask"][:1]
        if patch_mask.shape[-2:] != grid:
            raise ValueError(
                f"Patch mask grid {tuple(patch_mask.shape[-2:])} does not match "
                f"backbone output grid {grid}"
            )

        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        if trainable_params != 0:
            raise ValueError(
                f"Backbone has {trainable_params} trainable parameters; expected 0."
            )

        self.head_feature_keys = resolved_keys
        self.primary_feature_key = primary_key
        self.head_feature_channels = feature_channels

        if self.use_multi_head:
            fuse_mode = str(self.head_cfg.get("fuse_mode", "concat")).lower()
            if fuse_mode == "concat":
                embed_dim = sum(feature_channels[key] for key in resolved_keys)
            else:
                reference = feature_channels[resolved_keys[0]]
                for key in resolved_keys[1:]:
                    if feature_channels[key] != reference:
                        raise ValueError(
                            f"fuse_mode '{fuse_mode}' requires equal channel dimensions; "
                            f"got {feature_channels[key]} for feature {key!r}, expected {reference}."
                        )
                embed_dim = reference
        else:
            embed_dim = feature_channels[resolved_keys[0]]

        return embed_dim, grid

    def _apply_head(self, head: nn.Module, features: Dict[object, torch.Tensor]) -> torch.Tensor:
        if self.use_multi_head:
            return head(features)
        return head(features[self.target_layer])

    def _merge_geometry_features(self, features: Dict[object, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[object, torch.Tensor]:
        """Attach optional geometry features present in the batch to the feature dict."""
        merged = dict(features)
        # Preferred: take from batch if provided
        for gk in ("geom_depth", "geom_normal"):
            if gk in batch and isinstance(batch[gk], torch.Tensor):
                merged[gk] = batch[gk].to(self.device, non_blocking=True)
        # If head expects geometry but current split lacks it, provide zero placeholders
        if self.use_multi_head and self.head_feature_keys is not None:
            # derive grid from primary feature
            primary = self.primary_feature_key if self.primary_feature_key is not None else self.target_layer
            ref = merged.get(primary, next(iter(merged.values())))
            B = ref.shape[0] if ref.dim() == 4 else 1
            grid = ref.shape[-2:]
            for gk, ch in (("geom_depth", 1), ("geom_normal", 3)):
                if gk in self.head_feature_keys and gk not in merged:
                    merged[gk] = torch.zeros((B, ch, *grid), dtype=torch.float32, device=self.device)
        return merged

    def _build_head(self) -> nn.Module:
        if self.use_multi_head:
            if self.head_feature_keys is None or self.primary_feature_key is None:
                raise RuntimeError("Head feature metadata has not been initialised.")
            fuse_mode = str(self.head_cfg.get("fuse_mode", "concat")).lower()
            dropout = float(self.head_cfg.get("dropout", 0.0))
            use_batchnorm = bool(self.head_cfg.get("use_batchnorm", True))
            affine_bn = bool(self.head_cfg.get("affine_bn", True))
            align_corners = bool(self.head_cfg.get("align_corners", False))
            head = MultiLayerLinearHead(
                feature_keys=self.head_feature_keys,
                in_channels=self.head_feature_channels,
                num_classes=self.num_classes,
                primary_key=self.primary_feature_key,
                fuse_mode=fuse_mode,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                affine_bn=affine_bn,
                align_corners=align_corners,
            )
        else:
            affine_bn = bool(self.head_cfg.get("affine_bn", True)) if "affine_bn" in self.head_cfg else True
            head = LinearProbeHead(self.embed_dim, self.num_classes, affine_bn=affine_bn)
        return head
    def _dump_run_config(self, run_dir: Path, lr: float, weight_decay: float) -> None:
        snapshot = _serialize_config(self.cfg.data)
        run_info = {
            "timestamp": self.run_timestamp,
            "output_run_dir": str(run_dir),
            "hyperparams": {"lr": lr, "weight_decay": weight_decay},
        }
        if isinstance(snapshot, dict):
            snapshot = dict(snapshot)
            existing = snapshot.get("run_info", {})
            if isinstance(existing, dict):
                run_info = {**existing, **run_info}
            snapshot["run_info"] = run_info
        config_path = run_dir / "config_snapshot.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, sort_keys=False)

    def train(self) -> Dict[str, float]:
        lr_grid = self.training_cfg["lr_grid"]
        wd_grid = self.training_cfg["weight_decay_grid"]
        max_epochs = self.training_cfg["max_epochs"]
        patience = self.training_cfg.get("patience", 8)
        eval_interval = self.training_cfg.get("eval_interval", 1)
        grad_clip = self.training_cfg.get("grad_clip_norm", 0.0)
        precision = self.training_cfg.get("precision", "bf16")
        accumulate = max(1, self.training_cfg.get("accumulate_grad_batches", 1))

        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        history: List[TrainingRecord] = []
        best_overall = {
            "miou": -1.0,
            "state_dict": None,
            "hyperparams": None,
            "val_metrics": None,
            "train_history": [],
            "examples": [],
            "model_target": self.model_target,
        }

        for lr, weight_decay in product(lr_grid, wd_grid):
            run_dir = self.output_dir / f"lr{lr:.0e}_wd{weight_decay:.0e}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_logger = create_logger(
                run_dir,
                name=f"linear_probe.lr{lr:.0e}_wd{weight_decay:.0e}",
                filename="train.log",
            )
            self._dump_run_config(run_dir, lr, weight_decay)
            previous_logger = self.logger
            self.logger = run_logger

            self.base_logger.info(
                "Starting sweep lr=%g, weight_decay=%g (logs: %s)", lr, weight_decay, run_dir
            )
            self.logger.info("Starting sweep lr=%g, weight_decay=%g", lr, weight_decay)

            head = self._build_head().to(self.device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

            best_val = -1.0
            best_state = None
            wait = 0

            for epoch in range(1, max_epochs + 1):
                train_metrics = self._train_one_epoch(
                    head=head,
                    optimizer=optimizer,
                    criterion=criterion,
                    precision=precision,
                    grad_clip=grad_clip,
                    accumulate=accumulate,
                )

                if epoch % eval_interval == 0:
                    val_metrics, _ = evaluate_linear_probe(
                        self.backbone,
                        head,
                        self.val_loader,
                        self.device,
                        precision,
                        self.num_classes,
                        self.ignore_index,
                        criterion,
                        target_layer=self.target_layer,
                        max_examples=0,
                        logger=self.logger,
                        log_interval=self.val_log_interval,
                        ignore_indices=self.metric_ignore_indices,
                        split="val",
                        use_multi_head=self.use_multi_head,
                    )
                    history.append(
                        TrainingRecord(
                            epoch=epoch,
                            lr=lr,
                            weight_decay=weight_decay,
                            train_loss=train_metrics["loss"],
                            train_miou=train_metrics["miou"],
                            val_loss=val_metrics["loss"],
                            val_miou=val_metrics["miou"],
                            val_per_class=list(val_metrics["per_class_iou"]),
                        )
                    )

                    self.logger.info(
                        "Epoch %03d | lr=%g wd=%g | train loss=%.4f mIoU=%.3f | val loss=%.4f mIoU=%.3f",
                        epoch,
                        lr,
                        weight_decay,
                        train_metrics["loss"],
                        train_metrics["miou"],
                        val_metrics["loss"],
                        val_metrics["miou"],
                    )
                    self.logger.info(
                        "Val per-class IoU: %s",
                        [round(x, 4) for x in val_metrics["per_class_iou"]],
                    )

                    if val_metrics["miou"] > best_val:
                        best_val = val_metrics["miou"]
                        best_state = _state_dict_to_cpu(head)
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            self.logger.info("Early stopping triggered (patience=%d)", patience)
                            break

            if best_state is None:
                self.logger.warning("No validation improvement for lr=%g wd=%g", lr, weight_decay)
                self.logger = previous_logger
                continue

            ckpt_path = run_dir / "linear_probe.pth"
            torch.save({"state_dict": best_state, "hyperparams": {"lr": lr, "weight_decay": weight_decay}}, ckpt_path)

            # Restore best state for evaluation
            head.load_state_dict(best_state)
            val_metrics, examples = evaluate_linear_probe(
                self.backbone,
                head,
                self.val_loader,
                self.device,
                precision,
                self.num_classes,
                self.ignore_index,
                criterion,
                target_layer=self.target_layer,
                max_examples=self.cfg.get("visualization", {}).get("num_samples", 0),
                logger=self.logger,
                log_interval=self.val_log_interval,
                ignore_indices=self.metric_ignore_indices,
                split="val",
                use_multi_head=self.use_multi_head,
            )
            self.logger.info("Val per-class IoU: %s", [round(x, 4) for x in val_metrics["per_class_iou"]])

            if val_metrics["miou"] > best_overall["miou"]:
                best_overall.update(
                    {
                        "miou": val_metrics["miou"],
                        "state_dict": dict(best_state),
                        "hyperparams": {"lr": lr, "weight_decay": weight_decay},
                        "val_metrics": val_metrics,
                        "train_history": [record.__dict__ for record in history],
                        "examples": examples,
                    }
                )
                best_overall["checkpoint_path"] = ckpt_path

            self.logger = previous_logger

        if best_overall["state_dict"] is None:
            raise RuntimeError("No successful training run produced a model")

        # Evaluate best model on the test split
        head = self._build_head().to(self.device)
        head.load_state_dict(best_overall["state_dict"])

        test_metrics, test_examples = evaluate_linear_probe(
            self.backbone,
            head,
            self.test_loader,
            self.device,
            precision=self.training_cfg.get("precision", "bf16"),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            criterion=nn.CrossEntropyLoss(ignore_index=self.ignore_index),
            target_layer=self.target_layer,
            max_examples=self.cfg.get("visualization", {}).get("num_samples", 0),
            logger=self.base_logger,
            log_interval=self.val_log_interval,
            ignore_indices=self.metric_ignore_indices,
            split="test",
            use_multi_head=self.use_multi_head,
        )
        self.base_logger.info("Test per-class IoU: %s", [round(x, 4) for x in test_metrics["per_class_iou"]])

        summary = {
            "best_hyperparams": best_overall["hyperparams"],
            "best_val": self._filter_metrics_for_summary(best_overall["val_metrics"]),
            "test_metrics": self._filter_metrics_for_summary(test_metrics),
            "checkpoint_path": str(best_overall["checkpoint_path"]),
            "model_target": self.model_target,
            "feature_grid": list(self.patch_hw),
            "run_timestamp": self.run_timestamp,
        }

        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        self.base_logger.info("Best configuration: %s", summary)

        # Store examples for visualization modules to use later
        examples_path = self.output_dir / "val_examples.pt"
        torch.save(best_overall["examples"], examples_path)
        torch.save(test_examples, self.output_dir / "test_examples.pt")

        history_path = self.output_dir / "training_history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump([record.__dict__ for record in history], handle, indent=2)

        return test_metrics

    def _train_one_epoch(
        self,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        precision: str,
        grad_clip: float,
        accumulate: int,
    ) -> Dict[str, float]:
        head.train()
        self.backbone.eval()

        confmat = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        total_loss = 0.0
        total_samples = 0

        optimizer.zero_grad(set_to_none=True)

        total_steps = len(self.train_loader)
        for step, batch in enumerate(self.train_loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            patch_targets = batch["patch_mask"].to(self.device, non_blocking=True)
            pixel_targets = batch["pixel_mask"].to(self.device, non_blocking=True)

            features = self.backbone(images, autocast_precision=precision)
            features = self._merge_geometry_features(features, batch)
            logits = self._apply_head(head, features)
            loss = criterion(logits, patch_targets)

            (loss / accumulate).backward()

            if step % accumulate == 0 or step == len(self.train_loader):
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                upsampled = F.interpolate(
                    logits,
                    size=pixel_targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                preds = upsampled.argmax(dim=1).cpu()
                confmat = update_confusion_matrix(
                    confmat,
                    preds.view(-1),
                    pixel_targets.cpu().view(-1),
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                )

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            if self.log_interval and (step % self.log_interval == 0 or step == len(self.train_loader)):
                avg_so_far = total_loss / max(total_samples, 1)
                metrics_so_far = compute_iou(confmat)
                metrics_so_far = self._apply_metric_mask(metrics_so_far)
                miou_so_far = float(metrics_so_far["miou"].item())
                self.logger.info(
                    "train step %05d/%05d | loss=%.4f | mIoU=%.3f",
                    step,
                    len(self.train_loader),
                    avg_so_far,
                    miou_so_far,
                )

        metrics = compute_iou(confmat)
        metrics = self._apply_metric_mask(metrics)
        avg_loss = total_loss / max(total_samples, 1)
        return {"loss": avg_loss, "miou": float(metrics["miou"].item())}
