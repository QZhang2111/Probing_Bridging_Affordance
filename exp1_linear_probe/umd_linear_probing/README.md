# UMD Affordance Linear Probing

This workspace implements dense linear probing on the UMD part-affordance dataset with **frozen transformer backbones**. Supported vision encoders:

- **DINOv3 ViT-7B/16** (default)
- **OpenCLIP ViT-B/16 LAION**
- **DINOv2 ViT-g/14**

All variants share the same protocol: freeze the backbone, train a BatchNorm + 1×1 Conv head on patch tokens, bilinearly upsample predictions, and report pixel-level mIoU (ignoring background if configured).

## Layout

```
.
├── configs/                 # Default + local overrides for paths and hyper-parameters
├── metadata/splits/         # Train/val/test mapping generated from category_split.txt
├── scripts/                 # CLI entry-points (prepare_splits, train, eval, visualize)
├── src/
│   ├── data/                # Dataset, padding, transforms, collate utilities
│   ├── engine/              # Training loop and evaluation helpers
│   ├── models/              # Backbone wrappers + linear probe head
│   ├── utils/               # Config loader, metrics, logging, seeding helpers
│   └── visualization/       # Training curve + qualitative plot helpers
└── requirements.txt         # Python dependencies
```

## Quick start

1. **Create the split mapping** (once per seed/ratio):
   ```bash
   ./scripts/prepare_splits.py \
     --dataset-root ../../dataset/UMD/part-affordance-dataset \
     --category-split ../../dataset/UMD/part-affordance-dataset/category_split.txt \
     --val-ratio 0.2 \
     --val-seed 42 \
     --ensure-val-all-classes \
     --num-classes 8 \
     --exclude-background \
     --output metadata/splits/category_split_seed42_v20.json
   ```
   `configs/default.yaml` references this file by default; adjust if you generate alternatives.

2. **Review the config** (`configs/default.yaml`). Key fields:
   - `dataset.*`: dataset root/split and affordance settings. `pad_to_patch_multiple` keeps inputs divisible by the backbone patch size (14 for DINOv2, 16 otherwise). Background pixels are remapped to `ignore_index`, so the model trains on seven foreground affordances only.
   - `model.*`: `target` selects the backbone family; `config_path` points to a backbone-specific YAML (`configs/models/dinov3.yaml`, `configs/models/openclip_vitb16.yaml`, `configs/models/dinov2.yaml`, …).
   - `training.*`: optimisation grid, logging cadence (per 100 steps by default), output directory, etc.

   Machine-specific overrides (dataset root, checkpoint path, batch size…) live in `configs/local*.yaml` (e.g., `local.yaml`, `local_openclip.yaml`, `local_dinov2.yaml`). They are merged after the defaults and the model-specific YAML.

3. **Install dependencies** (consider using a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch training** (choose the local override for the backbone you want):
   ```bash
   # DINOv3 ViT-7B/16
   ./scripts/train.py --defaults configs/default.yaml --local configs/local.yaml

   # OpenCLIP ViT-B/16
   ./scripts/train.py --defaults configs/default.yaml --local configs/local_openclip.yaml

   # DINOv2 ViT-g/14 (pads inputs to multiples of 14, sets num_workers=0)
   ./scripts/train.py --defaults configs/default.yaml --local configs/local_dinov2.yaml
   ```

   The trainer performs a grid-search over the configured learning rates and weight decays with patience-based early stopping. Each combination gets its own run folder (e.g. `outputs/dinov3/lr1e-03_wd1e-02/`) with a dedicated `train.log` that records train/val progress every 100 steps. Top-level artifacts include:
   - `summary.json`: best hyper-parameters + metrics (background indices listed in `metric_ignore_indices` are removed when serialised).
   - `linear_probe.pth`: checkpoint for each sweep setting.
   - `val_examples.pt` / `test_examples.pt`: cached qualitative batches.
   - `training_history.json`: per-epoch train/val metrics with per-class IoU snapshots.

5. **Evaluate a checkpoint** (without re-training):
   ```bash
   ./scripts/eval.py outputs/lr1e-04_wd1e-04/linear_probe.pth --split test
   ```

6. **Produce plots and overlays**:
   ```bash
   ./scripts/visualize_results.py outputs/lr1e-04_wd1e-04
   ```
   This generates `training_curves.png`, `palette_legend.png`, and side-by-side prediction overlays in `val_visuals/` and `test_visuals/`.

## Training details

- **Input resolution:**
  - DINOv3 & OpenCLIP operate on the raw 640×480 frames → 40×30 patch grid (patch size 16).
  - DINOv2 pads frames to 490×644 → 35×46 patch grid (patch size 14) to ensure patch alignment.
- **Targets:** pixel masks from `*_label.mat` are majority-voted to the patch grid (ignore if top class <55 % of the patch) and used for training. Pixel-level metrics are computed on bilinearly upsampled logits.
- **Backbones:** frozen transformers. `layers_to_hook` may contain negative indices (counting from the end) and multiple entries if you want to probe more than one block.
- **Head:** BatchNorm2d + 1×1 Conv, identical to the linear probe head in the DINO releases.
- **Optimiser:** AdamW with configurable LR / weight-decay grids. Grad clipping and gradient accumulation are built in.
- **Monitoring:** train/val loss, mIoU, per-class IoU (stored in JSON). Background classes are masked wherever `metric_ignore_indices` applies, including when writing `summary.json`.

## Extending the probe

- **Multi-layer probes:** change `model.layers_to_hook` and optionally add `model.primary_layer` in `configs/local.yaml`. The dataloaders and backbone wrapper already expose `{layer_idx: feature_map}`; only the head initialisation in `LinearProbeExperiment` needs to be adapted if you want per-layer heads.
- **Cross-validation or novel splits:** run `scripts/prepare_splits.py` with a different input list (e.g., `novel_split.txt`) and point the config to the resulting JSON.
- **Custom augmentations:** replace `get_default_image_transform()` in `src/data/transforms.py` or pass a different callable when constructing `UMDAffordanceDataset`.

## Troubleshooting checklist

- Ensure the backbone checkpoint paths in `configs/local*.yaml` are reachable; otherwise the trainer will raise an informative error during loading.
- Large models such as ViT-7B/16 demand significant GPU memory. Lower `training.batch_size` or switch to a smaller checkpoint if needed.
- DINOv2 ViT-g/14 emits warnings if xFormers is unavailable—this is expected; the fallback attention/FFN implementations are used automatically.
- If you change the validation ratio or seed, regenerate the split JSON so downstream scripts stay consistent.

Feel free to tailor the configs and scripts for additional probes (e.g., depth estimation, cross-dataset evaluation). The current layout keeps backbone extraction, data preparation, and visualisation modular so future experiments can reuse them with minimal changes.
