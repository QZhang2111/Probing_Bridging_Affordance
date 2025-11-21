# DINO Affordance Toolkit

Unified workspace for DINOv3 affordance experiments. The folder is organised to
separate reusable utilities, configuration, and generated artefacts so that new
experiments can reuse the same tooling without relying on absolute paths.

## Layout
- `assets/` – lightweight shared resources (colour palettes, demo prompts, etc.).
- `configs/` – global settings. Copy `local.example.yaml` to `local.yaml` to point
to external checkpoints or datasets; dataset index CSVs now live under
  `configs/lists/`.
- `data/` – local staging area for datasets (UMD, my_kitchen, scenefun3d, ...).
- `experiments/` – curated experiment results; each subdirectory owns its own
  README and output structure (see `F1_PCA`, `F2_SIM`, `F3_CLUST`).
- `outputs/` – generated caches/logs (ignored by git). Scripts write dense token
  caches here via the new settings loader.
- `scripts/` – command-line entry points (feature extraction, PCA batching,
  inspection utilities) that import helpers from `src/`.
- `src/` – reusable Python modules (settings loader, feature extraction helpers,
  etc.).
- `third_party/` – vendored Dinov3 repository used as the Torch Hub source.

## Configuration
- Default paths are defined in `configs/defaults.yaml`. Only relative paths to
  this directory are committed.
- Override anything by creating `configs/local.yaml` (ignored by git) or setting
  environment variables; e.g. provide `model.checkpoint_path` or
  `paths.umd_label_root` when working with the UMD affordance dataset.
- Scripts call `dino.src.settings.get_settings()` to pick up these values; avoid
  editing hard-coded paths inside experiment folders.

## Running scripts
Examples (assumes `configs/local.yaml` supplies a valid checkpoint path):

```bash
# Batch feature extraction for all datasets
time python -m dino.scripts.batch_extract_features

# Single-image ROI PCA
python experiments/F1_PCA/ROI-PCA/run_single_roi_pca.py \
  --rgb data/UMD/mug/mug_01_00000001_rgb.jpg \
  --tokens-dir outputs/cache/tokens/umd \
  --masks-dir outputs/cache/masks/umd
```

All scripts accept `--help` for the latest arguments.
