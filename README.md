# Probing & Bridging Affordance Experiments

A streamlined repository for three main affordance-reasoning experiments:

1. **Geometry Probing** (linear probing)
2. **Interaction Probing** (Flux Kontext cross-attention)
3. **Geometry × Interaction Fusion** (zero-shot AGD20K evaluation)

An optional auxiliary analysis module is included for supporting visual analysis.

## Repository Layout

- `geometry_probing/` - main geometry probing pipeline
- `interaction_probing/` - Flux cross-attention probing
- `fusion_zero_shot/` - geometry + interaction fusion pipeline
- `auxiliary_analysis/` - optional analysis scripts and shared utilities
- `datasets/` - reserved location for dataset downloads/symlinks
- `models/` - reserved location for model downloads/symlinks
- `run.py` - unified launcher

## Environment Setup

### 1) Conda environment

```bash
conda create -n diffDINO python=3.10 -y
conda activate diffDINO
```

If you already have the environment, just activate it:

```bash
conda activate diffDINO
```

### 2) Install dependencies

Main experiments:

```bash
cd /path/to/Probing_Briding_Affordance
pip install -r geometry_probing/umd_linear_probing/requirements.txt
```

Optional auxiliary analysis:

```bash
pip install -r auxiliary_analysis/requirements.txt
```

### 3) PyTorch/CUDA note

If your machine requires a specific CUDA build, install matching `torch`/`torchvision` first from official PyTorch instructions, then install the requirements above.

## Data and Model Preparation

This repository does not include datasets or checkpoints. Place them under `datasets/` and `models/` (or symlink to external storage).

### Recommended dataset structure

```text
datasets/
  UMD/
    part-affordance-dataset/
  AGD20K/
    AGD20K/
      Unseen/
        testset/
```

### Recommended model structure

```text
models/
  FLUX.1-Kontext-dev/
  dinov2_vitb14_pretrain.pth
  dinov3_vit7b16_pretrain_lvd1689m.pth
  sam_vit_b_01ec64.pth          # optional (if used by selected config)
```

See:
- `datasets/README.md`
- `models/README.md`

## Required Assets by Experiment

### Main Experiment 1 (Geometry Probing)

Required:
- UMD dataset
- at least one selected backbone checkpoint (e.g., DINOv2 or DINOv3)
- optional geometry side-data referenced by config (if enabled)

Config files:
- `geometry_probing/umd_linear_probing/configs/*.yaml`

### Main Experiment 2 (Interaction Probing)

Required:
- Flux Kontext model directory
- input image(s)

Command uses `--model-id` and `--image` directly.

### Main Experiment 3 (Fusion Zero-shot)

Required:
- AGD20K Unseen testset
- Flux Kontext model directory
- DINO checkpoint/backend selected in `fusion_zero_shot/src/agd20k_eval/config.yaml`

## Configuration Checklist (Before Running)

Update paths in:

1. `geometry_probing/umd_linear_probing/configs/*.yaml`
2. `fusion_zero_shot/src/agd20k_eval/config.yaml`
3. (optional) `auxiliary_analysis/configs/defaults.yaml`

Verify at least:
- dataset roots
- checkpoint/model paths
- split/metadata paths
- output/cache directories

## Run Commands

From repository root:

```bash
cd /path/to/Probing_Briding_Affordance
conda activate diffDINO
```

### Main Experiment 1: Geometry Probing

```bash
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval  -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

### Main Experiment 2: Interaction Probing

```bash
python run.py interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

### Main Experiment 3: Fusion Zero-shot Evaluation

```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

### Optional: Auxiliary Analysis

```bash
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-pca       -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
```

## Launcher Command Map

Main:
- `geometry-train`
- `geometry-eval`
- `interaction-probe`
- `fusion-eval`

Optional:
- `aux-knife-sim`
- `aux-cross-sim`
- `aux-pca`
- `aux-clip-probe`

Backward-compatible aliases (`exp1-*`, `exp2-*`, `exp3-*`, `exp4-*`) remain available.

## Quick Sanity Check

Check that all three main launchers are callable:

```bash
python run.py geometry-train -- --help
python run.py interaction-probe -- --help
python run.py fusion-eval -- --help
```

## Troubleshooting

### File not found errors

- Usually caused by unresolved local paths in YAML configs.
- Confirm files exist and the path is correct relative to your machine.

### CUDA out-of-memory

- Reduce batch size in geometry configs.
- Start from lighter model configs first.

### Diffusers/Transformers compatibility issues

- Recreate a clean environment.
- Ensure `torch`, `diffusers`, and `transformers` versions are compatible.

## TODO (Tracked in README)

- Add one consolidated lock file for stricter reproducibility.
- Add expected output examples and target metrics per experiment.
- Add CI-level smoke checks for launcher commands.

## Third-Party Notice

Third-party code is included under `fusion_zero_shot/src/dino/third_party/`. Follow original licenses for redistribution and modification.
