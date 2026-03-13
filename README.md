# Probing & Bridging Affordance Experiments

This repository provides a streamlined, reproducible implementation of three main experimental tracks for affordance reasoning:

1. **Geometry Probing** (linear probing on UMD/AGD20K-style setup)
2. **Interaction Probing** (Flux Kontext cross-attention)
3. **Geometry × Interaction Fusion** (zero-shot evaluation on AGD20K)

An optional auxiliary module is also included for additional analysis.

## Who This README Is For

If you are installing and running this project for the first time, this guide is meant to be enough to get you from zero to running commands.

## Repository Layout

- `geometry_probing/`: main geometry probing pipeline (UMD linear probing code)
- `interaction_probing/`: interaction prior probing via Flux cross-attention
- `fusion_zero_shot/`: geometry + interaction zero-shot fusion pipeline
- `auxiliary_analysis/`: optional support analysis scripts
- `common/`: shared helper utilities
- `run.py`: unified launcher for all tasks

## Environment Setup

### 1) Create/activate conda environment

This project is developed and tested in:

```bash
conda activate diffDINO
```

If you need to create it:

```bash
conda create -n diffDINO python=3.10 -y
conda activate diffDINO
```

### 2) Install Python dependencies

For the three main experiments, install the main requirement set:

```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
pip install -r geometry_probing/umd_linear_probing/requirements.txt
```

For optional auxiliary analysis (not required for main results):

```bash
pip install -r auxiliary_analysis/requirements.txt
```

### 3) PyTorch/CUDA note

If your machine requires a specific CUDA build of PyTorch, install matching `torch`/`torchvision` first (from official PyTorch instructions), then run the requirements commands above.

## Required Data and Model Assets

This repository does **not** ship datasets or model checkpoints. You must prepare them locally.

### A) Datasets

You need at least:

- **UMD part-affordance dataset** (for geometry probing)
- **AGD20K Unseen testset** (for fusion evaluation)

Typical paths used in current configs:

- UMD root: `/home/li325/qing_workspace/dataset/UMD/part-affordance-dataset`
- AGD20K root: `/home/li325/qing_workspace/dataset/AGD20K/AGD20K/Unseen/testset`

### B) Checkpoints / model weights

Depending on which configs you run, prepare these assets:

- **Flux Kontext model** (local directory), e.g. `FLUX.1-Kontext-dev`
- **DINOv2 checkpoint** (for geometry probing and/or fusion backend)
- **DINOv3 checkpoint** (for geometry probing and fusion backend)
- Optional backbones referenced in configs:
  - Stable Diffusion 2.1 (`stabilityai/stable-diffusion-2-1`)
  - SAM weights
  - OpenCLIP / SigLIP models

Important: many YAMLs currently contain machine-specific absolute paths. You must replace them with your local paths.

## Configuration Checklist (Before Running)

Update paths in these files first:

1. `geometry_probing/umd_linear_probing/configs/*.yaml`
2. `fusion_zero_shot/src/agd20k_eval/config.yaml`
3. (optional) `auxiliary_analysis/configs/defaults.yaml`

At minimum, verify:

- dataset roots
- split/metadata files
- model repository paths
- checkpoint paths
- output directories

## Quick Start: Three Main Experiments

From repository root:

```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
conda activate diffDINO
```

### 1) Main Experiment 1: Geometry Probing

```bash
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval  -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

### 2) Main Experiment 2: Interaction Probing

```bash
python run.py interaction-probe -- \
  --model-id /path/to/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

### 3) Main Experiment 3: Fusion Zero-shot Evaluation

```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

## Optional: Auxiliary Analysis

```bash
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-pca       -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
```

## Unified Launcher Command Map

Main commands:

- `geometry-train`
- `geometry-eval`
- `interaction-probe`
- `fusion-eval`

Auxiliary commands:

- `aux-knife-sim`
- `aux-cross-sim`
- `aux-pca`
- `aux-clip-probe`

Backward-compatible aliases (`exp1-*`, `exp3-*`, `exp4-*`, `exp2-*`) are still available.

## Sanity Check

This does not run full experiments; it only verifies all three main launchers are callable.

```bash
python run.py geometry-train -- --help
python run.py interaction-probe -- --help
python run.py fusion-eval -- --help
```

## Troubleshooting

### `FileNotFoundError` for dataset/checkpoint

- Check every absolute path in your selected YAML config.
- Ensure mounted paths are visible inside your current environment/session.

### CUDA out-of-memory

- Reduce batch size in geometry probing configs.
- Use lighter model configs first (for smoke testing).

### Diffusers / Transformers version issues

- Reinstall with a clean environment.
- Ensure `torch`, `diffusers`, and `transformers` are mutually compatible.

### Flux model load errors

- Confirm `--model-id` points to a valid local Flux Kontext directory (or valid HF model ID if supported by your setup).

## Reproducibility Notes

- Keep a copy of your modified YAMLs for exact reruns.
- Record commit hash and environment package versions when reporting results.

## Project TODO (Tracked Here)

- Add a single consolidated dependency lock file for reproducible installs.
- Add benchmark target metrics and expected output examples per experiment.
- Add optional automated CI smoke checks for launcher commands.

## License / Third-Party

This repository includes third-party components under `fusion_zero_shot/src/dino/third_party/`. Follow their original licenses when redistributing or modifying.
