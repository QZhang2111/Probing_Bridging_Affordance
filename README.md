# Probing & Bridging Affordance

Code release for studying affordance reasoning from four complementary angles:

1. `geometry probing`: dense linear probing on visual backbones with optional geometry side inputs
2. `auxiliary analysis`: patch similarity, PCA projection, and CLIP patch probing for qualitative analysis
3. `interaction probing`: FLUX Kontext cross-attention probing from text-conditioned interaction prompts
4. `fusion zero-shot`: zero-shot affordance prediction on AGD20K by combining geometry and interaction priors

The repository is organized as a reproducible research codebase rather than a monolithic training framework. Each module can be run independently, and `run.py` provides a unified entrypoint for the main commands.

## Introduction

Affordance understanding is not purely geometric and not purely interaction-driven. This repository isolates and connects both sources of information:

- geometry priors from frozen visual representations
- interaction priors from text-conditioned generative attention
- fusion strategies that combine both at inference time

The code is structured around the full analysis pipeline used in this project: first probe geometric cues, then inspect auxiliary patch-level behavior, then extract interaction heatmaps, and finally evaluate fused zero-shot predictions.

## Repository Layout

```text
Probing_Briding_Affordance/
├── geometry_probing/      # linear probing on UMD part-affordance data
├── auxiliary_analysis/    # supporting analysis scripts on precomputed tokens
├── interaction_probing/   # FLUX Kontext cross-attention probing
├── fusion_zero_shot/      # AGD20K zero-shot fusion pipeline
├── datasets/              # reserved location for datasets or symlinks
├── models/                # reserved location for checkpoints or symlinks
└── run.py                 # unified launcher
```

## Quick Start

### 1. Environment

This repository is expected to run in:

```bash
conda activate diffDINO
```

If you need to create it from scratch:

```bash
conda create -n diffDINO python=3.10 -y
conda activate diffDINO
cd /path/to/Probing_Briding_Affordance
pip install -r geometry_probing/umd_linear_probing/requirements.txt
pip install -r auxiliary_analysis/requirements.txt
```

If your machine requires a specific CUDA build, install the matching `torch` and `torchvision` first, then install the repository requirements.

### 2. Data and Model Placement

This repository does not ship datasets or model weights. Put them under `datasets/` and `models/`, or create symlinks to external storage.

Recommended layout:

```text
datasets/
├── UMD/
│   └── part-affordance-dataset/
└── AGD20K/
    └── AGD20K/
        └── Unseen/
            └── testset/

models/
├── FLUX.1-Kontext-dev/
├── dinov2_vitb14_pretrain.pth
├── dinov3_vit7b16_pretrain_lvd1689m.pth
└── sam_vit_b_01ec64.pth        # optional
```

See [datasets/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/datasets/README.md) and [models/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/models/README.md) for the expected storage convention.

### 3. Update Config Paths

Before running experiments, check the paths in:

1. [geometry_probing/umd_linear_probing/configs](/home/li325/qing_workspace/Probing_Briding_Affordance/geometry_probing/umd_linear_probing/configs)
2. [auxiliary_analysis/configs/defaults.yaml](/home/li325/qing_workspace/Probing_Briding_Affordance/auxiliary_analysis/configs/defaults.yaml)
3. [fusion_zero_shot/src/agd20k_eval/config.yaml](/home/li325/qing_workspace/Probing_Briding_Affordance/fusion_zero_shot/src/agd20k_eval/config.yaml)

At minimum, verify:

- dataset roots
- checkpoint paths
- model directories
- split and metadata paths
- output and cache directories

## Experiments

### 1. Geometry Probing

Purpose: probe how well frozen backbone features support dense affordance prediction on UMD.

Required assets:

- UMD part-affordance dataset
- one selected backbone checkpoint such as DINOv2 or DINOv3
- geometry side-data if enabled by config

Run training:

```bash
cd /path/to/Probing_Briding_Affordance
conda activate diffDINO
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

Run evaluation:

```bash
python run.py geometry-eval -- \
  /path/to/linear_probe.pth \
  --config geometry_probing/umd_linear_probing/configs/dinov2.yaml \
  --split test
```

More details: [geometry_probing/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/geometry_probing/README.md)

### 2. Auxiliary Analysis

Purpose: inspect patch-level behavior using precomputed token caches, anchor similarities, PCA projections, and CLIP patch probing.

Required assets:

- precomputed token cache for the selected model
- source analysis images referenced by the config

Examples:

```bash
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-pca -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
```

CLIP patch probing:

```bash
python run.py aux-clip-probe -- \
  --image /path/to/image.png \
  --prompts "hold knife" "cut with knife" \
  --output-root ./outputs/clip_probe
```

More details: [auxiliary_analysis/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/auxiliary_analysis/README.md)

### 3. Interaction Probing

Purpose: extract interaction heatmaps from FLUX Kontext cross-attention under affordance-specific prompts.

Required assets:

- local FLUX Kontext model directory
- input image

Run:

```bash
python run.py interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance hold \
  --steps 20 \
  --guidance 3.0
```

Outputs are written under `probe_outputs/`.

More details: [interaction_probing/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/interaction_probing/README.md)

### 4. Fusion Zero-Shot

Purpose: evaluate zero-shot affordance localization on AGD20K by fusing geometric and interaction cues.

Required assets:

- AGD20K unseen test set
- local FLUX Kontext model directory
- selected DINO backend or precomputed geometry cache, depending on config

Run:

```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

More details: [fusion_zero_shot/README.md](/home/li325/qing_workspace/Probing_Briding_Affordance/fusion_zero_shot/README.md)

## Unified Launcher

Main commands:

- `geometry-train`
- `geometry-eval`
- `interaction-probe`
- `fusion-eval`

Optional analysis commands:

- `aux-knife-sim`
- `aux-cross-sim`
- `aux-pca`
- `aux-clip-probe`

Backward-compatible aliases such as `exp1-*`, `exp2-*`, `exp3-*`, and `exp4-*` are still supported by [run.py](/home/li325/qing_workspace/Probing_Briding_Affordance/run.py).

## Recommended Validation Order

For a fresh machine or a first-time setup, validate the repository in this order:

1. `geometry probing`
2. `auxiliary analysis`
3. `interaction probing`
4. `fusion zero-shot`

This order isolates data, cache, model, and pipeline issues progressively instead of debugging the full fusion stack first.

## Minimal Sanity Checks

Launcher checks:

```bash
python run.py geometry-train -- --help
python run.py interaction-probe -- --help
python run.py fusion-eval -- --help
```

If you want a first runtime check, start with:

```bash
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

and then:

```bash
python run.py interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold object" \
  --affordance hold \
  --steps 1
```

## Troubleshooting

### Missing file or path errors

Most runtime failures come from unresolved local paths in YAML configs or from missing checkpoints. Check the config paths first.

### CUDA out-of-memory

Reduce batch size in geometry configs, use a lighter model, or shorten inference settings for debugging.

### Diffusers, Transformers, or xFormers mismatches

If FLUX or diffusion-based code fails during import or model loading, rebuild the environment with a compatible `torch`, `transformers`, `diffusers`, and `xformers` stack.

### Auxiliary analysis key mismatch

The auxiliary scripts expect the model keys in `auxiliary_analysis/configs/defaults.yaml` to match the cache directory names exactly.

## Notes

- `datasets/` and `models/` are placeholders for public reproducibility. Symlinks are supported.
- `fusion_zero_shot/src/dino/third_party/` contains third-party code and should follow its original licensing terms.
- The repository currently prioritizes reproducible experiment entrypoints over packaging as a pip module.

## TODO

- Add a single consolidated environment lock file
- Add public download instructions for all required model weights
- Add expected outputs and reference metrics for each experiment
- Add CI-style smoke tests for the main launchers
