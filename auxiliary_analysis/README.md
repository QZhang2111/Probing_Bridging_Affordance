# Auxiliary Analysis

Supporting analysis scripts for patch-level inspection and qualitative interpretation.

## Scripts

- `scripts/run_knife_patch_similarity.py`
- `scripts/run_cross_domain_similarity.py`
- `scripts/run_pca_analysis.py`
- `scripts/run_clip_patch_probe.py`

## Config

Default config: [`configs/defaults.yaml`](./configs/defaults.yaml)

Machine-specific template: [`configs/local.example.yaml`](./configs/local.example.yaml)

The first three scripts expect a precomputed cache at:

```text
output_root/<model_key>/
├── meta/
├── tokens/
└── pca/
```

The model keys in the config must match the cache directory names exactly.

## Quick Start

```bash
cd /path/to/Probing_Briding_Affordance
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
python run.py aux-pca -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
```

CLIP patch probe:

```bash
python run.py aux-clip-probe -- \
  --image datasets/analysis_images/knife.jpg \
  --prompts "hold knife" "cut with knife" \
  --output-root outputs/clip_probe
```
