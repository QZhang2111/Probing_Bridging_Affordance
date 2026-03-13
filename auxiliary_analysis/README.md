# Optional Module: Auxiliary Analysis (P0)

Goal: provide supporting analysis scripts outside the three main benchmark tracks.

This module includes four runnable scripts:
- `scripts/run_knife_patch_similarity.py`: build knife anchor patch similarity maps.
- `scripts/run_cross_domain_similarity.py`: transfer knife anchor similarity to kitchen/office scenes.
- `scripts/run_pca_analysis.py`: project kitchen/office features into knife ROI PCA subspace.
- `scripts/run_clip_patch_probe.py`: CLIP patch-level prompt probing.

## Config
- Default: `configs/defaults.yaml`
- Template: `configs/local.example.yaml`

The first three scripts expect precomputed cache files (`output_root/<model_key>/meta/*.json` + token npz).
If missing, generate caches from your original preprocessing pipeline first.

## Quick Start
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/auxiliary_analysis

python scripts/run_knife_patch_similarity.py --config configs/defaults.yaml --model dinov3_vit7b16
python scripts/run_cross_domain_similarity.py --config configs/defaults.yaml --model dinov3_vit7b16
python scripts/run_pca_analysis.py --config configs/defaults.yaml --model dinov3_vit7b16

python scripts/run_clip_patch_probe.py \
  --image /path/to/toothbrush.png \
  --prompts "hold toothbrush" "brush teeth" \
  --output-root outputs_clip_probe
```

## Outputs
- `output_root/<model_key>/similarity/*`: similarity heatmaps and overlays
- `output_root/<model_key>/pca/knife/*`: PCA projection outputs
- `outputs_clip_probe/attention/*`: CLIP probing heatmaps, overlays, and npy files

## Dependencies
```bash
pip install -r requirements.txt
```
