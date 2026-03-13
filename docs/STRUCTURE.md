# Structure Guide

This repository is organized around three main experiments and one optional module.

## Main Experiments
1. `geometry_probing`: geometry probing on UMD/AGD20K
2. `interaction_probing`: interaction probing via Flux cross-attention
3. `fusion_zero_shot`: training-free geometry-interaction fusion on AGD20K

## Optional Module
- `auxiliary_analysis`: PCA projection, cross-scene similarity, and CLIP patch probing

## Shared Module
- `common`: shared IO/visualization/PCA/resize/similarity helpers reused by `auxiliary_analysis` and `fusion_zero_shot`

## Unified Launcher
Use root-level `run.py` for a consistent interface:

```bash
python run.py <command> -- <script_args>
```

Common commands:
- `geometry-train`, `geometry-eval`
- `interaction-probe`
- `fusion-eval`
- `aux-knife-sim`, `aux-cross-sim`, `aux-pca`, `aux-clip-probe`

## Quick Validation
```bash
bash scripts/smoke_test.sh
```

This check confirms that all three main experiment launchers are callable.
