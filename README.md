# Probing & Bridging Affordance Experiments

A streamlined reproducible repository centered on three main experimental tracks:
1. Geometry Probing (UMD linear probing)
2. Interaction Probing (Flux cross-attention)
3. Geometry × Interaction Fusion (AGD20K zero-shot evaluation)

Default environment:
```bash
conda activate diffDINO
```

## Repository Layout
- `geometry_probing/`: main geometry probing experiment
- `interaction_probing/`: interaction prior probing with Flux
- `fusion_zero_shot/`: geometry + interaction fusion evaluation
- `auxiliary_analysis/`: optional supporting analysis module (not required for main results)
- `common/`: shared utilities used across experiments
- `run.py`: unified launcher for all experiments

## Quick Start

### Main Experiment 1: Geometry Probing
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval  -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

### Main Experiment 2: Interaction Probing
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py interaction-probe -- \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image /path/to/toothbrush.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

### Main Experiment 3: Fusion Zero-shot Evaluation
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

## Optional: Auxiliary Analysis
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-pca       -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
```

## Validation
```bash
bash scripts/smoke_test.sh
```

## Notes
- This repository includes code and configs only.
- Datasets, checkpoints, caches, and outputs must be prepared locally.
- Update local paths in config files before running experiments.
