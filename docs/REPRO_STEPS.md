# Reproducibility Steps

Follow this checklist to run the three main experiments in sequence.

## 1. Environment
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
conda activate diffDINO
```

## 2. Path Configuration
1. Update dataset and checkpoint paths in:
- `geometry_probing/umd_linear_probing/configs/*.yaml`
- `fusion_zero_shot/src/agd20k_eval/config.yaml`
2. Ensure Flux model path is valid for Exp2.

## 3. Main Experiment 1: Geometry Probing
```bash
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval  -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

## 4. Main Experiment 2: Interaction Probing
```bash
python run.py interaction-probe -- \
  --model-id /path/to/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

## 5. Main Experiment 3: Fusion Zero-shot Evaluation
```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

## 6. Smoke Test
```bash
bash scripts/smoke_test.sh
```

## 7. Optional Supporting Analysis
```bash
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
python run.py aux-pca       -- --config auxiliary_analysis/configs/defaults.yaml --model dinov3_vit7b16
```
