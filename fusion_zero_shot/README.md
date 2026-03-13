# Main Experiment 3: Geometry × Interaction Fusion (AGD20K)

Goal: evaluate training-free geometry-interaction fusion on AGD20K in a zero-shot setting.

## Recommended Entry (Unified)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

## Direct Entry (Backward Compatible)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/fusion_zero_shot
python ./run_agd20k_eval.py --config ./src/agd20k_eval/config.yaml
```

## Main Components
- `src/agd20k_eval/`: evaluation pipeline
- `src/flux_kontext_interaction/`: cross-attention extraction and warping tools
- `src/pipeline/`: ROI/PCA/geometry fusion modules
- `src/dino/`: DINO feature extraction dependencies

Configure local dataset and model paths in `src/agd20k_eval/config.yaml` before running.
