# Fusion Zero-Shot

Zero-shot affordance localization on AGD20K by combining geometry and interaction priors.

## Recommended Entry

```bash
cd /path/to/Probing_Briding_Affordance
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

## Direct Entry

```bash
cd /path/to/Probing_Briding_Affordance/fusion_zero_shot
python ./run_agd20k_eval.py --config ./src/agd20k_eval/config.yaml
```

## Main Components

- `src/agd20k_eval/`: evaluation loop and metric aggregation
- `src/flux_kontext_interaction/`: attention extraction and heatmap warping
- `src/pipeline/`: ROI, PCA, geometry, and fusion stages
- `src/dino/`: DINO feature extraction dependency layer

## Public Default Assets

- dataset under `datasets/AGD20K/AGD20K/Unseen/testset/`
- FLUX model under `models/FLUX.1-Kontext-dev/`
- DINO checkpoints under `models/`
