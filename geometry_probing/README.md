# Main Experiment 1: Geometry Probing (Linear Probe)

Goal: evaluate geometric affordance awareness on UMD/AGD20K via linear probing.

The core implementation is kept under `./umd_linear_probing`.

## Recommended Entry (Unified)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval  -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

## Direct Entry (Backward Compatible)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/geometry_probing
python ./train.py --config ./umd_linear_probing/configs/dinov2.yaml
python ./eval.py  --config ./umd_linear_probing/configs/dinov2.yaml
```

## Key Files
- Train/eval scripts: `umd_linear_probing/scripts/train.py`, `umd_linear_probing/scripts/eval.py`
- Model configs: `umd_linear_probing/configs/*.yaml`
- Config guide: `umd_linear_probing/configs/README.md`

Update dataset and checkpoint paths in YAML files before running.
