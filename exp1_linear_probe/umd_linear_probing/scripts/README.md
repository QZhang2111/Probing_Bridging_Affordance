## Experiment Scripts Usage

All commands are meant to be executed from the project root:
```
cd /home/li325/qing_workspace/exps/affordance-experiments/umd_linear_probing
```

### `prepare_splits.py`
Generate train/val/test mappings (example: 80/20 split with class coverage).
```bash
python scripts/prepare_splits.py \
  --dataset-root /home/li325/qing_workspace/dataset/UMD/part-affordance-dataset \
  --category-split /home/li325/qing_workspace/dataset/UMD/part-affordance-dataset/category_split.txt \
  --val-ratio 0.2 \
  --val-seed 42 \
  --ensure-val-all-classes \
  --num-classes 8 \
  --exclude-background \
  --output metadata/splits/category_split_seed42_v20.json
```

### `train.py` (DINOv3 linear probe)
```bash
python scripts/train.py \
  --defaults configs/default.yaml \
  --local configs/local.yaml
```

### `train.py` (OpenCLIP linear probe)
```bash
python scripts/train.py \
  --defaults configs/default.yaml \
  --local configs/local_openclip.yaml
```

### `train.py` (DINOv2 ViT-g/14 linear probe)
```bash
python scripts/train.py \
  --defaults configs/default.yaml \
  --local configs/local_dinov2.yaml
```

### `eval.py`
Evaluate a saved checkpoint on the test split (adjust path as needed).
```bash
python scripts/eval.py \
  outputs/lr1e-04_wd1e-04/linear_probe.pth \
  --split test \
  --defaults configs/default.yaml \
  --local configs/local.yaml
```

### `visualize_results.py`
Generate training curves and qualitative overlays for a run directory.
```bash
python scripts/visualize_results.py \
  outputs/lr1e-04_wd1e-04 \
  --defaults configs/default.yaml \
  --local configs/local.yaml
```
