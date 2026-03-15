# Geometry Probing

Dense linear probing on the UMD part-affordance dataset.

## Entry Points

Recommended:

```bash
cd /path/to/Probing_Briding_Affordance
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval -- \
  /path/to/linear_probe.pth \
  --config geometry_probing/umd_linear_probing/configs/dinov2.yaml \
  --split test
```

Direct:

```bash
cd /path/to/Probing_Briding_Affordance/geometry_probing
python train.py --config ./umd_linear_probing/configs/dinov2.yaml
python eval.py /path/to/linear_probe.pth --config ./umd_linear_probing/configs/dinov2.yaml --split test
```

## Assets

Expected public defaults:

- dataset under `datasets/UMD/part-affordance-dataset/`
- model source trees under `models/`
- checkpoints under `models/*.pth`

Main configs live in [`umd_linear_probing/configs/`](./umd_linear_probing/configs/).
