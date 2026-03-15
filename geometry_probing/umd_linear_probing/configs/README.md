# Geometry Configs

This folder keeps the public default configs for geometry probing.

Available configs:

- `clip.yaml`
- `dino.yaml`
- `dinov2.yaml`
- `dinov3.yaml`
- `sam.yaml`
- `sd21.yaml`
- `siglip.yaml`

Usage:

```bash
python scripts/train.py --config configs/dinov2.yaml
python scripts/eval.py /path/to/linear_probe.pth --config configs/dinov2.yaml --split test
```

Conventions:

- dataset assets resolve from `datasets/`
- model source trees and checkpoints resolve from `models/`
- geometry side-data manifest resolves from `metadata/splits/`
