# Configs (Minimal Set)

This folder keeps only the base configs needed for geometry probing.

Base configs:
- `clip.yaml`
- `dino.yaml`
- `dinov2.yaml`
- `dinov3.yaml`
- `sam.yaml`
- `sd21.yaml`
- `siglip.yaml`

Usage (recommended):
```bash
python scripts/train.py --config configs/dinov2.yaml
python scripts/eval.py  --config configs/dinov2.yaml
```

Notes:
- Default/local configs are not used in the minimal setup; pass `--config` explicitly.
- Update dataset/model paths inside the YAMLs to match your environment.
