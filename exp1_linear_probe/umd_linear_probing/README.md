# UMD Affordance Linear Probing (Minimal)

This workspace runs **linear probing** on the UMD part-affordance dataset with frozen vision backbones.

Supported backbones (minimal set):
- DINO, DINOv2, DINOv3
- OpenCLIP (CLIP)
- SAM
- SD2.1
- SigLIP

## Layout

```
.
├── configs/                 # Minimal configs
├── metadata/                # Dataset splits (pre-generated)
├── scripts/                 # CLI entry-points (train, eval)
├── src/                     # Core code (data, models, engine)
└── requirements.txt
```

## Usage (recommended)

Always pass a **single config**:

```bash
python scripts/train.py --config configs/dinov2.yaml
python scripts/eval.py  --config configs/dinov2.yaml
```

See `configs/README.md` for the minimal set of configs and expected path edits.

## Notes

- Update dataset paths and model checkpoints inside the YAMLs.
- Background handling and geometry channels are controlled by each YAML.
- This minimal version omits split-generation and visualization helpers.
