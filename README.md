# Probing & Bridging Affordance

Public-facing repository for affordance reasoning experiments built around four components:

1. `geometry_probing`: dense linear probing on frozen visual backbones
2. `auxiliary_analysis`: patch-level supporting analysis on precomputed token caches
3. `interaction_probing`: FLUX Kontext cross-attention probing
4. `fusion_zero_shot`: geometry-interaction fusion on AGD20K

The repository is organized around runnable experiment entrypoints rather than a single training framework. Each module can be used independently, and [`run.py`](./run.py) provides a unified launcher.

## Repository Layout

```text
Probing_Bridging_Affordance/
├── geometry_probing/
├── auxiliary_analysis/
├── interaction_probing/
├── fusion_zero_shot/
├── datasets/
├── models/
└── run.py
```

## Environment

Recommended environment name:

```bash
conda activate affordance
```

Recommended setup:

```bash
conda create -n affordance python=3.10 -y
conda activate affordance
cd /path/to/Probing_Bridging_Affordance
pip install -r requirements.txt
```

CUDA note:

- install a `torch` and `torchvision` build that matches your driver and CUDA runtime before installing the repository requirements
- the current validation work for this repository was performed in a Python 3.10 environment with CUDA-enabled PyTorch

Example for CUDA 12.1:

```bash
conda create -n affordance python=3.10 -y
conda activate affordance
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The top-level [`requirements.txt`](./requirements.txt) is the recommended public installation entrypoint. It reuses the maintained module requirement files and adds the extra packages needed by interaction probing and fusion.

## Data and Model Layout

This repository does not include datasets, checkpoints, or external model source trees. The tracked configs assume they are placed under `datasets/` and `models/` or exposed there via symlinks.

Recommended structure:

```text
datasets/
├── UMD/
│   └── part-affordance-dataset/
├── AGD20K/
│   └── AGD20K/
│       └── Unseen/
│           └── testset/
└── analysis_images/
    ├── knife.jpg
    ├── kitchen.jpg
    └── office.jpg

models/
├── FLUX.1-Kontext-dev/
├── dinov2/
├── dinov3/
├── dino/
├── open_clip/
│   └── src/
├── dinov2_vitb14_pretrain.pth
├── dinov3_vit7b16_pretrain_lvd1689m.pth
└── dino_vitbase16_pretrain.pth
```

See [datasets/README.md](./datasets/README.md) and [models/README.md](./models/README.md).

## Configuration Policy

Tracked configs are now public-safe:

- no hardcoded `/home/...` paths
- dataset and model assets resolve from `datasets/` and `models/`
- machine-specific overrides should live in untracked local files such as `local.yaml`

Before running experiments, check:

1. [`geometry_probing/umd_linear_probing/configs/`](./geometry_probing/umd_linear_probing/configs/)
2. [`auxiliary_analysis/configs/defaults.yaml`](./auxiliary_analysis/configs/defaults.yaml)
3. [`fusion_zero_shot/src/agd20k_eval/config.yaml`](./fusion_zero_shot/src/agd20k_eval/config.yaml)

## Experiments

### 1. Geometry Probing

Purpose: evaluate how well frozen backbone features support dense affordance prediction on UMD.

```bash
python run.py geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
python run.py geometry-eval -- \
  /path/to/linear_probe.pth \
  --config geometry_probing/umd_linear_probing/configs/dinov2.yaml \
  --split test
```

More details: [geometry_probing/README.md](./geometry_probing/README.md)

### 2. Auxiliary Analysis

Purpose: inspect patch similarity, PCA structure, and prompt-conditioned CLIP patch responses.

```bash
python run.py aux-knife-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
python run.py aux-cross-sim -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
python run.py aux-pca -- --config auxiliary_analysis/configs/defaults.yaml --model dinov2_vitb14_layer12
```

CLIP patch probing:

```bash
python run.py aux-clip-probe -- \
  --image datasets/analysis_images/knife.jpg \
  --prompts "hold knife" "cut with knife" \
  --output-root outputs/clip_probe
```

More details: [auxiliary_analysis/README.md](./auxiliary_analysis/README.md)

### 3. Interaction Probing

Purpose: extract affordance-specific cross-attention heatmaps from FLUX Kontext.

```bash
python run.py interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance hold \
  --steps 20 \
  --guidance 3.0
```

Outputs are written to `probe_outputs/`.

More details: [interaction_probing/README.md](./interaction_probing/README.md)

### 4. Fusion Zero-Shot

Purpose: evaluate geometry-interaction fusion on AGD20K unseen samples.

```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

More details: [fusion_zero_shot/README.md](./fusion_zero_shot/README.md)

## Unified Launcher

Main commands:

- `geometry-train`
- `geometry-eval`
- `interaction-probe`
- `fusion-eval`

Optional analysis commands:

- `aux-knife-sim`
- `aux-cross-sim`
- `aux-pca`
- `aux-clip-probe`

Backward-compatible aliases such as `exp1-*`, `exp2-*`, `exp3-*`, and `exp4-*` are still supported.

## Recommended Validation Order

For a fresh setup, validate the repository in this order:

1. `geometry_probing`
2. `auxiliary_analysis`
3. `interaction_probing`
4. `fusion_zero_shot`

This isolates dataset, cache, model, and fusion issues progressively.

## Minimal Sanity Checks

```bash
python run.py geometry-train -- --help
python run.py interaction-probe -- --help
python run.py fusion-eval -- --help
```

## Notes

- `datasets/` and `models/` are public placeholders. Symlinks are supported.
- `fusion_zero_shot/src/dino/third_party/` contains third-party code that follows its original license terms.
- Local machine overrides should not be committed. The repository ignores `local.yaml` style files.

## Citation

If you use this repository in academic work, cite the associated paper:

```bibtex
@article{zhang2026probing,
  title={Probing and Bridging Geometry-Interaction Cues for Affordance Reasoning in Vision Foundation Models},
  author={Zhang, Qing and Li, Xuesong and Zhang, Jing},
  journal={arXiv preprint arXiv:2602.20501},
  year={2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](./LICENSE).

Third-party components under `fusion_zero_shot/src/dino/third_party/` retain their original licenses.
