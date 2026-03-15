# Models Folder

Place checkpoints, local model directories, and external source trees here or expose them via symlinks.

Suggested structure:

- `models/FLUX.1-Kontext-dev/`
- `models/dinov2/`
- `models/dinov3/`
- `models/dino/`
- `models/open_clip/src/`
- `models/dinov2_vitb14_pretrain.pth`
- `models/dinov3_vit7b16_pretrain_lvd1689m.pth`
- `models/dino_vitbase16_pretrain.pth`
- optional: `models/sam_vit_b_01ec64.pth`

The geometry probing configs expect both:

- checkpoint files
- local source trees for backbones loaded with `torch.hub.load(..., source=\"local\")`
