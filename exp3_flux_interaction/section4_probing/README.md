# Section4 Probing

Small sandbox for inspecting cross-attention inside Flux / Stable Diffusion
pipelines on arbitrary (image, prompt) pairs.  The main entry-point is
`cross_attention_probe.py`, which

1. loads the requested diffusion pipeline (Flux img2img or SD img2img),
2. encodes the provided image into latents,
3. runs the denoising loop with the prompt tokens you care about,
4. records the true cross-attention probabilities between the image queries
   and the selected text tokens, and
5. writes Viridis heatmaps + overlays per token.

```
cd exps/affordance-experiments/section4_probing
python cross_attention_probe.py \
  --backend sd \
  --model-id stabilityai/stable-diffusion-2-1-base \
  --image ../Section2_exp/data/toothbrush.png \
  --prompt "hold toothbrush" \
  --tokens hold toothbrush \
  --output-root outputs_toothbrush_sd
```

For Flux (requires diffusers ≥ 0.30):

```
python cross_attention_probe.py \
  --backend flux \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image ../Section2_exp/data/toothbrush.png \
  --prompt "hold toothbrush" \
  --tokens hold toothbrush \
  --steps 20 --guidance 3.0 \
  --output-root outputs_toothbrush_flux
```

The script only touches code inside `section4_probing/`; the rest of the
repository remains unchanged.

## CLIP Patch Probing

To probe CLIP ViT patch-token responses without any training:

```
cd exps/affordance-experiments/section4_probing
python clip_patch_probe.py \
  --model-id laion/CLIP-ViT-B-16-laion2B-s34B-b88K \
  --image ../Section2_exp/data/toothbrush.png \
  --prompts "hold toothbrush" "brush teeth" \
  --feat-source value \
  --layer-index -1 \
  --force-size 224 \
  --output-root clip_patch_outputs
```

This computes cosine similarity between every ViT patch token and each prompt
embedding (defaulting to the transformer output tokens; pass `--feat-source value`
to reuse the value vectors, `--layer-index` to pick an earlier/later block, and
`--force-size` to run CLIP at its native 224×224 resolution before padding),
saving Viridis heatmaps / overlays under `clip_patch_outputs/attention/`.
