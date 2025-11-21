# 实验3：Flux Kontext / CLIP 交互先验探测

目标：提取生成模型（Flux Kontext / SD）跨注意力热图，以及 CLIP patch 相似度，解析交互先验。

代码来源：`./section4_probing`（跨注意力 & CLIP）+ `./FLUX/Flux_Kontext_Interaction`（Kontext 热图、warp）。

快捷运行
```bash
cd /home/li325/qing_workspace/probing_briding_affordance/experiments/exp3_flux_interaction
# Flux 跨注意力
./run_flux_probe.sh --backend flux \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image ../Section2_exp/data/toothbrush.png \
  --prompt "hold toothbrush" \
  --tokens hold toothbrush \
  --steps 20 --guidance 3.0

# CLIP patch 探测
./run_clip_probe.sh --model-id laion/CLIP-ViT-B-16-laion2B-s34B-b88K \
  --image ../Section2_exp/data/toothbrush.png \
  --prompts "hold toothbrush" "brush teeth" \
  --feat-source value --layer-index -1 --force-size 224
```

核心脚本（原路径）：
- `section4_probing/cross_attention_probe.py`
- `section4_probing/clip_patch_probe.py`
- Kontext 辅助：`FLUX/Flux_Kontext_Interaction/{visualize_flux_kontext_cross_attention.py, warp_heatmap_to_original.py}`

注意：需要 Flux/SD/CLIP 模型权重，`section4_probing/data` 未包含，请自备输入图。
