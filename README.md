# Probing & Bridging Affordance Experiments

基于论文思路，将代码按 4 个实验重组到 `experiments/` 下，每个实验目录自包含所需代码（逻辑未改）。默认环境：`conda activate diffDINO`，需自备模型权重、数据和输出路径。

目录总览
- `experiments/exp1_linear_probe`：几何线性探针（UMD/AGD20K）
- `experiments/exp2_geometry_vis`：几何可视化（PCA、余弦）与论文图脚本
- `experiments/exp3_flux_interaction`：Flux/SD 跨注意力与 CLIP patch 探测（交互先验）
- `experiments/exp4_fusion_zero_shot`：几何 + 交互融合零样本流水线（依赖 DINO/Flux/Section2 模块）

## 实验1：线性探针
目的：在 UMD/AGD20K 上验证 VFMs 的几何感知与 affordance 相关性。  
入口：`experiments/exp1_linear_probe/run.sh` 调用 `./umd_linear_probing/scripts/train.py`
示例：
```bash
cd experiments/exp1_linear_probe
./run.sh --config ./umd_linear_probing/configs/single_layer/dinov2/8_both.yaml
```
准备：
- 在 `umd_linear_probing/configs/local*.yaml` 配置数据集与预训练权重。
- 评估/可视化：`umd_linear_probing/scripts/{eval.py,visualize_results.py}`。

## 实验2：几何可视化（PCA/余弦）
目的：展示 DINO 几何结构，可视化 PCA/余弦相似度，生成论文图。  
入口：`experiments/exp2_geometry_vis/run_pca.sh` → `./Section2_exp/scripts/extract_all.py`
示例：
```bash
cd experiments/exp2_geometry_vis
./run_pca.sh --config ./Section2_exp/config/settings.yaml
```
组成：
- 核心模块：`Section2_exp/modules/{feature,roi,pca,similarity,io,config}.py`
- 脚本：`Section2_exp/scripts/{cross_domain_pca.py,cross_domain_similarity.py,extract_all.py,knife_roi_pipeline.py,mark_mug_point.py}`
- 论文图：`Figure/sec2/sec2_1.py`, `Figure/sec2/sec2_2.py`, `Figure/sup/sup1.py`
- 示例叠加：`Section1_vis/umd_overlay.py`  
准备：`Section2_exp/config/settings.yaml` 填写数据/缓存路径（数据与缓存未包含）。

## 实验3：交互先验探测（Flux/SD/CLIP）
目的：提取生成模型跨注意力热图、CLIP patch 相似度，分析交互先验。  
入口：
- Flux/SD：`experiments/exp3_flux_interaction/run_flux_probe.sh` → `./section4_probing/cross_attention_probe.py`
- CLIP：`experiments/exp3_flux_interaction/run_clip_probe.sh` → `./section4_probing/clip_patch_probe.py`
示例（Flux）：
```bash
cd experiments/exp3_flux_interaction
./run_flux_probe.sh \
  --backend flux \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image ../exp2_geometry_vis/Section2_exp/data/toothbrush.png \
  --prompt "hold toothbrush" \
  --tokens hold toothbrush \
  --steps 20 --guidance 3.0
```
示例（CLIP）：
```bash
./run_clip_probe.sh \
  --model-id laion/CLIP-ViT-B-16-laion2B-s34B-b88K \
  --image ../exp2_geometry_vis/Section2_exp/data/toothbrush.png \
  --prompts "hold toothbrush" "brush teeth" \
  --feat-source value --layer-index -1 --force-size 224
```
Kontext 辅助脚本在 `FLUX/Flux_Kontext_Interaction/{visualize_flux_kontext_cross_attention.py,warp_heatmap_to_original.py}`。需自备 Flux/SD/CLIP 模型与输入图。

## 实验4：几何 + 交互融合（零样本）
目的：融合 Flux verb/object 热图与 DINO 几何 PCA，生成零样本 affordance 掩码，可选评估。  
入口：`experiments/exp4_fusion_zero_shot/run_pipeline.sh` → `./zero_shot/run_knife_affordance_pipeline.py`
示例：
```bash
cd experiments/exp4_fusion_zero_shot
./run_pipeline.sh \
  --image /path/to/knife.jpg \
  --prompt "Grasp knife" \
  --flux-model /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --output-root outputs_demo
```
组成：
- 阶段代码：`zero_shot/pipeline/{flux_stage,roi_stage,pca_stage,geometry_stage,utils}.py`
- 评估：`zero_shot/eval_agd20k_metrics.py`
- 依赖同级：`dino/`（DINOv3 特征），`Section2_exp/`（ROI/PCA），`FLUX/Flux_Kontext_Interaction/`（Kontext 热图与 warp）
准备：
- DINOv3 权重：在 `dino/configs/local.yaml` 或环境变量 `DINO_CHECKPOINT_PATH`
- Flux Kontext 模型目录（默认 `/home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev`）
- 输入图、可选缓存 `zero_shot/cache/`（未包含）

## 快速检查
```bash
# 查看参数帮助
python experiments/exp3_flux_interaction/section4_probing/cross_attention_probe.py -h
python experiments/exp4_fusion_zero_shot/zero_shot/run_knife_affordance_pipeline.py -h
```
s
> 仓库仅保留代码/配置/README；数据、模型、缓存、输出需按上述路径自行准备。
