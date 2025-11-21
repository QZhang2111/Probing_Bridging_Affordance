# 实验4：几何 + 交互融合（零样本流水线）

目标：融合 Flux verb/object 热图与 DINO 几何 PCA，得到零样本 affordance 掩码；可选评估。

代码来源：`./zero_shot`（流水线），依赖同级的 `./FLUX/Flux_Kontext_Interaction`（Kontext 热图/warp）、`./dino`（DINOv3 token），`./Section2_exp`（ROI/PCA 辅助）。逻辑未改。

快捷运行
```bash
cd /home/li325/qing_workspace/probing_briding_affordance/experiments/exp4_fusion_zero_shot
./run_pipeline.sh \
  --image /path/to/knife.jpg \
  --prompt "Grasp knife" \
  --flux-model /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --output-root outputs_demo
```

核心脚本（原路径）：
- 入口：`zero_shot/run_knife_affordance_pipeline.py`
- 阶段：`zero_shot/pipeline/{flux_stage,roi_stage,pca_stage,geometry_stage,utils}.py`
- 评估：`zero_shot/eval_agd20k_metrics.py`

注意：
- 配置 DINOv3 权重：在 `dino/configs/local.yaml` 或环境变量 `DINO_CHECKPOINT_PATH`
- `zero_shot/cache/`、输入图未包含；Flux 模型目录默认 `/home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev`
