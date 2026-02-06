# 实验3：几何 × 交互融合（AGD20K）

目标：在 AGD20K 上一键评估 Geometry × Interaction 的 training-free 融合结果。

入口脚本（最外层暴露）：
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/exp4_fusion_zero_shot
python ./run_agd20k_eval.py --config ./src/agd20k_eval/config.yaml
```

依赖（已集中到 src/）：
- `src/agd20k_eval/`（评估主逻辑）
- `src/flux_kontext_interaction/`（cross-attention 与 warp 工具）
- `src/pipeline/`（ROI/PCA/几何融合模块）
- `src/dino/`（DINOv3 特征抽取依赖）

数据与模型路径请在 `src/agd20k_eval/config.yaml` 中配置。
