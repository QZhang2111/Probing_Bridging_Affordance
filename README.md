# Probing & Bridging Affordance Experiments

最小可运行版本，仅保留三条主线：
- **Geometry Probing（UMD）**
- **Interaction Probing（Flux cross-attention）**
- **Geometry × Interaction Fusion（AGD20K）**

默认环境：`conda activate diffDINO`。数据/模型/缓存路径需自行配置。

目录总览
- `exp1_linear_probe`：几何线性探针（UMD/AGD20K）
- `exp3_flux_interaction`：Flux Kontext 跨注意力提取（交互先验）
- `exp4_fusion_zero_shot`：几何 + 交互融合零样本评估（AGD20K）

## 实验1：线性探针
入口：`exp1_linear_probe/train.py` / `exp1_linear_probe/eval.py`
```bash
cd exp1_linear_probe
python ./train.py --config ./umd_linear_probing/configs/dinov2.yaml
python ./eval.py  --config ./umd_linear_probing/configs/dinov2.yaml
```

## 实验2：交互先验探测（Flux Kontext）
入口：`exp3_flux_interaction/section4_probing/cross_attention_probe.py`
```bash
cd exp3_flux_interaction
python ./section4_probing/cross_attention_probe.py \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image /path/to/toothbrush.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

## 实验3：几何 + 交互融合（AGD20K）
入口：`exp4_fusion_zero_shot/run_agd20k_eval.py`
```bash
cd exp4_fusion_zero_shot
python ./run_agd20k_eval.py --config ./src/agd20k_eval/config.yaml
```

> 仓库仅保留代码/配置/README；数据、模型、缓存、输出需按上述路径自行准备。
