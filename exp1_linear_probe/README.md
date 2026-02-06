# 实验1：线性探针（几何能力）

目标：在 UMD/AGD20K 上对 VFMs 做线性探针，验证几何感知与 affordance 的相关性。

代码位置：`./umd_linear_probing`（原逻辑未改），请在该目录下补齐数据/权重路径。

最小接口（仅训练/评估）
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/exp1_linear_probe
python ./train.py --config ./umd_linear_probing/configs/dinov2.yaml
python ./eval.py  --config ./umd_linear_probing/configs/dinov2.yaml
```

核心脚本/配置：
- 训练/评估：`umd_linear_probing/scripts/train.py`, `umd_linear_probing/scripts/eval.py`
- 配置：`umd_linear_probing/configs/*.yaml`（基础模型）
- 配置说明：`umd_linear_probing/configs/README.md`

注意：请在 YAML 中填写本地数据/权重路径。
