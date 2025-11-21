# 实验1：线性探针（几何能力）

目标：在 UMD/AGD20K 上对 VFMs 做线性探针，验证几何感知与 affordance 的相关性。

代码位置：`./umd_linear_probing`（原逻辑未改），请在该目录下补齐数据/权重路径。

快捷运行
```bash
cd /home/li325/qing_workspace/probing_briding_affordance/experiments/exp1_linear_probe
./run.sh --config ./umd_linear_probing/configs/single_layer/dinov2/8_both.yaml
```

核心脚本/配置：
- 训练/评估：`umd_linear_probing/scripts/train.py`, `scripts/eval.py`
- 配置：`umd_linear_probing/configs/*.yaml`（模型）、`configs/single_layer/*`（分层）
- 可视化：`umd_linear_probing/scripts/visualize_results.py`

注意：`checkpoint_weights` 已移除，自行下载并在 `configs/local*.yaml` 填路径。
