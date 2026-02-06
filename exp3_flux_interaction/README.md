# 实验3：Flux Kontext 交互先验探测（Cross-Attention）

目标：提取 Flux Kontext 跨注意力热图，解析交互先验。

代码来源：`./section4_probing/cross_attention_probe.py`（跨注意力提取）。

最小接口
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/exp3_flux_interaction
python ./section4_probing/cross_attention_probe.py \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image /path/to/toothbrush.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

输出：
- `probe_outputs/attention/<affordance>_heat.png` / `*_overlay.png` / `*_heat.npy`

注意：需要 Flux Kontext 模型权重与输入图。
