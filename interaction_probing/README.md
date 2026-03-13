# Main Experiment 2: Interaction Probing (Flux Cross-Attention)

Goal: extract and visualize interaction priors via Flux Kontext cross-attention maps.

Core script: `cross_attention_probe/cross_attention_probe.py`.

## Recommended Entry (Unified)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance
python run.py interaction-probe -- \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image /path/to/toothbrush.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

## Direct Entry (Backward Compatible)
```bash
cd /home/li325/qing_workspace/Probing_Briding_Affordance/interaction_probing
python ./cross_attention_probe/cross_attention_probe.py \
  --model-id /home/li325/qing_workspace/model_for_test/FLUX.1-Kontext-dev \
  --image /path/to/toothbrush.png \
  --prompt "hold toothbrush" \
  --affordance "hold" \
  --steps 20 --guidance 3.0
```

## Outputs
- `probe_outputs/attention/<affordance>_heat.png`
- `probe_outputs/attention/<affordance>_overlay.png`
- `probe_outputs/attention/<affordance>_heat.npy`

Requires local Flux Kontext weights and input image files.
