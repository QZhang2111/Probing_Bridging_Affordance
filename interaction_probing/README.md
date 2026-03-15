# Interaction Probing

FLUX Kontext cross-attention probing for affordance-specific prompts.

## Recommended Entry

```bash
cd /path/to/Probing_Briding_Affordance
python run.py interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance hold \
  --steps 20 \
  --guidance 3.0
```

## Direct Entry

```bash
cd /path/to/Probing_Briding_Affordance/interaction_probing
python ./cross_attention_probe/cross_attention_probe.py \
  --model-id ../models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance hold \
  --steps 20 \
  --guidance 3.0
```

## Outputs

- `probe_outputs/attention/<affordance>_heat.png`
- `probe_outputs/attention/<affordance>_overlay.png`
- `probe_outputs/attention/<affordance>_heat.npy`
