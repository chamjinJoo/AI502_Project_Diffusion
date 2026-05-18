# pred_len10 sample outputs

This directory contains denormalized future-reference chunks sampled from the
curated `pred_len=10` Transformer checkpoint candidates with DDIM 50 steps.

Each `.npy` sample has shape `[10, 65]` and keeps the model-predicted
`joint_vel` channels. This is the recommended export path for now.

## Layout

```text
transformer_baseline/
  sample_00_future.npy
  sample_01_future.npy
  sample_02_future.npy
  sample_03_future.npy

transformer_velocity_consistent/
  sample_00_future.npy
  sample_01_future.npy
  sample_02_future.npy
  sample_03_future.npy

candidate_comparison.json
```

`candidate_comparison.json` contains a compact offline validation comparison
using shared validation windows and shared initial DDIM noise. It is a reference
quality check, not a substitute for a real GR00T/SONIC rollout A/B test.
