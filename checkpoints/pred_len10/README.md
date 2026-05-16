# pred_len10 checkpoint candidates

This directory contains two curated Transformer checkpoints for the 10-frame
future-reference diffusion planner. Both use the fixed 65D representation:

```text
[joint_pos(29), joint_vel(29), body_quat(4), body_pos(3)]
```

## Files

- `transformer_baseline.pt`: default `pred_len=10` Transformer checkpoint. Start here for practical sampling.
- `transformer_velocity_consistent.pt`: Transformer checkpoint trained with a small velocity-consistency auxiliary term. Use this as an A/B rollout candidate when smooth velocity/reference consistency matters.

Both checkpoints were selected by validation loss and are intended for GR00T/SONIC tracking-reference experiments.
