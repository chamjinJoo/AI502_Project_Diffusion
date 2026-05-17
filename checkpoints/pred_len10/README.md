# pred_len10 checkpoint candidates

This directory contains curated Transformer checkpoints for the 10-frame future-reference diffusion planner. All checkpoints use the fixed 65D representation:

```text
[joint_pos(29), joint_vel(29), body_quat(4), body_pos(3)]
```

## Files

- `transformer_mdm_style.pt`: recommended checkpoint. This MDM-inspired Transformer uses a dedicated diffusion timestep token plus history/target segment embeddings. It has the best current offline reference-quality metrics.
- `transformer_baseline.pt`: older Transformer baseline, kept for comparison.
- `transformer_velocity_consistent.pt`: Transformer checkpoint trained with a small velocity-consistency auxiliary term. Use this as an A/B rollout candidate when smooth velocity/reference consistency matters.

All checkpoints were selected by validation loss and are intended for GR00T/SONIC tracking-reference experiments.

## Required normalization stats

Use these checkpoints with the committed normalization file:

```text
processed_dataset/stats/stats.json
```

The stats are computed from the training split and contain the 65D z-score mean/std used by training and sampling. They are not generated-output metrics.
