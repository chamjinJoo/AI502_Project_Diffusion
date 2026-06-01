# pred_len10 checkpoints

Primary checkpoint:

```text
rectified_flow_512_12layer_8head
.pt
```

This is the 10 Hz window-stats model: an MDM-inspired Transformer denoiser trained with
`data.root_relative: true`, `data.body_pos_mode: delta`, sampled-window root-relative/delta stats,
`pred_len: 10`, `data.fps: 10`, and rectified-flow sampling. Details are in `configs/heavy.yaml`.

The matching explicit stats file for old code paths is:

```text
processed_dataset_10hz/stats/root_relative_delta_window_stats.json
```

Checkpoints embed normalization stats, so most sampling/evaluation scripts can load the stats directly from the checkpoint.

Additional smaller-model checkpoint:

```text
rectified_flow_mdm_root_relative.pt
```

This is the model with dim=256, layers=4, and heads=4.
