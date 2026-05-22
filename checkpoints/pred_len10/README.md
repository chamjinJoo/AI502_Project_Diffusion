# pred_len10 checkpoint

Use `rectified_flow_mdm_root_relative.pt` as the recommended diffusion planner checkpoint.
It is the recommended 10 Hz window-stats model: an MDM-inspired Transformer denoiser trained with
`data.root_relative: true`, `data.body_pos_mode: delta`, sampled-window root-relative/delta stats,
`pred_len: 10`, `data.fps: 10`, and rectified-flow sampling.

The matching explicit stats file for old code paths is:

```text
processed_dataset_10hz/stats/root_relative_delta_window_stats.json
```

New checkpoints embed normalization stats, so most sampling/evaluation scripts can load the stats directly from the checkpoint.
