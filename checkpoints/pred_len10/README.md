# pred_len10 checkpoints

Primary checkpoint:

```text
rectified_flow_mdm_root_relative.pt
```

This is the default 10 Hz window-stats model: an MDM-inspired Transformer denoiser trained with
`data.root_relative: true`, `data.body_pos_mode: delta`, sampled-window root-relative/delta stats,
`pred_len: 10`, `data.fps: 10`, and rectified-flow sampling.

Additional candidate checkpoint:

```text
rectified_flow_mdm_root_relative_low_aux.pt
```

This is the job 863 low-auxiliary-loss candidate. It keeps the same data/model/diffusion setup as the
primary checkpoint, but uses smaller auxiliary loss coefficients:
`velocity=0.02`, `quaternion=0.0005`, `continuity=0.03`, `joint_x0=0.01`, `acceleration=0.01`, and
`sample_eval_weight=0.01`.

The matching explicit stats file for old code paths is:

```text
processed_dataset_10hz/stats/root_relative_delta_window_stats.json
```

New checkpoints embed normalization stats, so most sampling/evaluation scripts can load the stats directly from the checkpoint.

Additional smaller-model checkpoint:

```text
rectified_flow_mdm_root_relative_dim128.pt
```

This is the job 1272 dim=128 candidate. It uses the same data/diffusion/training setup as the primary
checkpoint but with `model.dim: 128` (half the hidden size) instead of 256.
