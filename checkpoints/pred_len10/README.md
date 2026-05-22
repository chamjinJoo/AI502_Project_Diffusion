# pred_len10 checkpoint

Use `rectified_flow_mdm_root_relative.pt` as the recommended diffusion planner checkpoint.
It is the recommended 10 Hz window-stats baseline: an MDM-inspired Transformer denoiser trained with
`data.root_relative: true`, sampled-window root-relative stats, `pred_len: 10`, `data.fps: 10`, and rectified-flow sampling.
Use `processed_dataset/stats/stats.json` when evaluating or exporting this
checkpoint with older code paths that require an explicit stats file.
