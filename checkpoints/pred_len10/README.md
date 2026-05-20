# pred_len10 checkpoint

Use `rectified_flow_mdm_root_relative.pt` as the recommended diffusion planner checkpoint.
It is the job 103 setting: an MDM-inspired Transformer denoiser trained with
`data.root_relative: true`, `pred_len: 10`, and rectified-flow sampling.
The checkpoint expects normalization stats at `processed_dataset/stats/stats.json`.
