# pred_len10 checkpoint

Use `transformer_mdm_root_relative.pt` as the recommended diffusion planner checkpoint.
It is an MDM-inspired Transformer denoiser trained with `data.root_relative: true` and `pred_len: 10`.
The checkpoint expects normalization stats at `processed_dataset/stats/stats.json`.
