# pred_len10 sample outputs

These `.npy` files are denormalized future-reference chunks sampled from
`checkpoints/pred_len10/best.pt` with DDIM 50 steps.

Each sample has shape `[10, 65]` and keeps the model-predicted `joint_vel`
channels. The repository recommendation is to use these predicted velocities by
default rather than replacing them with finite-difference velocities.

`evaluation_summary.json` contains the lightweight reference-quality checks for
the curated predicted-velocity samples.
