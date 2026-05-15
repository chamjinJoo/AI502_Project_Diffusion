# pred_len10 recommended checkpoint

`best.pt` is the curated job 27 best checkpoint for the default Transformer
conditional DDIM planner.

- representation: `[joint_pos(29), joint_vel(29), body_quat(4), body_pos(3)]`
- prediction horizon: `K=10`
- history length: `H=20`
- checkpoint epoch: 4034
- validation loss: 0.08243497461080551
- source checkpoint: `checkpoints/transformer_pred_len10_fps120_20260515/best.pt`

Use this checkpoint with `scripts/sample.py` unless you are intentionally
training or evaluating a new model.
