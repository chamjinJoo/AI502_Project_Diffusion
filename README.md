# Conditional Humanoid Motion Planner

Minimal PyTorch research code for a history-conditioned motion planner that predicts future humanoid tracking references directly in GR00T/IsaacLab-compatible motion space. This is not a text-to-motion model.

## Representation

Every frame is a fixed 65-dimensional vector:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

- `joint_pos`: 29 Unitree G1 / IsaacLab-order joint positions
- `joint_vel`: 29 matching joint velocities
- `body_quat`: root orientation quaternion in `(w, x, y, z)`
- `body_pos`: root position channels

For the current default model, processed sequences are sampled at **10 Hz**. With `H=20` and `K=10`, the model conditions on 2 seconds of history and predicts a 1-second future reference chunk:

```text
cond   = sequence[t-H+1 : t+1]   # [20, 65]
target = sequence[t+1   : t+1+K] # [10, 65]
```

At inference time the planner takes `cond [H, 65]` plus initial future noise `x_T [K, 65]` or `[B, K, 65]`. If `x_T` is omitted, it is sampled from a standard Gaussian. Supplying `x_T` is the hook for future steering-policy integration.

## Dataset Source

The preprocessing pipeline targets **BONES-SEED**, specifically the **Unitree G1 MuJoCo-compatible CSV trajectories**. The default build filters to locomotion-style clips.

Primary sources:

- BONES-SEED Hugging Face dataset: https://huggingface.co/datasets/bones-studio/seed
- BONES-SEED dataset page: https://bones.studio/datasets/seed
- BONES-SEED license page: https://bones.studio/info/seed-license

The source CSV files are converted into internal `.npy` arrays with shape `[T, 65]`. Source root pose is kept canonical/global in the processed sequence files; root-relative and delta-space transforms are applied when training windows are sampled.

Current preprocessing assumptions are in [configs/dataset_build.yaml](configs/dataset_build.yaml):

```yaml
output_root: processed_dataset_10hz
target_fps: 10
assume_fps_if_missing: 120
source_joint_unit: degrees
source_root_pos_unit: centimeters
source_root_euler_unit: degrees
dataset_category: locomotion
allow_joint_vel_fallback: true
allow_body_pos_zero_fallback: false
allow_body_quat_identity_debug_fallback: false
```

Root position and root orientation are not silently fabricated. Clips missing required root fields are skipped. Joint velocity may be reconstructed from finite differences because many source CSVs provide joint position but not velocity.

## Data Files

Raw and processed motion data are mostly excluded from git because they are large downloaded/generated artifacts. The repository includes only the small files needed to run the curated model:

```text
checkpoints/pred_len10/(model_name).pt
samples/pred_len10/(model_name)/*.gif
processed_dataset_10hz/stats/root_relative_delta_window_stats.json
```

The checkpoint also embeds its normalization stats, so recent sampling and evaluation scripts can usually load stats directly from the checkpoint. The stats JSON is included for compatibility and inspection.

## Dataset Build

Place BONES-SEED Unitree G1 MuJoCo-compatible CSV data under:

```text
data/raw/bones_seed_g1
```

One possible download route is the Hugging Face CLI:

```bash
mkdir -p data/raw
huggingface-cli download bones-studio/seed \
  --repo-type dataset \
  --local-dir data/raw/bones_seed_g1
```

If the downloaded layout differs, keep the relevant BONES-SEED / Unitree G1 MuJoCo CSV tree under `data/raw/bones_seed_g1`, or edit `source_roots` in [configs/dataset_build_10hz.yaml](configs/dataset_build_10hz.yaml).

Inspect source schemas:

```bash
python data_prep/inspect_sources.py --config configs/datasedataset_build_10hzt_build.yaml
```

Build 10 Hz processed sequences, manifests, initial raw stats, and reports:

```bash
python data_prep/build_dataset.py --config configs/dataset_build_10hz.yaml
```

Compute the sampled-window stats used by the default model:

```bash
python scripts/compute_stats.py \
  --file_list processed_dataset_10hz/manifests/train_manifest.jsonl \
  --output processed_dataset_10hz/stats/root_relative_delta_window_stats.json \
  --history_len 20 \
  --pred_len 10 \
  --frame_dim 65 \
  --root_relative \
  --fps 10 \
  --joint_vel_mode source \
  --body_pos_mode delta
```

Run sanity checks:

```bash
python data_prep/sanity_check.py --config configs/dataset_build_10hz.yaml
```

Training uses:

```text
processed_dataset_10hz/manifests/train_manifest.jsonl
processed_dataset_10hz/manifests/val_manifest.jsonl
processed_dataset_10hz/stats/root_relative_delta_window_stats.json
```

`root_relative_delta_window_stats.json` stores normalization stats computed in the actual training space: sampled windows after root-relative conversion and `body_pos_mode: delta`. It is not a summary of model-generated outputs. It is used to normalize condition/target chunks and denormalize generated chunks.

## Root-Relative And Delta Pose Convention

The raw processed `.npy` sequences remain source/global root-pose trajectories. During dataset sampling, `data.root_relative: true` anchors each window at the last history frame:

```text
anchor = cond[-1]
body_pos_rel[i]  = R_anchor^-1 * (body_pos[i] - body_pos_anchor)
body_quat_rel[i] = inverse(body_quat_anchor) * body_quat[i]
```

The default model then uses `data.body_pos_mode: delta`, so body position channels are represented as per-frame displacement rather than absolute global XY. This keeps locomotion references better centered for 10 Hz tracking-goal generation.

During inference, `scripts/sample.py` applies the same condition-history conversion when the checkpoint config contains these data settings. Generated future chunks from the recommended checkpoint are current-frame-relative, delta-style future references.

## Model

The current default is an MDM-inspired Transformer denoiser trained with a rectified-flow objective. It keeps the direct 65D tracking-reference output and does not use text, SMPL, or robotics middleware.

The denoiser interface is:

```text
model(xt [B, K, 65], timestep [B], cond [B, H, 65]) -> v_hat [B, K, 65]
```

References:

- Motion Diffusion Model repository: https://github.com/GuyTevet/motion-diffusion-model
- Motion Diffusion Model paper: https://arxiv.org/abs/2209.14916
- Hugging Face DDIMScheduler docs: https://huggingface.co/docs/diffusers/api/schedulers/ddim

The model uses target-token projection, sinusoidal positional encodings, a Transformer condition encoder over motion history, joint attention over `[timestep token, history tokens, noisy target tokens]`, and segment embeddings for token roles.

The default config is [configs/default.yaml](configs/default.yaml):

```yaml
data:
  fps: 10
  root_relative: true
  joint_vel_mode: source
  body_pos_mode: delta
  history_len: 20
  pred_len: 10

model:
  architecture: transformer
  dim: 256
  num_layers: 4
  num_heads: 4
  condition_encoder: transformer
  condition_summary: flatten

diffusion:
  objective: rectified_flow
  flow_solver: euler
  num_inference_steps: 30
```

Auxiliary losses encourage velocity consistency, unit quaternions, smoothness, and continuity from the last history frame.

## Recommended Checkpoint And Samples

Recommended checkpoint:

```text
checkpoints/pred_len10/rectified_flow_512_12layer_8head.pt
```

This file currently contains a hevier model than default setting: a 10 Hz `pred_len=10` root-relative/delta rectified-flow MDM-style Transformer checkpoint with 512 dim, 12-depth layer, and total 8 multi-head.

Additional low-auxiliary-loss candidate:

Example GIF visualizations are included under:

```text
samples/pred_len10/(model_name)/*.gif
```

The sample directory intentionally contains GIF files only, so it stays small enough for git.

## Requirements

The code has been run with:

```text
Python 3.11.15
PyTorch 2.5.1
NumPy 2.4.4
diffusers 0.38.0
```

Install an equivalent Python environment with PyTorch, NumPy, PyYAML, and Hugging Face diffusers.

## Train

Train the default 10 Hz configuration:

```bash
python scripts/train.py --config configs/default_10hz.yaml
```

`training.checkpoint_dir` may contain date tokens expanded at train launch time:

```yaml
checkpoint_dir: checkpoints/rectified_flow_mdm_rootrel_delta_pred_len10_fps10_windowstats_{date}
```

Supported tokens are `{date}` -> `YYYYMMDD` and `{datetime}` -> `YYYYMMDD_HHMMSS`.

The training script checks processed manifest and stats metadata before training. If FPS, root-relative mode, body-position mode, or dataset assumptions change, rebuild the dataset and recompute training-window stats before training.

## Sample

Sample from a condition history:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/(model_name).pt \
  --cond path/to/cond_history.npy \
  --num_inference_steps 30 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

Sample with externally supplied initial noise `x_T`:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/(model_name).pt \
  --cond path/to/cond_history.npy \
  --x_T path/to/initial_noise_x_T.npy \
  --num_inference_steps 30 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

`--normalize_quat` normalizes generated `body_quat` channels after sampling. By default, use model-predicted `joint_vel`. `--reconstruct_velocity --fps 10` is only a diagnostic option and may amplify noise when predicted joint positions are not smooth.

## Export CSV

Export the model-predicted 65D future chunk into grouped CSV files:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset_10hz/stats/root_relative_delta_window_stats.json \
  --output_dir exports/reference \
  --already_denormalized
```

Optional diagnostic velocity reconstruction:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset_10hz/stats/root_relative_delta_window_stats.json \
  --output_dir exports/reference \
  --already_denormalized \
  --reconstruct_velocity \
  --fps 10
```

This writes:

```text
joint_pos.csv
joint_vel.csv
body_quat.csv
body_pos.csv
```

`body_quat.csv` uses header order `w, x, y, z`.

## Evaluate And Visualize

Run a compact post-training check from a checkpoint:

```bash
python scripts/post_training_eval.py \
  --checkpoint checkpoints/pred_len10/(model_name).pt \
  --output_dir samples/post_training_eval \
  --num_eval 256 \
  --num_visual 18 \
  --num_inference_steps 30
```

This writes an evaluation summary, generated chunks, target chunks, and GIF visualizations under `samples/post_training_eval/`.

For a direct chunk-to-chunk metric comparison:

```bash
python scripts/evaluate.py \
  --pred samples/predicted_chunk.npy \
  --target samples/target_chunk.npy
```

## Tests

Run smoke tests:

```bash
python tests/run_smoke_tests.py
```

The tests use synthetic arrays and do not require BONES-SEED.
