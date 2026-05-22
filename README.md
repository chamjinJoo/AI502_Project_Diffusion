# Conditional Humanoid Motion Planner

Minimal PyTorch research code for a history-conditioned diffusion/flow planner that predicts future humanoid tracking references directly in GR00T/IsaacLab-compatible motion space.

## Representation

Every motion frame is a fixed 65-dimensional vector:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

- `joint_pos`: 29 Unitree G1 / IsaacLab-order joint positions
- `joint_vel`: 29 matching joint velocities
- `body_quat`: root orientation quaternion in `(w, x, y, z)`
- `body_pos`: root position `(x, y, z)`

Each processed training sequence is saved as:

```text
[T, 65]
```

For history length `H=20` and prediction horizon `K=10`, a training sample is:

```text
cond   = sequence[t-H+1 : t+1]   # [20, 65]
target = sequence[t+1   : t+1+K] # [10, 65]
```

The default planner runs at **10 Hz**: `H=20` covers 2 seconds of history and `K=10` predicts a 1-second future goal chunk.

At inference time the diffusion planner takes:

```text
cond  # previous motion history, [H, 65]
x_T   # initial future noise, [K, 65] or [B, K, 65]
```

`x_T` is sampled from standard Gaussian noise by default. It can also be supplied with `--x_T`, which is the intended hook for future steering-policy integration.

## Dataset Source

The current preprocessing pipeline targets **BONES-SEED**, specifically the **Unitree G1 MuJoCo-compatible CSV trajectories**.

Primary sources:

- BONES-SEED Hugging Face dataset: https://huggingface.co/datasets/bones-studio/seed
- BONES-SEED dataset page: https://bones.studio/datasets/seed
- BONES-SEED license page: https://bones.studio/info/seed-license

The source dataset provides Unitree G1-compatible motion CSV paths such as `move_g1_mujoco_path`. This project does not use the source files directly during training. Instead, it converts valid source clips into internal `.npy` arrays under:

```text
processed_dataset/
  sequences/*.npy
  manifests/*.jsonl
  stats/
  reports/
```

The converter inspects actual CSV columns and maps them into the fixed 65D representation:

```text
29 joint DOF columns -> joint_pos
missing joint_vel    -> finite-difference joint_pos when allowed
root Euler rotation  -> body_quat(w, x, y, z)
root translation     -> body_pos
```

Current preprocessing assumptions are in [configs/dataset_build.yaml](configs/dataset_build.yaml):

```yaml
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

Important: root position and root orientation are not silently fabricated. Clips missing required root fields are skipped. Joint velocity may be reconstructed from finite differences because many source CSVs provide joint position but not velocity.

## Data Files

Raw and processed motion data are not committed to git because they are large generated or downloaded artifacts. The repository `.gitignore` excludes:

```text
data/
processed_dataset/*  # except processed_dataset/stats/stats.json
exports/
```

The repository includes one curated root-relative checkpoint, a few example
future-reference samples, and the normalization stats needed for sampling:

```text
checkpoints/pred_len10/rectified_flow_mdm_root_relative.pt
samples/pred_len10/rectified_flow_mdm_root_relative/
processed_dataset/stats/stats.json
```

Full training runs, intermediate checkpoints, raw motion data, processed motion
sequences, and manifests should stay local.

On a new machine, download BONES-SEED separately, place the Unitree G1 MuJoCo-compatible CSV data under the path configured in [configs/dataset_build.yaml](configs/dataset_build.yaml), then run the preprocessing pipeline.

The default expected source path is:

```text
data/raw/bones_seed_g1
```

You may either download from the BONES-SEED Hugging Face dataset page or from the Bones Studio dataset page. If using the Hugging Face CLI, the workflow is:

```bash
mkdir -p data/raw
huggingface-cli download bones-studio/seed \
  --repo-type dataset \
  --local-dir data/raw/bones_seed_g1
```

If the downloaded archive or directory layout differs, keep only the relevant BONES-SEED / Unitree G1 MuJoCo CSV tree under `data/raw/bones_seed_g1`, or update `source_roots` in [configs/dataset_build.yaml](configs/dataset_build.yaml).

## Dataset Build

Inspect source schemas:

```bash
python data_prep/inspect_sources.py --config configs/dataset_build.yaml
```

Build 10 Hz processed `[T, 65]` sequences, train/val manifests, initial stats, and reports:

```bash
python data_prep/build_dataset.py --config configs/dataset_build.yaml
```

Then compute the root-relative sampled-window stats used by the default model:

```bash
python scripts/compute_stats.py \
  --file_list processed_dataset/manifests/train_manifest.jsonl \
  --output processed_dataset/stats/stats.json \
  --history_len 20 \
  --pred_len 10 \
  --frame_dim 65 \
  --root_relative \
  --fps 10 \
  --joint_vel_mode source \
  --body_pos_mode relative
```

Force rebuild after changing FPS/unit assumptions:

```bash
python data_prep/build_dataset.py \
  --config configs/dataset_build.yaml \
  --force_rebuild
```

Run sanity checks:

```bash
python data_prep/sanity_check.py --config configs/dataset_build.yaml
```

Training uses:

```text
processed_dataset/manifests/train_manifest.jsonl
processed_dataset/manifests/val_manifest.jsonl
processed_dataset/stats/stats.json
```

`processed_dataset/stats/stats.json` stores the 10 Hz **sampled-window root-relative** normalization stats used by the current default config. It is computed after slicing training windows and applying the same `data.root_relative: true` transform that `MotionChunkDataset` applies before normalization. It is not a summary of model-generated outputs; the checkpoint expects this file for condition normalization and denormalizing generated chunks.

## Root-Relative Pose Convention

Raw `processed_dataset/sequences/*.npy` files remain in the source/global root-pose coordinate system. For training, `data.root_relative: true` converts each sampled window dynamically inside `MotionChunkDataset` before normalization:

```text
anchor = cond[-1]
body_pos_rel[i]  = R_anchor^-1 * (body_pos[i] - body_pos_anchor)
body_quat_rel[i] = inverse(body_quat_anchor) * body_quat[i]
```

The last history frame becomes approximately identity pose: `body_pos=[0,0,0]`, `body_quat=[1,0,0,0]`.

During inference, `scripts/sample.py` applies the same condition-history conversion when the checkpoint config contains `data.root_relative: true`. Generated future chunks from such checkpoints are current-frame-relative references.

## Diffusion Model Source

The diffusion process uses Hugging Face **diffusers**:

- `diffusers.DDIMScheduler`: https://huggingface.co/docs/diffusers/api/schedulers/ddim

The current default is an MDM-inspired Transformer denoiser trained with a
rectified-flow objective in the same fixed 65D tracking-reference space.

The denoiser still predicts a future chunk-shaped tensor from noisy future
tokens and motion history:

```text
v_hat = model(x_t, timestep, cond)
loss  = MSE(v_hat, x1 - x0)
```

During inference, the default rectified-flow sampler uses Euler integration.

The denoiser interface is fixed as:

```text
model(xt [B, K, 65], cond [B, H, 65], timestep [B]) -> v_hat [B, K, 65]
```

The current recommended denoiser is the MDM-inspired Transformer implementation in [models/denoiser.py](models/denoiser.py). It keeps this project's direct 65D tracking-reference output while borrowing two simple sequence-design ideas from Motion Diffusion Model (MDM): a dedicated diffusion timestep token and explicit token-role separation.

References:

- Motion Diffusion Model repository: https://github.com/GuyTevet/motion-diffusion-model
- Motion Diffusion Model paper: https://arxiv.org/abs/2209.14916

The denoiser uses:

- target-token projection for noisy future chunks
- sinusoidal positional encodings over history and target tokens
- Transformer condition encoder over previous motion history
- joint self-attention over `[timestep token, history tokens, noisy target tokens]`
- segment embeddings that distinguish timestep, history, and target tokens

The default training config is the recommended 10 Hz MDM-style Transformer setup:

```yaml
model:
  architecture: transformer
  dim: 256
  num_layers: 4
  num_heads: 4

data:
  history_len: 20
  pred_len: 10
  fps: 10

training:
  velocity_loss_weight: 0.05
  quaternion_loss_weight: 0.001
  continuity_loss_weight: 0.1
  joint_x0_loss_weight: 0.02
  acceleration_loss_weight: 0.02
  auxiliary_max_timestep: 500

diffusion:
  objective: rectified_flow
  flow_solver: euler
  num_inference_steps: 30
```

The main objective is rectified-flow regression. Auxiliary terms encourage
velocity consistency, unit quaternions, smoothness, and continuity from the last
history frame.

Recommended checkpoint:

```text
checkpoints/pred_len10/rectified_flow_mdm_root_relative.pt
```

This is the recommended 10 Hz `pred_len=10` root-relative rectified-flow MDM-style
Transformer checkpoint. It is trained with sampled-window root-relative stats and
expects `processed_dataset/stats/stats.json` for normalization.

Example GIF visualizations are included under:

```text
samples/pred_len10/rectified_flow_mdm_root_relative/
```

The sample directory intentionally contains GIF visualizations only, so it stays
small enough for git.

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

Train the default 10 Hz MDM-style Transformer configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

`training.checkpoint_dir` may contain date tokens that are expanded at train launch time:

```yaml
checkpoint_dir: checkpoints/rectified_flow_mdm_rootrel_pred_len10_fps10_windowstats_{date}
```

Supported tokens are `{date}` -> `YYYYMMDD` and `{datetime}` -> `YYYYMMDD_HHMMSS`.

The training script checks processed manifest metadata before training. If `assume_fps_if_missing`, `data.fps`, or the processed dataset changes, rebuild the dataset or refresh any external dataset cache before training.

## Sample

Sample from a condition history:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/rectified_flow_mdm_root_relative.pt \
  --cond path/to/cond_history.npy \
  --num_inference_steps 30 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

Sample with externally supplied initial noise `x_T`:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/rectified_flow_mdm_root_relative.pt \
  --cond path/to/cond_history.npy \
  --x_T path/to/initial_noise_x_T.npy \
  --num_inference_steps 30 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

`--normalize_quat` normalizes the generated `body_quat` channels after sampling.
By default, use the model-predicted `joint_vel` channels. `--reconstruct_velocity
--fps 10` can replace them with finite differences of predicted joint positions,
but this may amplify noise if the generated joint positions are not smooth.

Included GIF visualizations from the recommended root-relative checkpoint:

```text
samples/pred_len10/rectified_flow_mdm_root_relative/*.gif
```

## Export CSV

Recommended path: export the model-predicted `joint_vel` channels directly.
This preserves the 65D diffusion output as generated by the model.

Export a predicted chunk into grouped CSV files:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset/stats/stats.json \
  --output_dir exports/reference \
  --already_denormalized
```

Optional diagnostic path: export velocity from finite differences instead of
the model-predicted `joint_vel` channels:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset/stats/stats.json \
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

`body_quat.csv` uses header order:

```text
w, x, y, z
```

## Evaluate

Compare a predicted future chunk with a target future chunk:

```bash
python scripts/evaluate.py \
  --pred samples/predicted_chunk.npy \
  --target samples/target_chunk.npy
```

It reports:

```text
full_mse
joint_pos_mse
joint_vel_mse
quaternion_norm_error
```

## Tests

Run smoke tests:

```bash
python tests/run_smoke_tests.py
```

The tests use synthetic arrays and do not require BONES-SEED.
