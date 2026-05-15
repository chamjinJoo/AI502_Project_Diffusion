# Conditional DDIM Humanoid Motion Planner

Minimal PyTorch research code for a history-conditioned diffusion planner that predicts future humanoid tracking references directly in GR00T/IsaacLab-compatible motion space.

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
target_fps: 50
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
processed_dataset/
checkpoints/
samples/
exports/
```

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

Build processed `[T, 65]` sequences, train/val manifests, stats, and reports:

```bash
python data_prep/build_dataset.py --config configs/dataset_build.yaml
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

## Diffusion Model Source

The diffusion process uses Hugging Face **diffusers**:

- `diffusers.DDIMScheduler`: https://huggingface.co/docs/diffusers/api/schedulers/ddim

The model is trained with epsilon prediction:

```text
eps_hat = model(xt, timestep, cond)
loss    = MSE(eps_hat, eps)
```

During inference, deterministic DDIM sampling is used with `eta=0`.

This repository provides two denoiser backbones behind the same interface:

```text
model(xt [B, K, 65], cond [B, H, 65], timestep [B]) -> eps_hat [B, K, 65]
```

### 1. Diffusion Policy Style ConditionalUnet1D

The U-Net denoiser in [models/conditional_unet1d.py](models/conditional_unet1d.py) is adapted from the public Diffusion Policy `ConditionalUnet1D` design:

- Diffusion Policy repository: https://github.com/real-stanford/diffusion_policy
- Diffusion Policy paper: https://arxiv.org/abs/2303.04137

This project does not import Diffusion Policy as a dependency. The implementation is simplified and adapted for `[B, K, 65]` humanoid tracking-reference chunks instead of action sequences from the original robotics manipulation setting.

Default config:

```yaml
model:
  architecture: unet
  condition_encoder: transformer
  condition_summary: flatten
  down_dims: [256, 512, 1024]
```

### 2. Transformer Baseline

The Transformer denoiser in [models/denoiser.py](models/denoiser.py) is a small project-local baseline. It uses:

- target-token projection for noisy future chunks
- timestep embedding
- condition history encoder
- temporal Transformer encoder over future tokens

It is selected with:

```yaml
model:
  architecture: transformer
```

The currently useful baseline/checkpoint family is Transformer-based:

```text
configs/transformer_baseline.yaml
checkpoints/transformer_pred_len10_fps120/
```

Auxiliary-loss scratch training for smoother tracking references uses:

```text
configs/transformer_aux_light_scratch.yaml
checkpoints/transformer_pred_len10_fps120_aux_light_scratch/
```

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

Train the default U-Net configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Train the Transformer baseline:

```bash
python scripts/train.py --config configs/transformer_baseline.yaml
```

Train the Transformer model from scratch with light tracking-reference auxiliary losses:

```bash
python scripts/train.py --config configs/transformer_aux_light_scratch.yaml
```

The training script checks processed manifest metadata before training. If `assume_fps_if_missing`, `data.fps`, or the processed dataset changes, rebuild the dataset or refresh any external dataset cache before training.

## Sample

Sample from a condition history:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/transformer_pred_len10_fps120/best.pt \
  --cond samples/transformer_future_reference_check/sample_00_cond_history.npy \
  --num_inference_steps 50 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

Sample with externally supplied initial noise `x_T`:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/transformer_pred_len10_fps120/best.pt \
  --cond samples/transformer_future_reference_check/sample_00_cond_history.npy \
  --x_T samples/transformer_future_reference_check/sample_00_x_T.npy \
  --num_inference_steps 50 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

`--normalize_quat` normalizes the generated `body_quat` channels after sampling. `--reconstruct_velocity --fps 50` can replace predicted joint velocity channels with finite differences of predicted joint positions, but this may amplify noise if the generated joint positions are not smooth.

## GR00T / SONIC Tracking Compatibility

The companion tracking-code directory `AI502TermProject/` is intentionally treated as read-only.

Its diffusion tracking manager expects future motion shaped:

```text
[num_envs, horizon, 65]
```

with the same 65D layout. It converts the first 10 future frames into SONIC tracking commands shaped:

```text
[num_envs, 10, 64] = [joint_pos(29), joint_vel(29), relative_root_rot6(6)]
```

Because this dataset preprocessing stores root orientation as a world/root quaternion, integration should use the converter path that treats diffusion orientation as world-frame and converts it relative to the current root orientation.

## Export CSV

Export a predicted chunk into grouped CSV files:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset/stats/stats.json \
  --output_dir exports/reference \
  --already_denormalized
```

To export velocity from finite differences:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats processed_dataset/stats/stats.json \
  --output_dir exports/reference \
  --already_denormalized \
  --reconstruct_velocity \
  --fps 50
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
