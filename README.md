# Conditional DDIM Planner

Minimal PyTorch research code for a history-conditioned DDIM planner that predicts humanoid future motion chunks directly in GR00T tracking-reference space. The diffusion core uses Hugging Face `diffusers.DDIMScheduler` with a Diffusion Policy style conditional 1D U-Net denoiser.

This is not a text-to-motion model. It does not use language encoders, SMPL, action decoders, or robotics framework dependencies.


## Requirements

The code has been run with:

```text
Python 3.11.15
PyTorch 2.5.1
NumPy 2.4.4
diffusers 0.38.0
```

Install an equivalent Python environment with PyTorch, NumPy, PyYAML, and
Hugging Face diffusers available.

## Representation

Each frame is a 65-dimensional vector:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

`body_quat` is ordered as `(w, x, y, z)`. Each `.npy` training sequence must have shape `[T, 65]`.

For history length `H=20` and prediction horizon `K=10`, samples are:

```text
cond   = sequence[t-H+1 : t+1]   # [H, 65]
target = sequence[t+1   : t+1+K] # [K, 65]
```

The diffusion model has two runtime inputs:

```text
cond  # previous motion history, [H, 65]
x_T   # initial future-chunk noise, [K, 65] or [B, K, 65]
```

`cond` is the last `H` tracking-reference frames. Each condition frame uses the same 65-dimensional layout:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

At inference time, `scripts/sample.py` accepts either an exact `[H, 65]`
condition array or a longer `[T, 65]` sequence. If a longer sequence is given,
the script uses the last `H` frames as the condition. This history is normalized
with the training statistics before being passed to the model.

`x_T` is the noisy initial future chunk for DDIM sampling. By default it is
sampled from standard Gaussian noise with shape `[K, 65]` per sample. It may
also be supplied explicitly with `--x_T`, which is the hook for later steering
policies. The model then denoises this latent into a future chunk:

```text
output = predicted future motion chunk  # [K, 65]
```

The default denoiser is configured with:

```yaml
model:
  architecture: unet
  down_dims: [256, 512, 1024]
```

This keeps the same project interface as the earlier Transformer baseline:
`model(xt, cond, timestep) -> eps_hat`, all shaped in normalized `[B, K, 65]`
diffusion space. Older checkpoints that do not contain `model.architecture`
are still treated as Transformer-denoiser checkpoints by the sampling script.


## Dataset

This project currently uses BONES-SEED as the source motion corpus, specifically
the Unitree G1 MuJoCo-compatible CSV trajectories.

Sources:

- Hugging Face dataset: https://huggingface.co/datasets/bones-studio/seed
- Bones Studio dataset page: https://bones.studio/datasets/seed

The local preprocessing pipeline converts source CSV clips into training-ready
internal arrays under `processed_dataset/`. Each processed sequence is stored as
one `.npy` file shaped `[T, 65]` using this fixed layout:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

For the BONES-SEED G1 CSV files used here, the observed source fields include
root translation, root Euler rotation, and 29 joint DOF columns. The converter
maps them as follows:

```text
29 joint DOF columns       -> joint_pos, converted to radians
finite-difference position -> joint_vel, when velocity columns are absent
root Euler rotation        -> body_quat in (w, x, y, z)
root translation           -> body_pos
```

The current processed dataset contains:

```text
processed_dataset/sequences/*.npy              61106 clips
processed_dataset/manifests/train_manifest.jsonl 54995 clips
processed_dataset/manifests/val_manifest.jsonl    6111 clips
processed_dataset/stats/stats.json             normalization stats
processed_dataset/reports/                     preprocessing audit reports
```

The dataset builder is strict by default: clips missing required root quaternion,
root position, or fps metadata are skipped rather than silently filled with
fabricated values. Joint velocity may be reconstructed from finite differences
when configured, and skip reasons are written to
`processed_dataset/reports/conversion_report.json`.

## Train

Scan a directory for valid `[T, 65]` files:

```bash
python scripts/preprocess_dataset.py --data_dir data/raw --output data/motion_files.json
```

Compute normalization stats before training:

```bash
python scripts/compute_stats.py \
  --file_list data/motion_files.json \
  --output checkpoints/normalization_stats.json
```

Edit `configs/default.yaml` so `data.train_paths` points to your `.npy` files, then run:

```bash
python scripts/train.py --config configs/default.yaml
```

With the default config, checkpoints are written to
`checkpoints/unet_pred_len10/latest.pt` and
`checkpoints/unet_pred_len10/best.pt`. Normalization stats are loaded from
`data.stats_path`.

Instead of listing every file in YAML, you can set:

```yaml
data:
  train_file_list: data/motion_files.json
  val_file_list:
```

Or recursively scan a data directory at training startup:

```yaml
data:
  train_data_dir: data/raw
  val_data_dir:
```

## Sample

Use a condition history `.npy` shaped `[H, 65]` or a longer `[T, 65]` sequence:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/unet_pred_len10/best.pt \
  --cond data/example_cond.npy \
  --num_inference_steps 20 \
  --output samples/predicted_chunk.npy
```

The DDIM sampler starts from Gaussian `x_T` by default. You can replace that later with a steering policy by passing externally supplied initial noise:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/unet_pred_len10/best.pt \
  --cond data/example_cond.npy \
  --x_T data/steered_noise.npy \
  --output samples/predicted_chunk.npy
```

The externally supplied `x_T` should be shaped `[K, 65]` for one sample or
`[B, K, 65]` for batched sampling. It lives in normalized diffusion space, not
denormalized tracking-reference units. Use `--denormalize` when you want the
saved prediction converted back to the original motion units.


### Concrete Input/Output Example

A small reproducible example is saved under `samples/pred_len10_example/`. It
uses the `pred_len=10` checkpoint:

```text
checkpoint: checkpoints/pred_len10/latest.pt
source:     processed_dataset/sequences/motion_014151.npy
```

The example input files are:

```text
samples/pred_len10_example/cond_history.npy       # [20, 65]
samples/pred_len10_example/initial_noise_x_T.npy  # [10, 65]
samples/pred_len10_example/target_future.npy      # [10, 65], held-out reference
```

`cond_history.npy` is the previous 20 tracking-reference frames. Each frame is:

```text
[ joint_pos(29), joint_vel(29), body_quat(4), body_pos(3) ]
```

`initial_noise_x_T.npy` is the initial DDIM latent/noise chunk. In this example
it has approximately standard Gaussian statistics:

```text
mean = -0.0110
std  =  0.9859
```

Run the example with externally supplied `x_T`:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/latest.pt \
  --cond samples/pred_len10_example/cond_history.npy \
  --x_T samples/pred_len10_example/initial_noise_x_T.npy \
  --num_inference_steps 20 \
  --denormalize \
  --output samples/pred_len10_example/predicted_future_latest.npy
```

This produces:

```text
samples/pred_len10_example/predicted_future_latest.npy  # [10, 65]
samples/pred_len10_example/summary.json                 # shape/stat summary
samples/pred_len10_example/csv/                         # exported CSV preview
```

For the saved example, the first predicted frame starts with:

```text
joint_pos[0:8] = [0.076020, 0.172944, 0.261720, 0.147468,
                  -0.167161, -0.135897, 0.095730, -0.154970]
body_quat     = [0.676549, -0.015048, -0.014110, -0.752402]
body_pos      = [-0.015890, 0.028918, 0.779900]
```

Compared with the held-out `target_future.npy`, this single example gives:

```text
full_mse:              0.00025365
joint_pos_mse:         0.00002620
joint_vel_mse:         0.00051899
quaternion_norm_error: 0.01255170
```

## Export CSV

Predictions are exported without changing representation:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats checkpoints/normalization_stats.json \
  --output_dir exports/reference
```

To export velocity from finite differences of `joint_pos` instead of the predicted velocity channels:

```bash
python scripts/export_reference.py \
  --chunk samples/predicted_chunk.npy \
  --stats checkpoints/normalization_stats.json \
  --output_dir exports/reference \
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

Use `--already_denormalized` if the chunk has already been converted back to original tracking-reference units.

## Evaluate

```bash
python scripts/evaluate.py --pred samples/predicted_chunk.npy --target data/target_chunk.npy
```

Reports full future chunk MSE, joint position MSE, joint velocity MSE, and quaternion norm error.

## Tests

Run the no-dependency smoke runner:

```bash
python tests/run_smoke_tests.py
```
