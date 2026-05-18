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
processed_dataset/*  # except processed_dataset/stats/stats.json
exports/
```

The repository does include a small curated set of candidate checkpoints,
example future-reference samples, and the training normalization stats needed
for sampling:

```text
checkpoints/pred_len10/
samples/pred_len10/
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

`processed_dataset/stats/stats.json` stores the per-dimension training-set
normalization mean/std for the fixed 65D representation. It is not a summary of
model-generated outputs. The released checkpoints expect this file for condition
normalization during sampling and for denormalizing generated chunks.

## Diffusion Model Source

The diffusion process uses Hugging Face **diffusers**:

- `diffusers.DDIMScheduler`: https://huggingface.co/docs/diffusers/api/schedulers/ddim

The model is trained with epsilon prediction:

```text
eps_hat = model(xt, timestep, cond)
loss    = MSE(eps_hat, eps)
```

During inference, deterministic DDIM sampling is used with `eta=0`.

The denoiser interface is fixed as:

```text
model(xt [B, K, 65], cond [B, H, 65], timestep [B]) -> eps_hat [B, K, 65]
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

The default training config follows the current MDM-style Transformer setup:

```yaml
model:
  architecture: transformer
  dim: 256
  num_layers: 4
  num_heads: 4

data:
  history_len: 20
  pred_len: 10
  fps: 50

training:
  velocity_loss_weight: 0.0000001
  quaternion_loss_weight: 0.00001
  continuity_loss_weight: 0.0001
  auxiliary_max_timestep: 200
```

The auxiliary losses are deliberately small. The main objective remains epsilon-prediction MSE, while the auxiliary terms lightly encourage velocity consistency, unit quaternions, and continuity from the last history frame.

Curated Transformer checkpoint candidates:

```text
checkpoints/pred_len10/transformer_mdm_style.pt
checkpoints/pred_len10/transformer_baseline.pt
checkpoints/pred_len10/transformer_velocity_consistent.pt
```

`transformer_mdm_style.pt` is the recommended `pred_len=10` checkpoint. It uses the MDM-inspired timestep token and segment embeddings and gave the best offline reference-quality metrics among the current candidates.

`transformer_baseline.pt` is the older Transformer baseline kept for comparison.

`transformer_velocity_consistent.pt` was trained with a small velocity
consistency auxiliary term. It is useful as an A/B rollout candidate when smooth velocity/reference consistency matters.

Example generated chunks are included under:

```text
samples/pred_len10/transformer_mdm_style/
samples/pred_len10/transformer_baseline/
samples/pred_len10/transformer_velocity_consistent/
samples/pred_len10/candidate_comparison.json
```

These samples keep the model-predicted `joint_vel` channels. That is the
recommended path for now; finite-difference velocity reconstruction is still
available as an export option, but it is not the default recommendation.

### Diffusion Policy Style ConditionalUnet1D

The repository also contains a Diffusion Policy style U-Net in [models/conditional_unet1d.py](models/conditional_unet1d.py), adapted from the public `ConditionalUnet1D` design:

- Diffusion Policy repository: https://github.com/real-stanford/diffusion_policy
- Diffusion Policy paper: https://arxiv.org/abs/2303.04137

This code path is retained for experiments and has been updated toward Diffusion Policy style global/local conditioning. It is not the current default. In our BONES-SEED locomotion setting, the MDM-style Transformer remains the stronger candidate so far.

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

Train the default MDM-style Transformer configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

The same configuration is also saved explicitly as:

```bash
python scripts/train.py --config configs/transformer_mdm.yaml
```

The experimental Diffusion Policy style UNet configuration is:

```bash
python scripts/train.py --config configs/unet_diffusion_policy.yaml
```

`training.checkpoint_dir` may contain date tokens that are expanded at train launch time:

```yaml
checkpoint_dir: checkpoints/transformer_mdm_pred_len10_fps120_{date}
```

Supported tokens are `{date}` -> `YYYYMMDD` and `{datetime}` -> `YYYYMMDD_HHMMSS`.

The training script checks processed manifest metadata before training. If `assume_fps_if_missing`, `data.fps`, or the processed dataset changes, rebuild the dataset or refresh any external dataset cache before training.

## Sample

Sample from a condition history:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/transformer_mdm_style.pt \
  --cond path/to/cond_history.npy \
  --num_inference_steps 50 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

Sample with externally supplied initial noise `x_T`:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/pred_len10/transformer_mdm_style.pt \
  --cond path/to/cond_history.npy \
  --x_T path/to/initial_noise_x_T.npy \
  --num_inference_steps 50 \
  --denormalize \
  --normalize_quat \
  --output samples/predicted_chunk.npy
```

`--normalize_quat` normalizes the generated `body_quat` channels after sampling.
By default, use the model-predicted `joint_vel` channels. `--reconstruct_velocity
--fps 50` can replace them with finite differences of predicted joint positions,
but this may amplify noise if the generated joint positions are not smooth.

Included sample outputs from the candidate checkpoints:

```text
samples/pred_len10/transformer_mdm_style/sample_00_future.npy
samples/pred_len10/transformer_mdm_style/sample_01_future.npy
samples/pred_len10/transformer_mdm_style/sample_02_future.npy
samples/pred_len10/transformer_mdm_style/sample_03_future.npy
samples/pred_len10/transformer_baseline/sample_00_future.npy
samples/pred_len10/transformer_baseline/sample_01_future.npy
samples/pred_len10/transformer_baseline/sample_02_future.npy
samples/pred_len10/transformer_baseline/sample_03_future.npy
samples/pred_len10/transformer_velocity_consistent/sample_00_future.npy
samples/pred_len10/transformer_velocity_consistent/sample_01_future.npy
samples/pred_len10/transformer_velocity_consistent/sample_02_future.npy
samples/pred_len10/transformer_velocity_consistent/sample_03_future.npy
samples/pred_len10/candidate_comparison.json
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
