# Diffusion Tracking Manager

This directory contains an isolated manager for connecting a future motion diffusion model to the pretrained `sonic_release` G1 tracking policy.

The intended runtime pipeline is:

```text
current robot state / 20-frame motion history
    -> future diffusion model
    -> future motion target
    -> SONIC tracking-command conversion
    -> fixed sonic_release tracking policy
    -> low-level robot action
```

The implementation is kept separate from the base SONIC / GR00T code so later experiments can change the diffusion side without editing the large tracking repository or the pretrained tracker.

## Design

The manager is policy-like: call `act(...)` with the current batched robot state and it returns a batched low-level action. Internally it maintains a rolling motion history for each parallel environment, queries a diffusion-model interface when a new future target is needed, converts the generated future target into the SONIC G1 command, and calls the provided tracking policy.

Main files:

- `manager.py`: `DiffusionTrackingManager`, the combined policy wrapper.
- `history.py`: per-environment rolling 20-frame history with full and partial reset support.
- `diffusion_interface.py`: future diffusion interface plus a temporary placeholder model.
- `command_converter.py`: converts diffusion output into SONIC G1 tracking commands.
- `motion_types.py`: typed containers and explicit shape validation.
- `dry_run.py`: minimal structural dry run using a mock SONIC tracker.

## Diffusion I/O Assumptions

History input to diffusion:

```text
[num_envs, 20, 65]
```

Each 65D frame is:

```text
joint_pos: 29
joint_vel: 29
root_quat: 4  # wxyz, relative to current frame
root_pos: 3
```

Future output from diffusion:

```text
[num_envs, horizon, 65]
```

The first implementation assumes `horizon == 10`, but the manager supports longer horizons. With longer horizons, it uses rolling 10-frame command windows and replans after `horizon - 10` tracker steps by default.

## SONIC Command Format

The converter builds the G1 encoder command:

```text
[num_envs, 10, 64] = [joint_pos(29), joint_vel(29), relative_root_rot6(6)]
```

Quaternions are assumed to be `wxyz`. By default, the diffusion output orientation is treated as already relative to the current robot frame, matching the task definition. If a future diffusion model emits world-frame target quaternions, construct `SonicCommandConverter(diffusion_orientation_is_relative=False)` through the manager and pass current root orientation through `act(...)`.

Existing SONIC rotation utilities from `gear_sonic.trl.utils.torch_transform` are reused when importable. A small local fallback exists so the dry run can still validate structure in lighter environments.

## Parallel Environments and Resets

All tensors are batched by `num_envs`. `MotionHistoryBuffer.reset()` supports:

```python
manager.reset()                         # all envs, zero state with identity quaternion
manager.reset(state=current_state)       # all envs, fill history with current state
manager.reset(env_ids=[1, 3], state=...) # selected envs
```

Any reset invalidates the cached diffusion plan, so the next `act()` regenerates a target.

## Placeholder Diffusion Model

`PlaceholderDiffusionModel` is temporary. It repeats the latest history frame for the requested horizon and only exists to test the integration path. It does not implement diffusion, sampling, denoising, or latent steering.

Replace it later with an object implementing:

```python
class RealDiffusionModel:
    def generate(self, history: torch.Tensor, task: dict | None = None) -> torch.Tensor:
        # history: [num_envs, 20, 65]
        # return:  [num_envs, horizon, 65]
        ...
```

## Tracking Policy Integration

The cleanest integration is to provide a tracker adapter with:

```python
def act_from_command(command, robot_state=None, obs_dict=None) -> torch.Tensor:
    ...
```

`command.sonic_encoder_command` contains `[num_envs, 10, 64]`. Named tokenizer fields are also available:

- `command.command_multi_future_nonflat`: `[num_envs, 10, 58]`
- `command.motion_anchor_ori_b_mf_nonflat`: `[num_envs, 10, 6]`
- `command.tokenizer_obs()`: dict for G1 tokenizer-style inputs

If wrapping the existing `sonic_release` actor directly, pass the normal proprioceptive `obs_dict` from the environment and let the adapter inject these command fields into the actor input.

`SonicActorTrackingPolicyAdapter` does this for the live Python actor. It preserves the Isaac Lab proprioceptive observations, replaces the flattened SONIC `tokenizer` observation with the diffusion-generated G1 command, runs the pretrained actor, and returns the deterministic action mean.

## Using It as a Low-Level Policy

The intended external interface is `DiffusionTrackingManager.act(...)`. A higher-level task environment or MDP should treat the manager as the low-level policy that maps the current G1 state and optional task conditioning to a 29D joint action:

```text
outer MDP observation
    -> extract current G1 motion state
    -> DiffusionTrackingManager.act(...)
    -> low-level G1 action [num_envs, 29]
```

Typical construction:

```python
from diffusion_tracking_manager.manager import DiffusionTrackingManager
from diffusion_tracking_manager.motion_types import MotionState
from diffusion_tracking_manager.tracking_policy import SonicActorTrackingPolicyAdapter

tracking_adapter = SonicActorTrackingPolicyAdapter(sonic_release_actor, deterministic=True)

manager = DiffusionTrackingManager(
    diffusion_model=diffusion_model,
    tracking_policy=tracking_adapter,
    num_envs=num_envs,
    device=device,
)
```

For another environment, the required loop is:

1. Create or load a diffusion model object with `generate(history, task=None)`.
2. Load the fixed `sonic_release` actor and wrap it with `SonicActorTrackingPolicyAdapter`.
3. Construct `DiffusionTrackingManager(diffusion_model, tracking_adapter, num_envs, device=...)`.
4. On environment reset, call `manager.reset(env_ids=..., state=current_motion_state)`. If the SONIC actor keeps rollout state, also call `tracking_adapter.reset_rollout()`.
5. On every control step, build a `MotionState` from the environment:

```text
joint_pos: [num_envs, 29]
joint_vel: [num_envs, 29]
root_quat: [num_envs, 4]  # wxyz, current-frame convention unless changed
root_pos: [num_envs, 3]
```

6. Pass the current SONIC-compatible observation dictionary as `obs_dict`.
7. Send the returned `[num_envs, 29]` action to the environment.

Minimal step pattern:

```python
obs_dict = env.reset_all()
state = extract_motion_state(env)
manager.reset(state=state)
tracking_adapter.reset_rollout()

for _ in range(num_steps):
    state = extract_motion_state(env)
    task = build_optional_task_conditioning(env)
    action = manager.act(state, obs_dict=obs_dict, task=task)

    obs_dict, rewards, dones, infos = env.step({"actions": action})
    reset_ids = dones.nonzero(as_tuple=True)[0]
    if reset_ids.numel() > 0:
        manager.reset(env_ids=reset_ids, state=extract_motion_state(env))
        tracking_adapter.reset_rollout()
```

The manager owns the 20-frame history, diffusion replanning cadence, future-command conversion, and SONIC actor call. The outer MDP should not build the 10-frame SONIC command itself.

There is one important constraint: `SonicActorTrackingPolicyAdapter` is not a fully observation-free policy. The pretrained SONIC actor still expects the normal proprioceptive observation fields produced by the SONIC G1 environment. The adapter only replaces the tokenizer command part with the diffusion-generated command. Therefore, another environment can use this manager directly if it either is the SONIC G1 Isaac Lab environment or can provide an equivalent actor `obs_dict`. If a different environment has a different observation layout, add a small adapter that converts that environment's robot state into the SONIC actor observation format before calling `manager.act(...)`.

This means the current implementation achieves the single-low-level-policy structure for environments that can provide SONIC-compatible actor observations. It does not yet remove the dependency on SONIC's proprioceptive observation format; that is the main integration point to solve when attaching the manager to a completely different MDP.

## Dry Run

Use the project conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate AI502
python diffusion_tracking_manager/dry_run.py
```

Expected shape checks:

```text
history: (4, 20, 65)
future_motion: (4, 10, 65)
sonic_command: (4, 10, 64)
action: (4, 29)
```

If `pytest` is available:

```bash
pytest diffusion_tracking_manager/tests/test_dry_run.py
```

## Isaac Lab Smoke Test

To run the manager inside the real G1 SONIC Isaac Lab environment with the released checkpoint:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate AI502
python diffusion_tracking_manager/isaac_lab_smoke.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    checkpoint=sonic_release/last.pt \
    headless=True \
    num_envs=1 \
    +diffusion_smoke_max_steps=20 \
    manager_env/terminations=tracking/eval \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False
```

`isaac_lab_smoke.py` is a bounded integration smoke test, not a full evaluation script. It does the following:

1. Applies smoke-test config defaults: headless mode, no wandb, one GPU, the released `sonic_release/last.pt` checkpoint, and one reference motion file.
2. Launches Isaac Lab through the normal SONIC/Isaac Lab launcher.
3. Builds the real SONIC G1 manager environment.
4. Instantiates the released SONIC actor and loads the checkpoint.
5. Resets the Isaac Lab environment and initializes the diffusion manager history from the current G1 joint state.
6. For each smoke step, extracts the current G1 state, runs the placeholder diffusion model, converts the generated future into the G1 SONIC tokenizer command, injects that command into the live SONIC actor observation, gets a low-level action, and calls `env.step(...)`.
7. Prints action, reward, done count, history, generated future, and SONIC command shapes.
8. Exits after `diffusion_smoke_max_steps`.

The script may look like it terminates right after reset if `+diffusion_smoke_max_steps=1` is used, or if the output is inspected only after the short smoke run finishes. This is expected. Increase the step count when you want a longer run:

```bash
+diffusion_smoke_max_steps=500
```

The smoke script disables reference-tracking failure terminations by default. This is intentional because the placeholder diffusion target is not synchronized with the environment's loaded reference motion, so terms such as `anchor_pos`, `anchor_ori_full`, and `ee_body_pos` can end the episode immediately after reset even when the diffusion-manager-to-SONIC-policy path is working. The command time-out remains active. To re-enable those failure terminations for stricter debugging:

```bash
+diffusion_smoke_disable_tracking_terminations=False
```

For visual inspection, run without headless mode:

```bash
headless=False +diffusion_smoke_max_steps=500
```

The smoke test intentionally uses a placeholder target that repeats the current robot state. It is only an integration test for the command and policy call path, not a meaningful locomotion policy evaluation.

The single motion file is loaded so the Isaac Lab command manager, resets, observations, and rewards have a valid reference motion. In the default smoke mode, that reference motion is not fed to the actor. The actor receives the placeholder diffusion command instead. This is why the robot is not expected to track the loaded motion when `diffusion_smoke_command_source=placeholder`.

To check whether the released `sonic_release` actor can still track the loaded motion through the adapter, run the smoke test with the reference-motion command source:

```bash
python diffusion_tracking_manager/isaac_lab_smoke.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    checkpoint=sonic_release/last.pt \
    headless=False \
    num_envs=1 \
    +diffusion_smoke_max_steps=500 \
    +diffusion_smoke_command_source=reference_motion \
    manager_env/terminations=tracking/eval \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False
```

The available smoke command sources are:

- `placeholder`: use the placeholder diffusion output, convert it to a SONIC command, and feed that to the tracker. This tests the diffusion-manager integration path.
- `reference_motion`: bypass the placeholder diffusion output and feed the environment's loaded reference-motion command to the same SONIC actor adapter. This is for verifying that the released tracker and adapter can track a real motion file.

The printed `reward_mean` is the scalar reward returned by the existing SONIC Isaac Lab environment. It is still computed from the environment's normal tracking reward terms, such as reference anchor/body tracking, action rate, joint limits, contacts, and related penalties. In this smoke test, that reward is not a diffusion-model score and is not a reliable success metric, because the placeholder diffusion target simply repeats the current robot state while the environment reward still compares behavior against the loaded reference-motion task. Treat it as a sanity signal that the real `env.step(...)` path ran and returned finite values.

By default the script points SONIC's motion library at a single release motion file and uses dummy SMPL data. This avoids scanning/loading the full motion dataset even though the command term still needs one reference motion for normal Isaac Lab reset/reward plumbing. To use a different reference file:

```bash
+diffusion_smoke_motion_file=/path/to/single_motion.pkl
```

## Limitations and Extension Points

- The actual diffusion model is not implemented here.
- The real checkpoint loading API is intentionally left to the future diffusion model implementation.
- The dry run uses a mock tracker and does not launch Isaac Sim.
- The Isaac Lab smoke test uses the live SONIC actor adapter, but the current-state placeholder target is not expected to produce a task-level behavior.
- Longer diffusion horizons are supported structurally, but the best replanning cadence should be tuned after the real model latency and tracker behavior are known.
