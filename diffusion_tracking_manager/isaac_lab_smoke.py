"""Run the diffusion tracking manager inside the real SONIC Isaac Lab G1 environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Allow direct execution as `python diffusion_tracking_manager/isaac_lab_smoke.py`
# while preserving the same imports used when the package is imported normally.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf, open_dict
import torch

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\nERROR: Isaac Lab is required for this smoke test.\n"
        "Activate the Isaac Lab / AI502 environment before running it.\n",
        file=sys.stderr,
    )
    raise

from diffusion_tracking_manager.diffusion_interface import PlaceholderDiffusionModel
from diffusion_tracking_manager.manager import DiffusionTrackingManager
from diffusion_tracking_manager.motion_types import MotionState, SonicTrackingCommand
from diffusion_tracking_manager.tracking_policy import SonicActorTrackingPolicyAdapter
from diffusion_tracking_manager.constants import DEFAULT_COMMAND_HORIZON, MOTION_FRAME_DIM
from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices
from gear_sonic import train_agent_trl
from gear_sonic.trl.trainer import ppo_trainer
from gear_sonic.trl.utils import common as trl_utils_common
from gear_sonic.utils import common as rl_utils_common
from gear_sonic.utils import config_utils, obs_utils

config_utils.register_rl_resolvers()


def _launch_isaac(config):
    """Launch Isaac Sim with the same lightweight settings as SONIC eval."""
    try:
        with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
            isaacsim_version = f.read().strip()
    except FileNotFoundError:
        isaacsim_version = "4.5"

    if isaacsim_version == "4.5":
        from isaaclab.app import AppLauncher
    elif isaacsim_version == "4.2":
        from omni.isaac.lab.app import AppLauncher
    else:
        raise ValueError(f"Unsupported Isaac Sim version marker: {isaacsim_version}")

    parser = argparse.ArgumentParser(description="Diffusion tracking manager Isaac Lab smoke test.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_args]

    args_cli.num_envs = config.num_envs
    args_cli.seed = config.seed
    args_cli.env_spacing = config.manager_env.config.env_spacing
    args_cli.output_dir = config.output_dir
    args_cli.headless = config.headless
    args_cli.multi_gpu = False
    args_cli.distributed = False
    args_cli.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args_cli.enable_cameras = False
    kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    args_cli.kit_args = f"{kit_args} --no-window" if args_cli.headless else kit_args

    app_launcher = AppLauncher(args_cli)
    return app_launcher.app, args_cli


def _prepare_config(config: OmegaConf) -> OmegaConf:
    """Apply smoke-test defaults without mutating unrelated SONIC behavior."""
    smoke_motion_file = config.get(
        "diffusion_smoke_motion_file",
        "data/motion_lib_bones_seed/robot_filtered/230214/jump_ff_360_002__A193_M.pkl",
    )
    with open_dict(config):
        config.headless = bool(config.get("headless", True))
        config.num_envs = int(config.get("num_envs", 1))
        config.use_wandb = False
        config.multi_gpu = False
        config.checkpoint = config.get("checkpoint") or "sonic_release/last.pt"
        config.output_dir = config.get("output_dir") or "logs_eval/diffusion_tracking_smoke/output"
        config.manager_env.config.save_rendering_dir = "logs_eval/diffusion_tracking_smoke/renderings"
        config.manager_env.config.experiment_dir = "logs_eval/diffusion_tracking_smoke"
        config.manager_env.config.headless = config.headless

        # This smoke test does not evaluate reference-motion tracking quality.
        # Point the motion library at one concrete file so Isaac Lab can reset
        # the command term without scanning the full 129k-motion release tree.
        config.manager_env.commands.motion.motion_lib_cfg.motion_file = smoke_motion_file
        config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = "dummy"
        config.manager_env.commands.motion.motion_lib_cfg.filter_motion_keys = None
        config.manager_env.commands.motion.filter_motion_keys = None

        # The placeholder diffusion target is not synchronized with SONIC's
        # reference-motion command term, so reference-tracking failure checks can
        # end the episode immediately after reset. Disable only those failure
        # terminations for this smoke path so the live policy call can be
        # inspected over multiple steps. The command time-out remains active.
        disable_tracking_terminations = bool(
            config.get("diffusion_smoke_disable_tracking_terminations", True)
        )
        if disable_tracking_terminations:
            config.manager_env.terminations.anchor_pos = None
            config.manager_env.terminations.anchor_ori = None
            config.manager_env.terminations.anchor_ori_full = None
            config.manager_env.terminations.ee_body_pos = None
            config.manager_env.terminations.anchor_pos_xy = None
            config.manager_env.terminations.foot_pos_xyz = None
            config.manager_env.terminations.cumm_body_pos_error = None
            config.manager_env.terminations.cumm_body_ori_error = None
            config.manager_env.terminations.cumm_body_pos_error_local = None
            config.manager_env.terminations.cumm_body_ori_error_local = None

        if "eval_overrides" in config and config.eval_overrides is not None:
            config.eval_overrides.headless = config.headless
            config.eval_overrides.num_envs = config.num_envs
    return config


def _build_env_and_policy(config: OmegaConf, device: str, args_cli):
    """Create the SONIC env and load the released checkpoint."""
    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in ["policy", "critic"]:
            continue
        group_obs_dims, group_obs_names, group_obs_total_dim = obs_utils.get_group_term_obs_shape(
            example_obs, key
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]
    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    # Materialize lazy layers before loading the checkpoint, matching the training path.
    trl_utils_common.materialize_lazy_params(policy, env)
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("actor_model_state_dict") or checkpoint.get("policy_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint {config.checkpoint} has no actor/policy state dict")
    policy.load_state_dict(state_dict)

    value_model = None
    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)
    model.eval()
    env.reinit_dr()
    return env, model.policy


def _extract_motion_state(env) -> MotionState:
    """Read current G1 body state from the live Isaac Lab environment."""
    robot = env.env.scene["robot"]
    body_indices = get_body_joint_indices(robot)
    joint_pos = robot.data.joint_pos[:, body_indices]
    joint_vel = robot.data.joint_vel[:, body_indices]

    # The temporary diffusion convention is current-frame relative, so the
    # current root pose is identity/zero. A real diffusion model can replace this
    # with its finalized egocentric root state encoding.
    root_quat = torch.zeros(joint_pos.shape[0], 4, device=joint_pos.device, dtype=joint_pos.dtype)
    root_quat[:, 0] = 1.0
    root_pos = torch.zeros(joint_pos.shape[0], 3, device=joint_pos.device, dtype=joint_pos.dtype)
    return MotionState(joint_pos=joint_pos, joint_vel=joint_vel, root_quat=root_quat, root_pos=root_pos)


def _reference_command_from_env(
    env,
    command_horizon: int = DEFAULT_COMMAND_HORIZON,
) -> SonicTrackingCommand:
    """Build a SONIC command from the environment's loaded reference motion.

    This is a smoke-test diagnostic path. It bypasses the placeholder diffusion
    output and feeds the same motion-library future command that SONIC normally
    uses, which helps verify that the released tracking actor and tokenizer
    adapter can track a real reference motion.
    """
    motion_command = env.env.command_manager.get_term("motion")
    num_envs = motion_command.num_envs
    num_future_frames = motion_command.num_future_frames
    if num_future_frames < command_horizon:
        raise ValueError(
            f"Reference command has {num_future_frames} future frames, "
            f"but {command_horizon} are required"
        )

    joint_pos = motion_command.joint_pos_multi_future.view(num_envs, num_future_frames, -1)[
        :, :command_horizon, :
    ]
    joint_vel = motion_command.joint_vel_multi_future.view(num_envs, num_future_frames, -1)[
        :, :command_horizon, :
    ]
    root_pos = motion_command.root_pos_multi_future.view(num_envs, num_future_frames, -1)[
        :, :command_horizon, :
    ]
    root_rot6_relative = motion_command.root_rot_dif_l_multi_future.view(
        num_envs, num_future_frames, -1
    )[:, :command_horizon, :]

    # The live SONIC actor consumes the 6D relative orientation field. Keep an
    # identity quaternion placeholder here only to satisfy the shared command
    # container shape; it is not used by SonicActorTrackingPolicyAdapter.
    root_quat_relative = joint_pos.new_zeros(num_envs, command_horizon, 4)
    root_quat_relative[..., 0] = 1.0
    future_motion = joint_pos.new_zeros(num_envs, command_horizon, MOTION_FRAME_DIM)
    future_motion[..., :29] = joint_pos
    future_motion[..., 29:58] = joint_vel
    future_motion[..., 58:62] = root_quat_relative
    future_motion[..., 62:65] = root_pos
    return SonicTrackingCommand(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_quat_relative=root_quat_relative,
        root_pos=root_pos,
        root_rot6_relative=root_rot6_relative,
        future_motion=future_motion,
    )


@hydra.main(config_path="../gear_sonic/config", config_name="base", version_base="1.1")
def main(config: OmegaConf) -> None:
    config = _prepare_config(config)
    Path("logs_eval/diffusion_tracking_smoke").mkdir(parents=True, exist_ok=True)

    simulation_app, args_cli = _launch_isaac(config)
    try:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.info(f"Hydra output: {HydraConfig.get().runtime.output_dir}")
        logger.info(f"Using checkpoint: {config.checkpoint}")
        command_source = str(config.get("diffusion_smoke_command_source", "placeholder"))
        valid_command_sources = {"placeholder", "reference_motion"}
        if command_source not in valid_command_sources:
            raise ValueError(
                f"diffusion_smoke_command_source must be one of {valid_command_sources}, "
                f"got {command_source!r}"
            )
        logger.info(f"Smoke command source: {command_source}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rl_utils_common.seeding(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        env, policy = _build_env_and_policy(config, device, args_cli)
        adapter = SonicActorTrackingPolicyAdapter(policy, deterministic=True)
        manager = DiffusionTrackingManager(
            diffusion_model=PlaceholderDiffusionModel(output_horizon=10),
            tracking_policy=adapter,
            num_envs=config.num_envs,
            device=device,
        )

        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        obs_dict = {key: value.to(device) for key, value in obs_dict.items()}
        state = _extract_motion_state(env)
        manager.reset(state=state)
        adapter.reset_rollout()

        max_steps = int(config.get("diffusion_smoke_max_steps", 20))
        done_count = 0
        last_reward_mean = 0.0
        last_future_motion = None
        last_command = None
        for step in range(max_steps):
            state = _extract_motion_state(env)
            if command_source == "placeholder":
                low_level_action = manager.act(state, obs_dict=obs_dict, force_replan=(step == 0))
                last_future_motion = manager.cached_future_motion
                last_command = manager.cached_command
            else:
                # Maintain the same 20-frame history buffer for debugging, but
                # use the real environment reference command instead of the
                # placeholder diffusion output for the actor call.
                manager.history_buffer.append(state)
                last_command = _reference_command_from_env(env)
                last_future_motion = last_command.future_motion
                low_level_action = adapter.act_from_command(last_command, obs_dict=obs_dict)
            results = env.step({"actions": low_level_action})
            obs_dict, rewards, dones, infos = results  # noqa: F841
            obs_dict = {key: value.to(device) for key, value in obs_dict.items()}
            done_count += int(dones.sum().item())
            last_reward_mean = float(rewards.mean().item())
            reset_ids = dones.nonzero(as_tuple=True)[0]
            if reset_ids.numel() > 0:
                manager.reset(env_ids=reset_ids, state=_extract_motion_state(env))
                adapter.reset_rollout()
            logger.info(
                f"step={step + 1}/{max_steps} action={tuple(low_level_action.shape)} "
                f"reward_mean={last_reward_mean:.4f} dones_total={done_count}"
            )

        assert last_command is not None
        logger.info(f"history_shape={tuple(manager.history_buffer.history().shape)}")
        logger.info(f"future_shape={tuple(last_future_motion.shape)}")
        logger.info(f"sonic_command_shape={tuple(last_command.sonic_encoder_command.shape)}")
        logger.info(f"last_reward_mean={last_reward_mean:.4f}, dones_total={done_count}")
    except Exception:
        simulation_app.close()
        raise
    else:
        # Isaac/Omniverse may hang during normal shutdown after a short smoke
        # run. The upstream SONIC eval script also exits this way.
        os._exit(0)


if __name__ == "__main__":
    main()
