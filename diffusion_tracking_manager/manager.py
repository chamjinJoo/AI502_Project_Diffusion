"""Policy-like manager combining future diffusion generation and SONIC tracking."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from diffusion_tracking_manager.command_converter import SonicCommandConverter
from diffusion_tracking_manager.constants import DEFAULT_COMMAND_HORIZON, DEFAULT_HISTORY_LEN
from diffusion_tracking_manager.diffusion_interface import DiffusionModelInterface
from diffusion_tracking_manager.history import MotionHistoryBuffer
from diffusion_tracking_manager.motion_types import (
    MotionState,
    SonicTrackingCommand,
    validate_motion_sequence,
    validate_sonic_command,
)


class DiffusionTrackingManager:
    """Single policy-like wrapper: history -> diffusion target -> SONIC tracker -> action."""

    def __init__(
        self,
        diffusion_model: DiffusionModelInterface,
        tracking_policy: Any,
        num_envs: int,
        history_len: int = DEFAULT_HISTORY_LEN,
        command_horizon: int = DEFAULT_COMMAND_HORIZON,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        diffusion_orientation_is_relative: bool = True,
        replan_interval: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.diffusion_model = diffusion_model
        self.tracking_policy = tracking_policy
        self.history_buffer = MotionHistoryBuffer(
            num_envs=self.num_envs,
            history_len=history_len,
            device=self.device,
            dtype=self.dtype,
        )
        self.converter = SonicCommandConverter(
            command_horizon=command_horizon,
            diffusion_orientation_is_relative=diffusion_orientation_is_relative,
        )
        self.replan_interval = replan_interval
        self.cached_future_motion: torch.Tensor | None = None
        self.cached_command: SonicTrackingCommand | None = None
        self.plan_cursor = 0
        self.steps_since_plan = 0

    @torch.no_grad()
    def reset(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
        state: MotionState | torch.Tensor | None = None,
    ) -> None:
        """Reset history and cached plans for all or selected parallel environments."""
        self.history_buffer.reset(env_ids=env_ids, state=state)
        # Partial resets invalidate the shared batched plan because each env may now
        # need a different diffusion context. Regenerating is simpler and explicit.
        self.cached_future_motion = None
        self.cached_command = None
        self.plan_cursor = 0
        self.steps_since_plan = 0

    @torch.no_grad()
    def act(
        self,
        state: MotionState | torch.Tensor,
        obs_dict: dict | None = None,
        task: dict | None = None,
        force_replan: bool = False,
    ) -> torch.Tensor:
        """Append current state, update/reuse diffusion target, and return tracker action."""
        self.history_buffer.append(state)
        if self._needs_replan(force_replan=force_replan):
            self._generate_new_plan(task=task)

        assert self.cached_future_motion is not None
        current_root_quat = self._current_root_quat(state)
        command = self.converter.convert(
            self.cached_future_motion,
            current_root_quat=current_root_quat,
            start=self.plan_cursor,
        )
        validate_sonic_command(command, horizon=self.converter.command_horizon)
        self.cached_command = command

        action = self._call_tracking_policy(command, state, obs_dict)
        self._advance_plan_cursor()
        return action

    def _needs_replan(self, force_replan: bool) -> bool:
        if force_replan or self.cached_future_motion is None:
            return True
        horizon = self.cached_future_motion.shape[1]
        if self.plan_cursor + self.converter.command_horizon > horizon:
            return True
        interval = self._current_replan_interval(horizon)
        return self.steps_since_plan >= interval

    def _generate_new_plan(self, task: dict | None = None) -> None:
        history = self.history_buffer.history(flatten=False)
        validate_motion_sequence(history, "history", min_horizon=self.history_buffer.history_len)
        future = self.diffusion_model.generate(history, task=task)
        future = future.to(device=self.device, dtype=self.dtype)
        validate_motion_sequence(future, "diffusion future", min_horizon=self.converter.command_horizon)
        self.cached_future_motion = future
        self.plan_cursor = 0
        self.steps_since_plan = 0

    def _advance_plan_cursor(self) -> None:
        self.plan_cursor += 1
        self.steps_since_plan += 1

    def _current_replan_interval(self, future_horizon: int) -> int:
        if self.replan_interval is not None:
            if self.replan_interval <= 0:
                raise ValueError(f"replan_interval must be positive, got {self.replan_interval}")
            return int(self.replan_interval)
        return max(1, future_horizon - self.converter.command_horizon)

    def _call_tracking_policy(
        self,
        command: SonicTrackingCommand,
        state: MotionState | torch.Tensor,
        obs_dict: dict | None,
    ) -> torch.Tensor:
        if hasattr(self.tracking_policy, "act_from_command"):
            return self.tracking_policy.act_from_command(command, robot_state=state, obs_dict=obs_dict)

        policy_obs = self._build_policy_obs(command, obs_dict)
        if hasattr(self.tracking_policy, "act"):
            return self.tracking_policy.act(policy_obs)
        if callable(self.tracking_policy):
            return self._call_callable_policy(command, state, policy_obs)
        raise TypeError("tracking_policy must implement act_from_command, act, or be callable")

    def _build_policy_obs(
        self,
        command: SonicTrackingCommand,
        obs_dict: dict | None,
    ) -> dict:
        """Inject converted command fields into an existing SONIC observation dict."""
        policy_obs = {} if obs_dict is None else dict(obs_dict)
        tokenizer_obs = command.tokenizer_obs()
        policy_obs.update(tokenizer_obs)
        policy_obs["sonic_encoder_command"] = command.sonic_encoder_command

        actor_module = getattr(self.tracking_policy, "actor_module", None)
        if actor_module is not None and hasattr(actor_module, "tokenizer_obs_names"):
            tokenizer = command.flatten_tokenizer_obs(
                tokenizer_obs_names=actor_module.tokenizer_obs_names,
                tokenizer_obs_dims=actor_module.tokenizer_obs_dims,
                encoder_order=getattr(actor_module, "encoder_sample_probs", {"g1": 1.0}).keys(),
            )
            policy_obs["tokenizer"] = tokenizer.unsqueeze(1)
        return policy_obs

    def _call_callable_policy(
        self,
        command: SonicTrackingCommand,
        state: MotionState | torch.Tensor,
        policy_obs: dict,
    ) -> torch.Tensor:
        signature = inspect.signature(self.tracking_policy)
        if "command" in signature.parameters:
            return self.tracking_policy(command=command, robot_state=state, obs_dict=policy_obs)
        return self.tracking_policy(policy_obs)

    def _current_root_quat(self, state: MotionState | torch.Tensor) -> torch.Tensor | None:
        if self.converter.diffusion_orientation_is_relative:
            return None
        if isinstance(state, MotionState):
            return state.root_quat.to(device=self.device, dtype=self.dtype)
        return MotionState.from_frame(state.to(device=self.device, dtype=self.dtype)).root_quat
