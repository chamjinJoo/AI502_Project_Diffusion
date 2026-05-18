"""Interfaces and adapters for fixed SONIC tracking policies."""

from __future__ import annotations

from typing import Protocol

import torch

from diffusion_tracking_manager.motion_types import MotionState, SonicTrackingCommand


class TrackingPolicyInterface(Protocol):
    """Minimal interface for a policy that can consume converted SONIC commands."""

    def act_from_command(
        self,
        command: SonicTrackingCommand,
        robot_state: MotionState | torch.Tensor | None = None,
        obs_dict: dict | None = None,
    ) -> torch.Tensor:
        """Return low-level action for each parallel environment."""
        ...


class CallableTrackingPolicyAdapter:
    """Adapter for callables that already know how to consume command objects."""

    def __init__(self, policy_callable) -> None:
        self.policy_callable = policy_callable

    def act_from_command(
        self,
        command: SonicTrackingCommand,
        robot_state: MotionState | torch.Tensor | None = None,
        obs_dict: dict | None = None,
    ) -> torch.Tensor:
        return self.policy_callable(command, robot_state=robot_state, obs_dict=obs_dict)


class SonicActorTrackingPolicyAdapter:
    """Adapter that feeds converted diffusion targets into the live SONIC actor.

    The pretrained SONIC actor still needs the normal proprioceptive observations
    from Isaac Lab. This adapter preserves those observations and replaces only
    the flattened `tokenizer` group with the G1 command produced by the diffusion
    manager. The actor then runs through its existing G1 encoder and dynamic
    decoder, so the pretrained tracking policy remains unchanged.
    """

    def __init__(self, policy, deterministic: bool = True) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.last_policy_output = None
        self.last_policy_obs_dict: dict | None = None

    def reset_rollout(self) -> None:
        """Reset temporal rollout buffers on the underlying SONIC actor."""
        if hasattr(self.policy, "init_rollout"):
            self.policy.init_rollout()

    @torch.no_grad()
    def act_from_command(
        self,
        command: SonicTrackingCommand,
        robot_state: MotionState | torch.Tensor | None = None,  # noqa: ARG002
        obs_dict: dict | None = None,
    ) -> torch.Tensor:
        if obs_dict is None:
            raise ValueError("SonicActorTrackingPolicyAdapter requires the live env obs_dict")

        policy_obs = dict(obs_dict)
        actor_module = getattr(self.policy, "actor_module", None)
        if actor_module is None:
            raise ValueError("SONIC policy must expose actor_module for tokenizer metadata")

        encoder_order = actor_module.encoder_sample_probs.keys()
        policy_obs["tokenizer"] = command.flatten_tokenizer_obs(
            tokenizer_obs_names=actor_module.tokenizer_obs_names,
            tokenizer_obs_dims=actor_module.tokenizer_obs_dims,
            encoder_order=encoder_order,
        )

        # Keep named fields for debugging and for adapters that inspect obs_dict.
        policy_obs.update(command.tokenizer_obs(encoder_order=encoder_order))
        self.last_policy_output = self.policy.rollout(obs_dict=policy_obs)
        self.last_policy_obs_dict = self.last_policy_output.get("obs_dict", policy_obs)

        if self.deterministic:
            return self.policy.action_mean.detach()
        return self.last_policy_output["actions"].detach()
