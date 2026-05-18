"""Typed containers and validators for diffusion tracking tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from diffusion_tracking_manager.constants import (
    DEFAULT_COMMAND_HORIZON,
    JOINT_POS_SLICE,
    JOINT_VEL_SLICE,
    MOTION_FRAME_DIM,
    NUM_JOINTS,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
    SONIC_FRAME_COMMAND_DIM,
)
from diffusion_tracking_manager.rotations import normalize_quat


@dataclass
class MotionState:
    """Current per-environment robot motion state in the diffusion frame layout."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_quat: torch.Tensor
    root_pos: torch.Tensor

    def as_frame(self) -> torch.Tensor:
        """Pack the state as [num_envs, 65] for the history buffer."""
        _expect_2d(self.joint_pos, "joint_pos", NUM_JOINTS)
        _expect_2d(self.joint_vel, "joint_vel", NUM_JOINTS)
        _expect_2d(self.root_quat, "root_quat", 4)
        _expect_2d(self.root_pos, "root_pos", 3)
        num_envs = self.joint_pos.shape[0]
        if self.joint_vel.shape[0] != num_envs:
            raise ValueError("joint_vel batch size must match joint_pos")
        if self.root_quat.shape[0] != num_envs:
            raise ValueError("root_quat batch size must match joint_pos")
        if self.root_pos.shape[0] != num_envs:
            raise ValueError("root_pos batch size must match joint_pos")
        return torch.cat(
            [self.joint_pos, self.joint_vel, normalize_quat(self.root_quat), self.root_pos],
            dim=-1,
        )

    @classmethod
    def from_frame(cls, frame: torch.Tensor) -> "MotionState":
        """Unpack a [num_envs, 65] frame tensor."""
        _expect_2d(frame, "frame", MOTION_FRAME_DIM)
        return cls(
            joint_pos=frame[:, JOINT_POS_SLICE],
            joint_vel=frame[:, JOINT_VEL_SLICE],
            root_quat=normalize_quat(frame[:, ROOT_QUAT_SLICE]),
            root_pos=frame[:, ROOT_POS_SLICE],
        )


@dataclass
class SonicTrackingCommand:
    """Converted command consumed by the fixed SONIC tracking policy."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_quat_relative: torch.Tensor
    root_pos: torch.Tensor
    root_rot6_relative: torch.Tensor
    future_motion: torch.Tensor

    @property
    def num_envs(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def command_multi_future_nonflat(self) -> torch.Tensor:
        """Per-frame G1 joint position and velocity command, shape [N, H, 58]."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def motion_anchor_ori_b_mf_nonflat(self) -> torch.Tensor:
        """Per-frame relative root orientation as rot6, shape [N, H, 6]."""
        return self.root_rot6_relative

    @property
    def sonic_encoder_command(self) -> torch.Tensor:
        """Full c_e command tensor, shape [N, H, 64] = [q, qdot, rot6]."""
        return torch.cat(
            [self.command_multi_future_nonflat, self.motion_anchor_ori_b_mf_nonflat],
            dim=-1,
        )

    @property
    def flat_sonic_encoder_command(self) -> torch.Tensor:
        """Flattened command, shape [N, H * 64]."""
        return self.sonic_encoder_command.reshape(self.num_envs, -1)

    def tokenizer_obs(self, encoder_order: Iterable[str] = ("g1", "teleop", "smpl")) -> dict[str, torch.Tensor]:
        """Return named tokenizer observations for a SONIC-style universal-token actor."""
        encoder_names = list(encoder_order)
        encoder_index = self.joint_pos.new_zeros((self.num_envs, len(encoder_names)))
        if "g1" not in encoder_names:
            raise ValueError("encoder_order must contain 'g1' for SONIC G1 tracking commands")
        encoder_index[:, encoder_names.index("g1")] = 1.0
        return {
            "encoder_index": encoder_index,
            "command_multi_future_nonflat": self.command_multi_future_nonflat,
            "command_z_multi_future_nonflat": self.root_pos[..., 2:3],
            "motion_anchor_ori_b_mf_nonflat": self.motion_anchor_ori_b_mf_nonflat,
        }

    def flatten_tokenizer_obs(
        self,
        tokenizer_obs_names: Iterable[str],
        tokenizer_obs_dims: dict[str, tuple[int, ...] | list[int]],
        encoder_order: Iterable[str] = ("g1", "teleop", "smpl"),
    ) -> torch.Tensor:
        """Flatten known tokenizer fields and zero-fill unrelated SONIC modes.

        This keeps the diffusion manager usable with the existing universal-token
        actor without fabricating teleop or SMPL data that the G1 encoder will not read.
        """
        known = self.tokenizer_obs(encoder_order=encoder_order)
        parts = []
        for name in tokenizer_obs_names:
            dims = tuple(int(v) for v in tokenizer_obs_dims[name])
            if name in known:
                value = known[name]
                expected_shape = (self.num_envs, *dims)
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"{name} has shape {tuple(value.shape)}, expected {expected_shape}"
                    )
            else:
                value = self.joint_pos.new_zeros((self.num_envs, *dims))
            parts.append(value.reshape(self.num_envs, -1))
        return torch.cat(parts, dim=-1)


def validate_motion_sequence(
    sequence: torch.Tensor,
    name: str,
    min_horizon: int = DEFAULT_COMMAND_HORIZON,
) -> None:
    """Validate [num_envs, horizon, 65] motion tensors from history or diffusion output."""
    if sequence.dim() != 3:
        raise ValueError(f"{name} must be rank 3 [num_envs, horizon, 65], got {tuple(sequence.shape)}")
    if sequence.shape[-1] != MOTION_FRAME_DIM:
        raise ValueError(f"{name} frame dim must be {MOTION_FRAME_DIM}, got {sequence.shape[-1]}")
    if sequence.shape[1] < min_horizon:
        raise ValueError(f"{name} horizon must be at least {min_horizon}, got {sequence.shape[1]}")


def validate_sonic_command(command: SonicTrackingCommand, horizon: int = DEFAULT_COMMAND_HORIZON) -> None:
    """Validate converted SONIC command shape before policy execution."""
    n, h, _ = command.joint_pos.shape
    expected_shapes = {
        "joint_pos": (n, h, NUM_JOINTS),
        "joint_vel": (n, h, NUM_JOINTS),
        "root_quat_relative": (n, h, 4),
        "root_pos": (n, h, 3),
        "root_rot6_relative": (n, h, 6),
        "sonic_encoder_command": (n, h, SONIC_FRAME_COMMAND_DIM),
    }
    if h != horizon:
        raise ValueError(f"command horizon must be {horizon}, got {h}")
    for name, expected in expected_shapes.items():
        value = getattr(command, name) if name != "sonic_encoder_command" else command.sonic_encoder_command
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match {expected}")


def _expect_2d(tensor: torch.Tensor, name: str, dim: int) -> None:
    if tensor.dim() != 2 or tensor.shape[-1] != dim:
        raise ValueError(f"{name} must have shape [num_envs, {dim}], got {tuple(tensor.shape)}")
