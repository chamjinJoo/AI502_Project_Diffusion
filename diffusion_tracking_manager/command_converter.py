"""Conversion from diffusion future motion to SONIC tracking commands."""

from __future__ import annotations

import torch

from diffusion_tracking_manager.constants import (
    DEFAULT_COMMAND_HORIZON,
    JOINT_POS_SLICE,
    JOINT_VEL_SLICE,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
)
from diffusion_tracking_manager.motion_types import SonicTrackingCommand, validate_motion_sequence
from diffusion_tracking_manager.rotations import (
    normalize_quat,
    quat_to_rot6d,
    relative_quat_from_world,
)


class SonicCommandConverter:
    """Build the pretrained SONIC G1 tracking command from diffusion output."""

    def __init__(
        self,
        command_horizon: int = DEFAULT_COMMAND_HORIZON,
        diffusion_orientation_is_relative: bool = True,
    ) -> None:
        if command_horizon <= 0:
            raise ValueError(f"command_horizon must be positive, got {command_horizon}")
        self.command_horizon = int(command_horizon)
        self.diffusion_orientation_is_relative = bool(diffusion_orientation_is_relative)

    def convert(
        self,
        future_motion: torch.Tensor,
        current_root_quat: torch.Tensor | None = None,
        start: int = 0,
    ) -> SonicTrackingCommand:
        """Convert [N, H, 65] future motion into the 10-frame SONIC command.

        The default assumes diffusion quaternions are already expressed in the
        current robot frame, matching the task description. If a future model
        emits world-frame target quaternions, set diffusion_orientation_is_relative
        to False and pass current_root_quat.
        """
        validate_motion_sequence(future_motion, "future_motion", min_horizon=self.command_horizon)
        end = start + self.command_horizon
        if start < 0 or end > future_motion.shape[1]:
            raise ValueError(
                f"Cannot take command window [{start}, {end}) from horizon {future_motion.shape[1]}"
            )

        window = future_motion[:, start:end, :]
        joint_pos = window[:, :, JOINT_POS_SLICE]
        joint_vel = window[:, :, JOINT_VEL_SLICE]
        root_quat = normalize_quat(window[:, :, ROOT_QUAT_SLICE])
        root_pos = window[:, :, ROOT_POS_SLICE]

        if self.diffusion_orientation_is_relative:
            root_quat_relative = root_quat
        else:
            if current_root_quat is None:
                raise ValueError(
                    "current_root_quat is required when diffusion_orientation_is_relative=False"
                )
            root_quat_relative = relative_quat_from_world(current_root_quat, root_quat)

        root_rot6_relative = quat_to_rot6d(root_quat_relative)
        return SonicTrackingCommand(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_quat_relative=root_quat_relative,
            root_pos=root_pos,
            root_rot6_relative=root_rot6_relative,
            future_motion=window,
        )
