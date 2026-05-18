"""Diffusion model interface and temporary placeholder implementation."""

from __future__ import annotations

from typing import Protocol

import torch

from diffusion_tracking_manager.constants import (
    DEFAULT_COMMAND_HORIZON,
    MOTION_FRAME_DIM,
    ROOT_QUAT_SLICE,
)
from diffusion_tracking_manager.motion_types import validate_motion_sequence
from diffusion_tracking_manager.rotations import normalize_quat


class DiffusionModelInterface(Protocol):
    """Interface expected from the future pretrained diffusion model."""

    def generate(self, history: torch.Tensor, task: dict | None = None) -> torch.Tensor:
        """Generate future motion as [num_envs, horizon, 65]."""
        ...


class PlaceholderDiffusionModel:
    """Temporary diffusion stub that repeats the latest history frame.

    This class exists only to exercise the manager pipeline before the real
    checkpoint and inference API are available.
    """

    def __init__(self, output_horizon: int = DEFAULT_COMMAND_HORIZON) -> None:
        if output_horizon < DEFAULT_COMMAND_HORIZON:
            raise ValueError(
                f"output_horizon must be at least {DEFAULT_COMMAND_HORIZON}, got {output_horizon}"
            )
        self.output_horizon = int(output_horizon)

    @torch.no_grad()
    def generate(self, history: torch.Tensor, task: dict | None = None) -> torch.Tensor:  # noqa: ARG002
        validate_motion_sequence(history, "history", min_horizon=1)
        latest = history[:, -1, :].clone()
        latest[:, ROOT_QUAT_SLICE] = normalize_quat(latest[:, ROOT_QUAT_SLICE])
        future = latest[:, None, :].expand(-1, self.output_horizon, -1).clone()
        if future.shape[-1] != MOTION_FRAME_DIM:
            raise RuntimeError("placeholder generated an invalid motion frame size")
        return future
