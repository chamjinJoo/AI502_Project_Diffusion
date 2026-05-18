"""Minimal structural dry run for the diffusion tracking manager."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

# Allow direct execution as `python diffusion_tracking_manager/dry_run.py`
# while keeping package-style imports for tests and library use.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_tracking_manager.constants import (
    DEFAULT_COMMAND_HORIZON,
    DEFAULT_HISTORY_LEN,
    MOTION_FRAME_DIM,
    NUM_JOINTS,
)
from diffusion_tracking_manager.diffusion_interface import PlaceholderDiffusionModel
from diffusion_tracking_manager.manager import DiffusionTrackingManager
from diffusion_tracking_manager.motion_types import MotionState, SonicTrackingCommand


class MockSonicTrackingPolicy:
    """Mock tracker used to test the call path without loading Isaac Sim or checkpoints."""

    def __init__(self, action_dim: int = NUM_JOINTS) -> None:
        self.action_dim = action_dim
        self.last_command: SonicTrackingCommand | None = None

    def act_from_command(
        self,
        command: SonicTrackingCommand,
        robot_state: MotionState | torch.Tensor | None = None,  # noqa: ARG002
        obs_dict: dict | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        assert command.sonic_encoder_command.shape == (
            command.num_envs,
            DEFAULT_COMMAND_HORIZON,
            64,
        )
        assert command.command_multi_future_nonflat.shape[-1] == NUM_JOINTS * 2
        assert command.motion_anchor_ori_b_mf_nonflat.shape[-1] == 6
        self.last_command = command
        return torch.zeros(command.num_envs, self.action_dim, device=command.joint_pos.device)


def make_state(num_envs: int, device: torch.device | str = "cpu") -> MotionState:
    joint_pos = torch.randn(num_envs, NUM_JOINTS, device=device) * 0.01
    joint_vel = torch.randn(num_envs, NUM_JOINTS, device=device) * 0.01
    root_quat = torch.zeros(num_envs, 4, device=device)
    root_quat[:, 0] = 1.0
    root_pos = torch.zeros(num_envs, 3, device=device)
    return MotionState(joint_pos=joint_pos, joint_vel=joint_vel, root_quat=root_quat, root_pos=root_pos)


def run_dry_run() -> dict[str, tuple[int, ...]]:
    num_envs = 4
    state = make_state(num_envs)
    tracker = MockSonicTrackingPolicy()
    manager = DiffusionTrackingManager(
        diffusion_model=PlaceholderDiffusionModel(output_horizon=DEFAULT_COMMAND_HORIZON),
        tracking_policy=tracker,
        num_envs=num_envs,
    )
    manager.reset(state=state)

    # Append explicit history frames so the test covers the 20-frame rolling window.
    for _ in range(DEFAULT_HISTORY_LEN):
        manager.history_buffer.append(make_state(num_envs))

    history = manager.history_buffer.history()
    assert history.shape == (num_envs, DEFAULT_HISTORY_LEN, MOTION_FRAME_DIM)

    action = manager.act(make_state(num_envs))
    assert action.shape == (num_envs, NUM_JOINTS)
    assert tracker.last_command is not None
    assert manager.cached_future_motion is not None

    manager.reset(env_ids=[1, 3], state=make_state(num_envs))
    assert manager.history_buffer.history().shape == (num_envs, DEFAULT_HISTORY_LEN, MOTION_FRAME_DIM)

    return {
        "history": tuple(history.shape),
        "future_motion": tuple(manager.diffusion_model.generate(history).shape),
        "sonic_command": tuple(tracker.last_command.sonic_encoder_command.shape),
        "action": tuple(action.shape),
    }


if __name__ == "__main__":
    shapes = run_dry_run()
    for name, shape in shapes.items():
        print(f"{name}: {shape}")
