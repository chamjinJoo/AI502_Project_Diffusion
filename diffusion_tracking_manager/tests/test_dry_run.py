"""Tests for the isolated diffusion tracking manager."""

from diffusion_tracking_manager.constants import (
    DEFAULT_COMMAND_HORIZON,
    DEFAULT_HISTORY_LEN,
    MOTION_FRAME_DIM,
    NUM_JOINTS,
)
from diffusion_tracking_manager.dry_run import run_dry_run


def test_dry_run_shapes():
    shapes = run_dry_run()
    assert shapes["history"] == (4, DEFAULT_HISTORY_LEN, MOTION_FRAME_DIM)
    assert shapes["future_motion"] == (4, DEFAULT_COMMAND_HORIZON, MOTION_FRAME_DIM)
    assert shapes["sonic_command"] == (4, DEFAULT_COMMAND_HORIZON, NUM_JOINTS * 2 + 6)
    assert shapes["action"] == (4, NUM_JOINTS)
