"""Motion history buffer for batched parallel environments."""

from __future__ import annotations

import torch

from diffusion_tracking_manager.constants import (
    DEFAULT_HISTORY_LEN,
    MOTION_FRAME_DIM,
    ROOT_QUAT_SLICE,
)
from diffusion_tracking_manager.motion_types import MotionState


class MotionHistoryBuffer:
    """Fixed-length [num_envs, history_len, 65] history with per-env reset support."""

    def __init__(
        self,
        num_envs: int,
        history_len: int = DEFAULT_HISTORY_LEN,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.device = torch.device(device)
        self.dtype = dtype
        self.buffer = torch.zeros(
            self.num_envs,
            self.history_len,
            MOTION_FRAME_DIM,
            device=self.device,
            dtype=self.dtype,
        )
        self.count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset()

    def reset(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
        state: MotionState | torch.Tensor | None = None,
    ) -> None:
        """Reset all or selected histories, optionally filling with the current state.

        Filling the complete window with the reset state lets the manager produce a
        valid 20-frame diffusion input immediately after an environment reset.
        """
        ids = self._normalize_env_ids(env_ids)
        if state is None:
            frame = torch.zeros(len(ids), MOTION_FRAME_DIM, device=self.device, dtype=self.dtype)
            frame[:, ROOT_QUAT_SLICE.start] = 1.0
        else:
            frame = self._state_to_frame(state, len(ids), env_ids=ids)
        self.buffer[ids] = frame[:, None, :].expand(-1, self.history_len, -1)
        self.count[ids] = self.history_len

    def append(self, state: MotionState | torch.Tensor) -> None:
        """Append one frame for every parallel environment."""
        frame = self._state_to_frame(state, self.num_envs)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1, :] = frame
        self.count = torch.clamp(self.count + 1, max=self.history_len)

    def history(self, flatten: bool = False) -> torch.Tensor:
        """Return the current history in [N, 20, 65] or flattened [N, 1300] form."""
        if flatten:
            return self.buffer.reshape(self.num_envs, -1)
        return self.buffer

    def _state_to_frame(
        self,
        state: MotionState | torch.Tensor,
        expected_num_envs: int,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(state, MotionState):
            frame = state.as_frame()
        else:
            frame = state
        frame = frame.to(device=self.device, dtype=self.dtype)
        if frame.dim() != 2 or frame.shape[-1] != MOTION_FRAME_DIM:
            raise ValueError(
                f"state frame must have shape [num_envs, {MOTION_FRAME_DIM}], got {tuple(frame.shape)}"
            )
        if env_ids is not None and frame.shape[0] == self.num_envs:
            frame = frame[env_ids]
        if frame.shape[0] != expected_num_envs:
            raise ValueError(f"state batch size must be {expected_num_envs}, got {frame.shape[0]}")
        return frame

    def _normalize_env_ids(self, env_ids: torch.Tensor | list[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError(f"env_ids must be 1D, got {tuple(ids.shape)}")
        if ids.numel() == 0:
            return ids
        if ids.min() < 0 or ids.max() >= self.num_envs:
            raise ValueError(f"env_ids out of range for num_envs={self.num_envs}: {ids.tolist()}")
        return ids
