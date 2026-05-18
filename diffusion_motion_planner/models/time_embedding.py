"""Sinusoidal diffusion timestep and positional embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_positional_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    """Compute fixed sinusoidal positional encoding for a sequence.

    Args:
        length: Number of positions.
        dim: Embedding dimension.
        device: Target device.
        dtype: Target dtype.
        offset: Starting position index (use to align history and target segments).

    Returns:
        Tensor with shape [1, length, dim].
    """
    half = dim // 2
    positions = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = positions[:, None] * freqs[None, :]  # [length, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [length, dim]
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0).to(dtype=dtype)  # [1, length, dim]


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for integer timesteps.

    Args:
        timesteps: Tensor with shape [B].
        dim: Embedding dimension.

    Returns:
        Tensor with shape [B, dim].
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None, :]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2 * half]
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(timesteps, self.dim)  # [B, dim]
        return self.mlp(emb)  # [B, dim]
