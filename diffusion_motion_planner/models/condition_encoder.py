"""History encoders for conditioning on recent motion."""

from __future__ import annotations

import torch
from torch import nn


class LinearConditionEncoder(nn.Module):
    """Frame-wise projection with no temporal mixing before the denoiser."""

    def __init__(self, frame_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(frame_dim, model_dim), nn.LayerNorm(model_dim))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)  # [B, H, D]


class ConvConditionEncoder(nn.Module):
    """Small Conv1D temporal encoder for condition history."""

    def __init__(self, frame_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(frame_dim, model_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, H, 65] -> channels-first: [B, 65, H]
        x = cond.transpose(1, 2)
        x = self.net(x)  # [B, D, H]
        return x.transpose(1, 2)  # [B, H, D]


class TransformerConditionEncoder(nn.Module):
    """Transformer encoder for condition history."""

    def __init__(self, frame_dim: int, model_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(frame_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers // 2))
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(cond)  # [B, H, D]
        return self.norm(self.encoder(x))  # [B, H, D]


class ConditionEncoder(nn.Module):
    """Dispatch wrapper for linear, Conv1D, or Transformer history encoders."""

    def __init__(
        self,
        frame_dim: int = 65,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
    ) -> None:
        super().__init__()
        if encoder_type in {"linear", "raw"}:
            self.encoder = LinearConditionEncoder(frame_dim, model_dim)
        elif encoder_type == "conv":
            self.encoder = ConvConditionEncoder(frame_dim, model_dim, dropout)
        elif encoder_type == "transformer":
            self.encoder = TransformerConditionEncoder(frame_dim, model_dim, num_layers, num_heads, dropout)
        else:
            raise ValueError("encoder_type must be 'linear', 'raw', 'conv', or 'transformer'")

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.encoder(cond)  # [B, H, D]
