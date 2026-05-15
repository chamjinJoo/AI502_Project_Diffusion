"""Conditional denoisers for future motion chunks."""

from __future__ import annotations

import torch
from torch import nn

from .condition_encoder import ConditionEncoder
from .conditional_unet1d import ConditionalUnet1D
from .time_embedding import TimestepEmbedding


class TransformerDenoiser(nn.Module):
    """Small Transformer baseline for GR00T tracking-reference chunks."""

    def __init__(
        self,
        frame_dim: int = 65,
        history_len: int = 20,
        pred_len: int = 10,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        condition_encoder: str = "transformer",
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.history_len = history_len
        self.pred_len = pred_len

        self.target_proj = nn.Linear(frame_dim, model_dim)
        self.time_embed = TimestepEmbedding(model_dim)
        self.condition_encoder = ConditionEncoder(
            frame_dim=frame_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            encoder_type=condition_encoder,
        )
        self.cond_summary = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, frame_dim))

    def forward(self, xt: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict eps_hat.

        Args:
            xt: Noisy future chunk with shape [B, K, 65].
            cond: Motion history with shape [B, H, 65].
            timesteps: Diffusion steps with shape [B].

        Returns:
            Predicted noise with shape [B, K, 65].
        """
        target_tokens = self.target_proj(xt)  # [B, K, D]
        time_tokens = self.time_embed(timesteps)[:, None, :]  # [B, 1, D]
        cond_tokens = self.condition_encoder(cond)  # [B, H, D]
        cond_token = self.cond_summary(cond_tokens.mean(dim=1, keepdim=True))  # [B, 1, D]

        x = target_tokens + time_tokens + cond_token  # [B, K, D]
        x = self.backbone(x)  # [B, K, D]
        return self.output(x)  # [B, K, 65]


class UnetDenoiser(nn.Module):
    """Diffusion Policy style U-Net denoiser with global history conditioning."""

    def __init__(
        self,
        frame_dim: int = 65,
        history_len: int = 20,
        pred_len: int = 10,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        condition_encoder: str = "transformer",
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        condition_summary: str = "flatten",
    ) -> None:
        super().__init__()
        del pred_len
        self.frame_dim = frame_dim
        self.condition_summary_type = condition_summary
        self.condition_encoder = ConditionEncoder(
            frame_dim=frame_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            encoder_type=condition_encoder,
        )
        if condition_summary == "mean":
            self.cond_summary = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        elif condition_summary == "flatten":
            self.cond_summary = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(history_len * model_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
        else:
            raise ValueError("condition_summary must be 'mean' or 'flatten'")
        self.unet = ConditionalUnet1D(
            input_dim=frame_dim,
            global_cond_dim=model_dim,
            diffusion_step_embed_dim=model_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(self, xt: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict eps_hat.

        Args:
            xt: Noisy future chunk with shape [B, K, 65].
            cond: Motion history with shape [B, H, 65].
            timesteps: Diffusion steps with shape [B].

        Returns:
            Predicted noise with shape [B, K, 65].
        """
        cond_tokens = self.condition_encoder(cond)  # [B, H, D]
        if self.condition_summary_type == "mean":
            global_cond = self.cond_summary(cond_tokens.mean(dim=1))  # [B, D]
        else:
            global_cond = self.cond_summary(cond_tokens)  # [B, D]
        return self.unet(xt, timesteps, global_cond)  # [B, K, 65]


class ConditionalDenoiser(nn.Module):
    """Compatibility wrapper selecting the configured denoiser architecture."""

    def __init__(
        self,
        frame_dim: int = 65,
        history_len: int = 20,
        pred_len: int = 10,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        condition_encoder: str = "transformer",
        architecture: str = "unet",
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        condition_summary: str = "flatten",
    ) -> None:
        super().__init__()
        if architecture == "transformer":
            self.net = TransformerDenoiser(
                frame_dim=frame_dim,
                history_len=history_len,
                pred_len=pred_len,
                model_dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                condition_encoder=condition_encoder,
            )
        elif architecture == "unet":
            self.net = UnetDenoiser(
                frame_dim=frame_dim,
                history_len=history_len,
                pred_len=pred_len,
                model_dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                condition_encoder=condition_encoder,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                condition_summary=condition_summary,
            )
        else:
            raise ValueError(f"unknown denoiser architecture: {architecture}")

    def forward(self, xt: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict eps_hat with shape [B, K, 65]."""
        return self.net(xt, cond, timesteps)
