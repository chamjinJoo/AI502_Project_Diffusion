"""Diffusion Policy style conditional 1D U-Net denoiser.

This module is adapted from the public Diffusion Policy ConditionalUnet1D
implementation, simplified to avoid extra project dependencies and to fit this
project's [B, K, 65] humanoid tracking-reference chunks.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for integer diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed timesteps with shape [B] into [B, D]."""
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale)
        emb = x.float()[:, None] * freqs[None, :]  # [B, D/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, D]
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return emb


class Downsample1d(nn.Module):
    """Temporal strided convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Temporal transposed convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        groups = min(n_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual temporal block with timestep/history FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply conditioned residual block.

        Args:
            x: Temporal features with shape [B, C, K].
            cond: Conditioning vector with shape [B, D].
        """
        out = self.blocks[0](x)  # [B, C_out, K]
        embed = self.cond_encoder(cond)[:, :, None]  # [B, C_out or 2*C_out, 1]
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)  # [B, C_out, K]
        return out + self.residual_conv(x)


def _match_horizon(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Crop or pad temporal features to match a skip-connection horizon."""
    current = x.shape[-1]
    if current == horizon:
        return x
    if current > horizon:
        return x[..., :horizon]
    return F.pad(x, (0, horizon - current))


class ConditionalUnet1D(nn.Module):
    """Conditional 1D U-Net for epsilon prediction over [B, K, frame_dim]."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int | None = None,
        local_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        if not down_dims:
            raise ValueError("down_dims must contain at least one channel dimension")

        all_dims = [input_dim, *down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        cond_dim = diffusion_step_embed_dim + int(global_cond_dim or 0)

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        self.local_cond_encoder: nn.ModuleList | None = None
        if local_cond_dim is not None:
            _, first_dim = in_out[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim, first_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim, first_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                ]
            )

        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ]
        )

        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 2
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor | None = None,
        local_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise.

        Args:
            sample: Noisy future chunk with shape [B, K, frame_dim].
            timestep: Diffusion timesteps with shape [B].
            global_cond: Encoded history summary with shape [B, Dg].
            local_cond: Per-horizon context with shape [B, K, Dl].
        """
        x = sample.transpose(1, 2)  # [B, frame_dim, K]
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=sample.device).expand(sample.shape[0])  # [B]
        cond = self.diffusion_step_encoder(timestep)  # [B, D_t]
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)  # [B, D_t + Dg]

        local_features: list[torch.Tensor] = []
        if local_cond is not None:
            if self.local_cond_encoder is None:
                raise ValueError("local_cond was provided but local_cond_dim was not configured")
            local_x = local_cond.transpose(1, 2)  # [B, Dl, K]
            down_local, up_local = self.local_cond_encoder
            local_features.append(down_local(local_x, cond))  # [B, C0, K]
            local_features.append(up_local(local_x, cond))  # [B, C0, K]

        skips: list[torch.Tensor] = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, cond)
            if idx == 0 and local_features:
                x = x + _match_horizon(local_features[0], x.shape[-1])
            x = resnet2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        last_up_idx = len(self.up_modules) - 1
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            skip = skips.pop()
            x = _match_horizon(x, skip.shape[-1])
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, cond)
            if idx == last_up_idx and len(local_features) > 1:
                x = x + _match_horizon(local_features[1], x.shape[-1])
            x = resnet2(x, cond)
            x = upsample(x)

        x = _match_horizon(x, sample.shape[1])
        x = self.final_conv(x)  # [B, frame_dim, K]
        return x.transpose(1, 2)  # [B, K, frame_dim]
