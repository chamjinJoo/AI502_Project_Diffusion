"""Conditional denoisers for future motion chunks."""

from __future__ import annotations

import torch
from torch import nn

from .condition_encoder import ConditionEncoder
from .conditional_unet1d import ConditionalUnet1D
from .time_embedding import TimestepEmbedding, sinusoidal_positional_encoding


class TransformerDenoiser(nn.Module):
    """Transformer denoiser with optional MDM-style timestep token."""

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
        use_time_token: bool = False,
        use_segment_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.history_len = history_len
        self.pred_len = pred_len
        self.model_dim = model_dim
        self.use_time_token = use_time_token
        self.use_segment_embedding = use_segment_embedding

        self.target_proj = nn.Linear(frame_dim, model_dim)
        self.time_embed = TimestepEmbedding(model_dim)
        self.segment_embed = nn.Embedding(3, model_dim) if use_segment_embedding else None
        self.condition_encoder = ConditionEncoder(
            frame_dim=frame_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            encoder_type=condition_encoder,
        )
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

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor | None,
        timesteps: torch.Tensor,
        *,
        conditioning_mode: str = "history",
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """Predict eps_hat.

        Args:
            xt: History mode: noisy future [B, K, 65]. Prefix mode: clean
                prefix plus noisy future [B, H+K, 65].
            cond: History [B, H, 65] in history mode. Unused in prefix mode.
            timesteps: Diffusion steps with shape [B].
            conditioning_mode: "history" or "prefix".
            prefix_len: Number of clean prefix tokens in prefix mode.

        Returns:
            Predicted future noise with shape [B, K, 65].
        """
        D = self.model_dim
        time_emb = self.time_embed(timesteps)[:, None, :]  # [B, 1, D]

        if conditioning_mode == "prefix":
            prefix_len = self.history_len if prefix_len is None else int(prefix_len)
            _, sequence_len, _ = xt.shape
            if prefix_len <= 0 or prefix_len >= sequence_len:
                raise ValueError(f"prefix_len must be in (0, sequence_len), got {prefix_len} for {sequence_len}")
            tokens = self.target_proj(xt)  # [B, H+K, D]
            token_offset = 1 if self.use_time_token else 0
            tokens = tokens + sinusoidal_positional_encoding(sequence_len, D, xt.device, xt.dtype, offset=token_offset)

            if self.segment_embed is not None:
                # Segment ids: 0=timestep, 1=clean prefix, 2=noisy future.
                tokens[:, :prefix_len] = tokens[:, :prefix_len] + self.segment_embed.weight[1].view(1, 1, D)
                tokens[:, prefix_len:] = tokens[:, prefix_len:] + self.segment_embed.weight[2].view(1, 1, D)

            if self.use_time_token:
                time_token = time_emb + sinusoidal_positional_encoding(1, D, xt.device, xt.dtype, offset=0)
                if self.segment_embed is not None:
                    time_token = time_token + self.segment_embed.weight[0].view(1, 1, D)
                x = torch.cat([time_token, tokens], dim=1)  # [B, 1+H+K, D]
                future_start = 1 + prefix_len
            else:
                x = tokens + time_emb  # [B, H+K, D]
                future_start = prefix_len

            x = self.backbone(x)  # [B, sequence, D]
            return self.output(x[:, future_start:])  # [B, K, 65]

        if conditioning_mode != "history":
            raise ValueError("conditioning_mode must be 'history' or 'prefix'")
        if cond is None:
            raise ValueError("cond is required in history conditioning mode")

        _, K, _ = xt.shape
        H = self.history_len

        cond_tokens = self.condition_encoder(cond)  # [B, H, D]
        target_tokens = self.target_proj(xt)  # [B, K, D]

        if self.use_time_token:
            # MDM-style sequence: [diffusion timestep token, history tokens, target tokens].
            cond_offset = 1
            target_offset = 1 + H
            time_pe = sinusoidal_positional_encoding(1, D, xt.device, xt.dtype, offset=0)
            time_token = time_emb + time_pe  # [B, 1, D]
        else:
            # Backward-compatible sequence used by older checkpoints.
            cond_offset = 0
            target_offset = H
            time_token = None

        cond_pe = sinusoidal_positional_encoding(H, D, cond.device, cond.dtype, offset=cond_offset)
        target_pe = sinusoidal_positional_encoding(K, D, xt.device, xt.dtype, offset=target_offset)
        cond_tokens = cond_tokens + cond_pe  # [B, H, D]
        target_tokens = target_tokens + target_pe  # [B, K, D]

        if self.segment_embed is not None:
            # Segment ids: 0=timestep, 1=history, 2=target.
            cond_tokens = cond_tokens + self.segment_embed.weight[1].view(1, 1, D)
            target_tokens = target_tokens + self.segment_embed.weight[2].view(1, 1, D)
            if time_token is not None:
                time_token = time_token + self.segment_embed.weight[0].view(1, 1, D)

        if time_token is not None:
            x = torch.cat([time_token, cond_tokens, target_tokens], dim=1)  # [B, 1+H+K, D]
            target_start = 1 + H
        else:
            x = torch.cat([cond_tokens, target_tokens], dim=1)  # [B, H+K, D]
            x = x + time_emb  # old behavior: broadcast timestep to all tokens
            target_start = H

        x = self.backbone(x)  # [B, sequence, D]
        return self.output(x[:, target_start:])  # [B, K, 65]


class UnetDenoiser(nn.Module):
    """Diffusion Policy style U-Net denoiser with history conditioning."""

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
        use_local_condition: bool = False,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.pred_len = pred_len
        self.condition_summary_type = condition_summary
        self.use_local_condition = use_local_condition
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
            local_cond_dim=model_dim if use_local_condition else None,
            diffusion_step_embed_dim=model_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor | None,
        timesteps: torch.Tensor,
        *,
        conditioning_mode: str = "history",
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """Predict eps_hat.

        Args:
            xt: Noisy future chunk with shape [B, K, 65].
            cond: Motion history with shape [B, H, 65].
            timesteps: Diffusion steps with shape [B].

        Returns:
            Predicted noise with shape [B, K, 65].
        """
        del prefix_len
        if conditioning_mode != "history":
            raise ValueError("prefix conditioning is currently implemented for the transformer denoiser only")
        if cond is None:
            raise ValueError("cond is required in history conditioning mode")
        cond_tokens = self.condition_encoder(cond)  # [B, H, D]
        if self.condition_summary_type == "mean":
            global_cond = self.cond_summary(cond_tokens.mean(dim=1))  # [B, D]
        else:
            global_cond = self.cond_summary(cond_tokens)  # [B, D]

        local_cond = None
        if self.use_local_condition:
            # Diffusion Policy style local conditioning: inject recent history features
            # into early U-Net blocks after aligning them to the target horizon.
            local_cond = cond_tokens[:, -self.pred_len :]  # [B, <=K, D]
            if local_cond.shape[1] < xt.shape[1]:
                pad = local_cond[:, -1:].expand(-1, xt.shape[1] - local_cond.shape[1], -1)
                local_cond = torch.cat([local_cond, pad], dim=1)
            local_cond = local_cond[:, : xt.shape[1]]  # [B, K, D]
        return self.unet(xt, timesteps, global_cond=global_cond, local_cond=local_cond)  # [B, K, 65]


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
        use_time_token: bool = False,
        use_segment_embedding: bool = False,
        use_local_condition: bool = False,
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
                use_time_token=use_time_token,
                use_segment_embedding=use_segment_embedding,
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
                use_local_condition=use_local_condition,
            )
        else:
            raise ValueError(f"unknown denoiser architecture: {architecture}")

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor | None,
        timesteps: torch.Tensor,
        *,
        conditioning_mode: str = "history",
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """Predict future eps_hat with shape [B, K, 65]."""
        return self.net(xt, cond, timesteps, conditioning_mode=conditioning_mode, prefix_len=prefix_len)
