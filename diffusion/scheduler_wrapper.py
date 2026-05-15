"""Thin wrapper around the official diffusers DDIMScheduler."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch import nn


def _diffusers_beta_schedule(name: str) -> str:
    """Map small project config names to diffusers schedule names."""
    if name == "cosine":
        return "squaredcos_cap_v2"
    return name


class DiffusionSchedulerWrapper(nn.Module):
    """DDIMScheduler adapter for [B, K, 65] epsilon-prediction training and sampling."""

    def __init__(
        self,
        model: nn.Module,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        velocity_loss_weight: float = 0.0,
        quaternion_loss_weight: float = 0.0,
        continuity_loss_weight: float = 0.0,
        fps: float = 50.0,
    ) -> None:
        super().__init__()
        if prediction_type != "epsilon":
            raise ValueError("This project expects an epsilon-prediction denoiser.")
        self.model = model
        self.velocity_loss_weight = velocity_loss_weight
        self.quaternion_loss_weight = quaternion_loss_weight
        self.continuity_loss_weight = continuity_loss_weight
        self.fps = float(fps)
        self.register_buffer("norm_mean", torch.empty(0), persistent=False)
        self.register_buffer("norm_std", torch.empty(0), persistent=False)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=_diffusers_beta_schedule(beta_schedule),
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store z-score stats so auxiliary losses can run in physical units."""
        self.norm_mean = mean.float().view(1, 1, -1)
        self.norm_std = std.float().view(1, 1, -1).clamp_min(1e-6)

    def _denormalize_if_available(self, chunk: torch.Tensor) -> torch.Tensor:
        """Convert normalized [B, T, 65] chunks back to tracking-reference units."""
        if self.norm_mean.numel() == chunk.shape[-1] and self.norm_std.numel() == chunk.shape[-1]:
            mean = self.norm_mean.to(device=chunk.device, dtype=chunk.dtype)
            std = self.norm_std.to(device=chunk.device, dtype=chunk.dtype)
            return chunk * std + mean
        return chunk

    def _move_scheduler_tensors(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move scheduler tensors because diffusers schedulers are not nn.Modules."""
        for name in ("alphas_cumprod", "final_alpha_cumprod"):
            value = getattr(self.scheduler, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.scheduler, name, value.to(device=device, dtype=dtype))

    @property
    def num_train_timesteps(self) -> int:
        """Number of training diffusion timesteps."""
        return int(self.scheduler.config.num_train_timesteps)

    def scheduler_config(self) -> dict[str, Any]:
        """Return a serializable scheduler config for checkpoints."""
        return dict(self.scheduler.config)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample integer timesteps for DDPM-style epsilon training.

        Returns:
            Tensor with shape [B].
        """
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Use DDIMScheduler forward diffusion to create xt.

        Args:
            x0: Clean future chunk with shape [B, K, 65].
            noise: Gaussian noise with shape [B, K, 65].
            timesteps: Integer timesteps with shape [B].
        """
        self._move_scheduler_tensors(x0.device, x0.dtype)
        return self.scheduler.add_noise(x0, noise, timesteps)  # [B, K, 65]

    def predict_x0_from_eps(self, xt: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover x0 from xt and predicted epsilon using scheduler alphas."""
        self._move_scheduler_tensors(xt.device, xt.dtype)
        alphas = self.scheduler.alphas_cumprod
        alpha_t = alphas[timesteps].view(-1, 1, 1)  # [B, 1, 1]
        return (xt - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t).clamp_min(1e-8)  # [B, K, 65]

    def velocity_consistency_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Compare joint_vel to finite differences of joint_pos in physical units."""
        if x0_pred.shape[1] < 2:
            return x0_pred.new_tensor(0.0)
        x0_pred = self._denormalize_if_available(x0_pred)
        joint_pos = x0_pred[:, :, :29]  # [B, K, 29]
        joint_vel = x0_pred[:, :, 29:58]  # [B, K, 29]
        finite_diff = (joint_pos[:, 1:] - joint_pos[:, :-1]) * self.fps  # [B, K-1, 29]
        return F.smooth_l1_loss(joint_vel[:, 1:], finite_diff)

    def quaternion_unit_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Encourage body_quat(w, x, y, z) to have unit norm."""
        x0_pred = self._denormalize_if_available(x0_pred)
        quat = x0_pred[:, :, 58:62]  # [B, K, 4]
        return ((quat.norm(dim=-1) - 1.0) ** 2).mean()

    def continuity_loss(self, x0_pred: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Encourage the first predicted frame to continue from the history."""
        x0_pred = self._denormalize_if_available(x0_pred)
        cond = self._denormalize_if_available(cond)

        last = cond[:, -1]  # [B, 65]
        expected_joint_pos = last[:, :29] + last[:, 29:58] / self.fps  # [B, 29]
        loss = F.smooth_l1_loss(x0_pred[:, 0, :29], expected_joint_pos)

        if cond.shape[1] >= 2:
            root_delta = cond[:, -1, 62:65] - cond[:, -2, 62:65]  # [B, 3]
            expected_body_pos = cond[:, -1, 62:65] + root_delta  # [B, 3]
            loss = loss + 0.1 * F.smooth_l1_loss(x0_pred[:, 0, 62:65], expected_body_pos)
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        pred_len: int,
        frame_dim: int = 65,
        num_inference_steps: int = 20,
        x_T: torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run deterministic DDIM sampling.

        Args:
            cond: Condition history with shape [B, H, 65].
            pred_len: Prediction horizon K.
            frame_dim: Frame dimension, fixed to 65 for this project.
            num_inference_steps: Number of DDIM reverse steps.
            x_T: Optional initial latent/noise with shape [B, K, 65].
            eta: DDIM stochasticity. Keep 0.0 for deterministic DDIM.

        Returns:
            Predicted future chunk with shape [B, K, 65].
        """
        device = cond.device
        dtype = cond.dtype
        batch_size = cond.shape[0]
        sample = x_T if x_T is not None else torch.randn(batch_size, pred_len, frame_dim, device=device, dtype=dtype)
        sample = sample.to(device=device, dtype=dtype)  # [B, K, 65]
        if sample.shape != (batch_size, pred_len, frame_dim):
            raise ValueError(f"x_T must have shape [{batch_size}, {pred_len}, {frame_dim}], got {tuple(sample.shape)}")

        self._move_scheduler_tensors(device, dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            timestep_int = int(timestep.item())
            timestep_batch = torch.full((batch_size,), timestep_int, device=device, dtype=torch.long)  # [B]
            model_input = self.scheduler.scale_model_input(sample, timestep)
            eps_hat = self.model(model_input, cond, timestep_batch)  # [B, K, 65]
            sample = self.scheduler.step(eps_hat, timestep_int, sample, eta=eta).prev_sample  # [B, K, 65]
        return sample
