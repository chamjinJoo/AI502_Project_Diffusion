"""Thin wrapper around the official diffusers DDIMScheduler.

The default objective is still DDIM/DDPM-style epsilon prediction. The wrapper
also supports a minimal rectified-flow objective for motion-generation
experiments while keeping the same denoiser input/output shapes.
"""

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
        joint_x0_loss_weight: float = 0.0,
        acceleration_loss_weight: float = 0.0,
        fps: float = 50.0,
        conditioning_mode: str = "history",
        objective: str = "epsilon",
        flow_solver: str = "euler",
        joint_vel_mode: str = "source",
        body_pos_mode: str = "relative",
    ) -> None:
        super().__init__()
        if prediction_type != "epsilon":
            raise ValueError("This project expects an epsilon-prediction denoiser.")
        if conditioning_mode not in {"history", "prefix"}:
            raise ValueError("conditioning_mode must be 'history' or 'prefix'")
        if objective not in {"epsilon", "rectified_flow"}:
            raise ValueError("objective must be 'epsilon' or 'rectified_flow'")
        if flow_solver not in {"euler", "heun"}:
            raise ValueError("flow_solver must be 'euler' or 'heun'")
        if joint_vel_mode not in {"source", "finite_difference"}:
            raise ValueError("joint_vel_mode must be 'source' or 'finite_difference'")
        if body_pos_mode not in {"relative", "delta"}:
            raise ValueError("body_pos_mode must be 'relative' or 'delta'")
        self.model = model
        self.velocity_loss_weight = velocity_loss_weight
        self.quaternion_loss_weight = quaternion_loss_weight
        self.continuity_loss_weight = continuity_loss_weight
        self.joint_x0_loss_weight = joint_x0_loss_weight
        self.acceleration_loss_weight = acceleration_loss_weight
        self.fps = float(fps)
        self.conditioning_mode = conditioning_mode
        self.objective = objective
        self.flow_solver = flow_solver
        self.joint_vel_mode = joint_vel_mode
        self.body_pos_mode = body_pos_mode
        self.register_buffer("norm_mean", torch.empty(0), persistent=False)
        self.register_buffer("norm_std", torch.empty(0), persistent=False)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=_diffusers_beta_schedule(beta_schedule),
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store z-score stats for physical/normalized auxiliary losses."""
        self.norm_mean = mean.float().view(1, 1, -1)
        self.norm_std = std.float().view(1, 1, -1).clamp_min(1e-6)

    def _has_normalization_stats(self, frame_dim: int = 65) -> bool:
        """Return whether per-dimension normalization stats are available."""
        return self.norm_mean.numel() == frame_dim and self.norm_std.numel() == frame_dim

    def _denormalize_if_available(self, chunk: torch.Tensor) -> torch.Tensor:
        """Convert normalized [B, T, 65] chunks back to tracking-reference units."""
        if self._has_normalization_stats(chunk.shape[-1]):
            mean = self.norm_mean.to(device=chunk.device, dtype=chunk.dtype)
            std = self.norm_std.to(device=chunk.device, dtype=chunk.dtype)
            return chunk * std + mean
        return chunk

    def _normalize_if_available(self, chunk: torch.Tensor) -> torch.Tensor:
        """Convert physical [B, T, 65] chunks into z-score units when stats exist."""
        if self._has_normalization_stats(chunk.shape[-1]):
            mean = self.norm_mean.to(device=chunk.device, dtype=chunk.dtype)
            std = self.norm_std.to(device=chunk.device, dtype=chunk.dtype)
            return (chunk - mean) / std
        return chunk

    def _normalize_joint_vel_if_available(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Normalize physical joint velocity [B, T, 29] when stats are available."""
        if not self._has_normalization_stats(65):
            return joint_vel
        mean = self.norm_mean[..., 29:58].to(device=joint_vel.device, dtype=joint_vel.dtype)
        std = self.norm_std[..., 29:58].to(device=joint_vel.device, dtype=joint_vel.dtype)
        if joint_vel.dim() == 2:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return (joint_vel - mean) / std

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

    def flow_time(self, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Map integer scheduler timesteps to continuous t in [0, 1].

        Shape:
            timesteps: [B]
            return: [B, 1, 1], broadcastable to [B, K, 65]
        """
        denom = max(1, self.num_train_timesteps - 1)
        return (timesteps.to(device=like.device, dtype=like.dtype) / float(denom)).view(-1, 1, 1)

    def interpolate_flow(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rectified-flow training pairs.

        x_t = (1 - t) * noise + t * x0, and target velocity = x0 - noise.
        All tensors have shape [B, K, 65].
        """
        t = self.flow_time(timesteps, x0)
        x_t = (1.0 - t) * noise + t * x0  # [B, K, 65]
        velocity = x0 - noise  # [B, K, 65]
        return x_t, velocity

    def predict_x0_from_flow(self, xt: torch.Tensor, timesteps: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Recover x0 from rectified-flow state and predicted velocity."""
        t = self.flow_time(timesteps, xt)
        return xt + (1.0 - t) * velocity  # [B, K, 65]

    def predict_x0_from_eps(self, xt: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover x0 from xt and predicted epsilon using scheduler alphas."""
        self._move_scheduler_tensors(xt.device, xt.dtype)
        alphas = self.scheduler.alphas_cumprod
        alpha_t = alphas[timesteps].view(-1, 1, 1)  # [B, 1, 1]
        return (xt - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t).clamp_min(1e-8)  # [B, K, 65]

    def _model_future_prediction(
        self,
        future_state: torch.Tensor,
        cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run the denoiser and return prediction for the future chunk only."""
        if self.conditioning_mode == "prefix":
            model_input = torch.cat([cond, future_state], dim=1)  # [B, H+K, 65]
            return self.model(
                model_input,
                None,
                timesteps,
                conditioning_mode="prefix",
                prefix_len=cond.shape[1],
            )  # [B, K, 65]
        return self.model(future_state, cond, timesteps)  # [B, K, 65]

    def velocity_consistency_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Compare predicted joint_vel to finite-difference velocity.

        When dataset normalization stats are available, the comparison is made
        in normalized velocity space so the auxiliary term is on a scale closer
        to the epsilon-prediction objective. Without stats, it falls back to
        physical tracking-reference units.
        """
        if x0_pred.shape[1] < 2:
            return x0_pred.new_tensor(0.0)
        x0_phys = self._denormalize_if_available(x0_pred)
        joint_pos_phys = x0_phys[:, :, :29]  # [B, K, 29]
        finite_diff_phys = (joint_pos_phys[:, 1:] - joint_pos_phys[:, :-1]) * self.fps  # [B, K-1, 29]

        if self._has_normalization_stats(65):
            joint_vel = x0_pred[:, :, 29:58]  # [B, K, 29], normalized
            finite_diff = self._normalize_joint_vel_if_available(finite_diff_phys)  # [B, K-1, 29]
        else:
            joint_vel = x0_phys[:, :, 29:58]  # [B, K, 29], physical
            finite_diff = finite_diff_phys
        return F.smooth_l1_loss(joint_vel[:, 1:], finite_diff)

    def quaternion_unit_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Encourage body_quat(w, x, y, z) to have unit norm."""
        x0_pred = self._denormalize_if_available(x0_pred)
        quat = x0_pred[:, :, 58:62]  # [B, K, 4]
        return ((quat.norm(dim=-1) - 1.0) ** 2).mean()

    def joint_x0_loss(self, x0_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Direct clean-sample loss on joint positions in normalized space."""
        return F.smooth_l1_loss(x0_pred[:, :, :29], target[:, :, :29])

    def acceleration_loss(self, x0_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match target second differences to reduce visual vibration.

        This is intentionally computed in normalized coordinate space without
        multiplying by fps**2. Physical accelerations are numerically large and
        can dominate the epsilon objective; normalized second differences keep
        this auxiliary term in a gentle regularization range.
        """
        if x0_pred.shape[1] < 3:
            return x0_pred.new_tensor(0.0)

        pred_joint_pos = x0_pred[:, :, :29]  # [B, K, 29], normalized
        target_joint_pos = target[:, :, :29]  # [B, K, 29], normalized
        pred_acc = pred_joint_pos[:, 2:] - 2.0 * pred_joint_pos[:, 1:-1] + pred_joint_pos[:, :-2]
        target_acc = target_joint_pos[:, 2:] - 2.0 * target_joint_pos[:, 1:-1] + target_joint_pos[:, :-2]
        loss = F.smooth_l1_loss(pred_acc, target_acc)

        pred_body_pos = x0_pred[:, :, 62:65]  # [B, K, 3], normalized
        target_body_pos = target[:, :, 62:65]  # [B, K, 3], normalized
        pred_body_acc = pred_body_pos[:, 2:] - 2.0 * pred_body_pos[:, 1:-1] + pred_body_pos[:, :-2]
        target_body_acc = target_body_pos[:, 2:] - 2.0 * target_body_pos[:, 1:-1] + target_body_pos[:, :-2]
        return loss + 0.25 * F.smooth_l1_loss(pred_body_acc, target_body_acc)

    def continuity_loss(self, x0_pred: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Encourage the first predicted frame to continue from the history.

        The model is trained in normalized space, so seam penalties are also
        computed in normalized units when stats are available. This keeps the
        auxiliary terms numerically comparable to the epsilon loss.
        """
        x0_phys = self._denormalize_if_available(x0_pred)
        cond_phys = self._denormalize_if_available(cond)
        x0_norm = self._normalize_if_available(x0_phys)

        last = cond_phys[:, -1]  # [B, 65]
        expected_phys = last.clone()  # [B, 65]
        expected_phys[:, :29] = last[:, :29] + last[:, 29:58] / self.fps  # [B, 29]
        expected_norm = self._normalize_if_available(expected_phys[:, None])[:, 0]  # [B, 65]
        loss = F.smooth_l1_loss(x0_norm[:, 0, :29], expected_norm[:, :29])

        # Velocity seam: first future velocity should agree with the history
        # velocity and with the first position step across the history/future boundary.
        expected_joint_vel = last[:, 29:58]  # [B, 29]
        seam_joint_vel = (x0_phys[:, 0, :29] - last[:, :29]) * self.fps  # [B, 29]
        expected_vel_norm = self._normalize_joint_vel_if_available(expected_joint_vel)
        seam_vel_norm = self._normalize_joint_vel_if_available(seam_joint_vel)
        pred_first_vel_norm = x0_norm[:, 0, 29:58]  # [B, 29]
        loss = loss + 0.25 * F.smooth_l1_loss(pred_first_vel_norm, expected_vel_norm)
        loss = loss + 0.25 * F.smooth_l1_loss(pred_first_vel_norm, seam_vel_norm)

        if self.body_pos_mode == "delta":
            expected_body_pos = cond_phys[:, -1, 62:65]  # [B, 3], keep root displacement continuous
            expected_body = last.clone()
            expected_body[:, 62:65] = expected_body_pos
            expected_body_norm = self._normalize_if_available(expected_body[:, None])[:, 0]
            loss = loss + 2.0 * F.smooth_l1_loss(x0_norm[:, 0, 62:65], expected_body_norm[:, 62:65])
        elif cond.shape[1] >= 2:
            root_delta = cond_phys[:, -1, 62:65] - cond_phys[:, -2, 62:65]  # [B, 3]
            expected_body_pos = cond_phys[:, -1, 62:65] + root_delta  # [B, 3]
            expected_body = last.clone()
            expected_body[:, 62:65] = expected_body_pos
            expected_body_norm = self._normalize_if_available(expected_body[:, None])[:, 0]
            loss = loss + 2.0 * F.smooth_l1_loss(x0_norm[:, 0, 62:65], expected_body_norm[:, 62:65])

        # In root-relative training, cond[-1] is close to identity orientation.
        # Penalizing the first future root quaternion helps avoid immediate pose flips.
        loss = loss + 0.25 * F.smooth_l1_loss(x0_phys[:, 0, 58:62], last[:, 58:62])
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

        if self.objective == "rectified_flow":
            denom = max(1, int(num_inference_steps))
            for step in range(denom):
                t_value = step / float(denom)
                timestep_int = round(t_value * float(max(1, self.num_train_timesteps - 1)))
                timestep_batch = torch.full((batch_size,), timestep_int, device=device, dtype=torch.long)  # [B]
                velocity = self._model_future_prediction(sample, cond, timestep_batch)  # [B, K, 65]
                dt = 1.0 / float(denom)
                if self.flow_solver == "heun" and step < denom - 1:
                    proposal = sample + dt * velocity  # [B, K, 65]
                    next_t = (step + 1) / float(denom)
                    next_timestep = round(next_t * float(max(1, self.num_train_timesteps - 1)))
                    next_batch = torch.full((batch_size,), next_timestep, device=device, dtype=torch.long)  # [B]
                    next_velocity = self._model_future_prediction(proposal, cond, next_batch)  # [B, K, 65]
                    sample = sample + 0.5 * dt * (velocity + next_velocity)  # Heun / improved Euler update
                else:
                    sample = sample + dt * velocity  # Euler update, future only
            return sample

        self._move_scheduler_tensors(device, dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            timestep_int = int(timestep.item())
            timestep_batch = torch.full((batch_size,), timestep_int, device=device, dtype=torch.long)  # [B]
            model_input = self.scheduler.scale_model_input(sample, timestep)
            eps_hat = self._model_future_prediction(model_input, cond, timestep_batch)  # [B, K, 65]
            sample = self.scheduler.step(eps_hat, timestep_int, sample, eta=eta).prev_sample  # [B, K, 65]
        return sample
