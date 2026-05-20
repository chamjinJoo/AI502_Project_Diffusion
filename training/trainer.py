"""Training and validation loops for the conditional DDIM planner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Trainer:
    """Small research-friendly trainer."""

    def __init__(
        self,
        diffusion: torch.nn.Module,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]] | None,
        cfg: dict[str, Any],
    ) -> None:
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        requested_device = cfg["training"].get("device", "cuda")
        self.device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
        self.diffusion.to(self.device)

        train_cfg = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.diffusion.model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, int(train_cfg["epochs"]))
        )
        self.grad_clip = float(train_cfg["grad_clip"])
        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val = float("inf")
        self.start_epoch = 0
        self.wandb = None
        if train_cfg.get("use_wandb", False):
            import wandb

            self.wandb = wandb
            self.wandb.init(project="conditional-ddim-planner", config=cfg)
        if train_cfg.get("resume", False):
            self._try_resume()

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _compute_losses(self, cond: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute scheduler-based epsilon training losses.

        Args:
            cond: Motion history with shape [B, H, 65].
            target: Clean future chunk with shape [B, K, 65].
        """
        batch_size = target.shape[0]
        timesteps = self.diffusion.sample_timesteps(batch_size, target.device)  # [B]
        noise = torch.randn_like(target)  # [B, K, 65]
        if self.diffusion.objective == "rectified_flow":
            xt, objective_target = self.diffusion.interpolate_flow(target, noise, timesteps)  # [B, K, 65]
            pred = self.diffusion._model_future_prediction(xt, cond, timesteps)  # [B, K, 65], velocity field
            noise_loss = F.mse_loss(pred, objective_target)
            x0_pred = self.diffusion.predict_x0_from_flow(xt, timesteps, pred)  # [B, K, 65]
        else:
            xt = self.diffusion.add_noise(target, noise, timesteps)  # [B, K, 65], create noisy future with scheduler
            pred = self.diffusion._model_future_prediction(xt, cond, timesteps)  # [B, K, 65], epsilon
            noise_loss = F.mse_loss(pred, noise)
            x0_pred = self.diffusion.predict_x0_from_eps(xt, timesteps, pred)  # [B, K, 65]
        aux_max_timestep = self.cfg["training"].get("auxiliary_max_timestep")
        if aux_max_timestep is not None:
            aux_mask = timesteps <= int(aux_max_timestep)
            x0_aux = x0_pred[aux_mask]
            cond_aux = cond[aux_mask]
        else:
            x0_aux = x0_pred
            cond_aux = cond
        if aux_max_timestep is not None:
            target_aux = target[aux_mask]
        else:
            target_aux = target
        if x0_aux.shape[0] == 0:
            vel_loss = x0_pred.new_tensor(0.0)
            quat_loss = x0_pred.new_tensor(0.0)
            continuity_loss = x0_pred.new_tensor(0.0)
            joint_x0_loss = x0_pred.new_tensor(0.0)
            acceleration_loss = x0_pred.new_tensor(0.0)
        else:
            vel_loss = self.diffusion.velocity_consistency_loss(x0_aux)
            quat_loss = self.diffusion.quaternion_unit_loss(x0_aux)
            continuity_loss = self.diffusion.continuity_loss(x0_aux, cond_aux)
            joint_x0_loss = self.diffusion.joint_x0_loss(x0_aux, target_aux)
            acceleration_loss = self.diffusion.acceleration_loss(x0_aux, target_aux)
        total = (
            noise_loss
            + self.diffusion.velocity_loss_weight * vel_loss
            + self.diffusion.quaternion_loss_weight * quat_loss
            + self.diffusion.continuity_loss_weight * continuity_loss
            + self.diffusion.joint_x0_loss_weight * joint_x0_loss
            + self.diffusion.acceleration_loss_weight * acceleration_loss
        )
        return {
            "loss": total,
            "noise_loss": noise_loss,
            "velocity_loss": vel_loss,
            "quaternion_loss": quat_loss,
            "continuity_loss": continuity_loss,
            "joint_x0_loss": joint_x0_loss,
            "acceleration_loss": acceleration_loss,
        }

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        val_loss: float,
        selection_loss: float | None = None,
    ) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.diffusion.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "val_loss": val_loss,
                "selection_loss": val_loss if selection_loss is None else selection_loss,
                "best_val": self.best_val,
                "config": self.cfg,
                "scheduler_config": self.diffusion.scheduler_config(),
            },
            path,
        )

    def _try_resume(self) -> None:
        """Resume from latest.pt if it exists in the checkpoint directory."""
        path = self.checkpoint_dir / "latest.pt"
        if not path.exists():
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self._validate_resume_compatibility(checkpoint, path)
        self.diffusion.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", self.best_val)))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        print(f"[resume] loaded {path} at epoch {self.start_epoch}", flush=True)

    def _validate_resume_compatibility(self, checkpoint: dict[str, Any], path: Path) -> None:
        """Reject checkpoint resumes that would mix incompatible model/data configs."""
        checkpoint_cfg = checkpoint.get("config")
        if not isinstance(checkpoint_cfg, dict):
            return
        if "model" not in checkpoint_cfg or "model" not in self.cfg:
            return
        current_model = self.cfg.get("model", {})
        previous_model = checkpoint_cfg.get("model", {})
        current_arch = str(current_model.get("architecture", "unet"))
        previous_arch = str(previous_model.get("architecture", "transformer"))
        current_mode = str(current_model.get("conditioning_mode", self.cfg.get("conditioning_mode", "history")))
        previous_mode = str(previous_model.get("conditioning_mode", checkpoint_cfg.get("conditioning_mode", "history")))
        if current_mode != previous_mode:
            raise RuntimeError(
                f"Refusing to resume {path}: checkpoint conditioning_mode={previous_mode!r}, "
                f"current conditioning_mode={current_mode!r}. Use a separate checkpoint_dir or set resume: false."
            )
        if current_arch != previous_arch:
            raise RuntimeError(
                f"Refusing to resume {path}: checkpoint architecture={previous_arch!r}, "
                f"current architecture={current_arch!r}. Use a separate checkpoint_dir or set resume: false."
            )
        if current_arch == "transformer":
            for key, default in (
                ("condition_encoder", "transformer"),
                ("dim", None),
                ("num_layers", None),
                ("num_heads", None),
                ("use_time_token", False),
                ("use_segment_embedding", False),
            ):
                current_value = current_model.get(key, default)
                previous_value = previous_model.get(key, default)
                if current_value != previous_value:
                    raise RuntimeError(
                        f"Refusing to resume {path}: checkpoint model.{key}={previous_value!r}, "
                        f"current model.{key}={current_value!r}."
                    )
        elif current_arch == "unet":
            for key, default in (
                ("condition_encoder", "transformer"),
                ("condition_summary", "flatten"),
                ("use_local_condition", False),
                ("cond_predict_scale", False),
                ("dim", None),
                ("down_dims", None),
            ):
                current_value = current_model.get(key, default)
                previous_value = previous_model.get(key, default)
                if current_value != previous_value:
                    raise RuntimeError(
                        f"Refusing to resume {path}: checkpoint model.{key}={previous_value!r}, "
                        f"current model.{key}={current_value!r}."
                    )

        current_diffusion = self.cfg.get("diffusion", {})
        previous_diffusion = checkpoint_cfg.get("diffusion", {})
        current_objective = str(current_diffusion.get("objective", "epsilon"))
        previous_objective = str(previous_diffusion.get("objective", "epsilon"))
        if current_objective != previous_objective:
            raise RuntimeError(
                f"Refusing to resume {path}: checkpoint diffusion.objective={previous_objective!r}, "
                f"current diffusion.objective={current_objective!r}. Use a separate checkpoint_dir or set resume: false."
            )
        current_flow_solver = str(current_diffusion.get("flow_solver", "euler"))
        previous_flow_solver = str(previous_diffusion.get("flow_solver", "euler"))
        if current_objective == "rectified_flow" and current_flow_solver != previous_flow_solver:
            raise RuntimeError(
                f"Refusing to resume {path}: checkpoint diffusion.flow_solver={previous_flow_solver!r}, "
                f"current diffusion.flow_solver={current_flow_solver!r}. Use a separate checkpoint_dir or set resume: false."
            )

        current_data = self.cfg.get("data", {})
        previous_data = checkpoint_cfg.get("data", {})
        for key in ("frame_dim", "history_len", "pred_len"):
            if key in current_data and key in previous_data and int(current_data[key]) != int(previous_data[key]):
                raise RuntimeError(
                    f"Refusing to resume {path}: checkpoint data.{key}={previous_data[key]}, "
                    f"current data.{key}={current_data[key]}."
                )
        for key, default in (("root_relative", False), ("joint_vel_mode", "source"), ("body_pos_mode", "relative")):
            current_value = current_data.get(key, default)
            previous_value = previous_data.get(key, default)
            if current_value != previous_value:
                raise RuntimeError(
                    f"Refusing to resume {path}: checkpoint data.{key}={previous_value!r}, "
                    f"current data.{key}={current_value!r}. Use a separate checkpoint_dir or set resume: false."
                )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.diffusion.train()
        totals: dict[str, float] = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "velocity_loss": 0.0,
            "quaternion_loss": 0.0,
            "continuity_loss": 0.0,
            "joint_x0_loss": 0.0,
            "acceleration_loss": 0.0,
        }
        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self._compute_losses(batch["cond"], batch["target"])
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {
                "loss": float("nan"),
                "noise_loss": float("nan"),
                "velocity_loss": float("nan"),
                "quaternion_loss": float("nan"),
                "continuity_loss": float("nan"),
                "joint_x0_loss": float("nan"),
                "acceleration_loss": float("nan"),
            }
        self.diffusion.eval()
        totals: dict[str, float] = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "velocity_loss": 0.0,
            "quaternion_loss": 0.0,
            "continuity_loss": 0.0,
            "joint_x0_loss": 0.0,
            "acceleration_loss": 0.0,
        }
        for batch in self.val_loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self._compute_losses(batch["cond"], batch["target"])
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
        return {key: value / len(self.val_loader) for key, value in totals.items()}

    @torch.no_grad()
    def sampled_validation(self, epoch: int) -> dict[str, float]:
        """Run a small deterministic DDIM validation sample.

        This metric is intentionally lightweight and optional. It catches cases
        where epsilon MSE looks fine but sampled futures have poor seam/root
        continuity. Values are computed in normalized space except seam RMSEs.
        """
        train_cfg = self.cfg["training"]
        interval = int(train_cfg.get("sample_eval_interval", 0))
        max_batches = int(train_cfg.get("sample_eval_batches", 0))
        if self.val_loader is None or interval <= 0 or max_batches <= 0 or epoch % interval != 0:
            return {}

        self.diffusion.eval()
        steps = int(train_cfg.get("sample_eval_num_inference_steps", self.cfg["diffusion"].get("num_inference_steps", 20)))
        pred_len = int(self.cfg["data"]["pred_len"])
        frame_dim = int(self.cfg["data"]["frame_dim"])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.cfg.get("seed", 0)) + epoch * 9973)

        future_mse = 0.0
        joint_seam_mse = 0.0
        body_seam_mse = 0.0
        count = 0
        for batch in self.val_loader:
            batch = self._move_batch(batch)
            cond = batch["cond"]  # [B, H, 65], normalized
            target = batch["target"]  # [B, K, 65], normalized
            x_t = torch.randn(
                target.shape[0], pred_len, frame_dim,
                device=self.device, dtype=target.dtype, generator=generator,
            )
            pred = self.diffusion.sample(cond, pred_len, frame_dim, steps, x_T=x_t, eta=0.0)
            future_mse += float(F.mse_loss(pred, target).detach().cpu())

            pred_phys = self.diffusion._denormalize_if_available(pred)
            cond_phys = self.diffusion._denormalize_if_available(cond)
            last = cond_phys[:, -1]
            expected_joint_pos = last[:, :29] + last[:, 29:58] / self.diffusion.fps
            joint_seam_mse += float(F.mse_loss(pred_phys[:, 0, :29], expected_joint_pos).detach().cpu())
            if getattr(self.diffusion, "body_pos_mode", "relative") == "delta":
                expected_body_pos = last[:, 62:65]
            elif cond_phys.shape[1] >= 2:
                expected_body_pos = last[:, 62:65] + (last[:, 62:65] - cond_phys[:, -2, 62:65])
            else:
                expected_body_pos = last[:, 62:65]
            body_seam_mse += float(F.mse_loss(pred_phys[:, 0, 62:65], expected_body_pos).detach().cpu())
            count += 1
            if count >= max_batches:
                break

        if count == 0:
            return {}
        return {
            "sample_future_mse": future_mse / count,
            "sample_joint_seam_rmse": math.sqrt(joint_seam_mse / count),
            "sample_body_seam_rmse": math.sqrt(body_seam_mse / count),
        }

    def fit(self) -> None:
        epochs = int(self.cfg["training"]["epochs"])
        for epoch in range(self.start_epoch + 1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            val_loss = val_metrics["loss"]
            latest_val = val_loss if not math.isnan(val_loss) else train_metrics["loss"]
            sample_metrics = self.sampled_validation(epoch)
            sample_weight = float(self.cfg["training"].get("sample_eval_weight", 0.0))
            selection_loss = latest_val + sample_weight * sample_metrics.get("sample_future_mse", 0.0)
            sample_interval = int(self.cfg["training"].get("sample_eval_interval", 0))
            require_sample_for_best = sample_weight > 0.0 and sample_interval > 0
            can_update_best = (not require_sample_for_best) or bool(sample_metrics)
            is_best = can_update_best and selection_loss < self.best_val
            if is_best:
                self.best_val = selection_loss
            self._save_checkpoint("latest.pt", epoch, latest_val, selection_loss)
            if is_best:
                self._save_checkpoint("best.pt", epoch, latest_val, selection_loss)

            msg = (
                f"epoch {epoch:04d} | train {train_metrics['loss']:.6f} "
                f"| val {val_metrics['loss']:.6f} | noise {train_metrics['noise_loss']:.6f} "
                f"| vel {train_metrics['velocity_loss']:.6f} "
                f"| quat {train_metrics['quaternion_loss']:.6f} "
                f"| cont {train_metrics['continuity_loss']:.6f} "
                f"| x0j {train_metrics['joint_x0_loss']:.6f} "
                f"| acc {train_metrics['acceleration_loss']:.6f}"
            )
            if sample_metrics:
                msg += (
                    f" | sample_mse {sample_metrics['sample_future_mse']:.6f}"
                    f" | sample_joint_seam {sample_metrics['sample_joint_seam_rmse']:.4f}"
                    f" | sample_body_seam {sample_metrics['sample_body_seam_rmse']:.4f}"
                    f" | select {selection_loss:.6f}"
                )
            print(msg, flush=True)
            if self.wandb is not None:
                log_metrics = (
                    {f"train/{k}": v for k, v in train_metrics.items()}
                    | {f"val/{k}": v for k, v in val_metrics.items()}
                    | {f"sample/{k}": v for k, v in sample_metrics.items()}
                    | {"val/selection_loss": selection_loss}
                )
                self.wandb.log(log_metrics, step=epoch)
