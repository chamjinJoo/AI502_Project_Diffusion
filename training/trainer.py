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
        xt = self.diffusion.add_noise(target, noise, timesteps)  # [B, K, 65]
        eps_hat = self.diffusion.model(xt, cond, timesteps)  # [B, K, 65]
        noise_loss = F.mse_loss(eps_hat, noise)

        x0_pred = self.diffusion.predict_x0_from_eps(xt, timesteps, eps_hat)  # [B, K, 65]
        aux_max_timestep = self.cfg["training"].get("auxiliary_max_timestep")
        if aux_max_timestep is not None:
            aux_mask = timesteps <= int(aux_max_timestep)
            x0_aux = x0_pred[aux_mask]
            cond_aux = cond[aux_mask]
        else:
            x0_aux = x0_pred
            cond_aux = cond
        if x0_aux.shape[0] == 0:
            vel_loss = x0_pred.new_tensor(0.0)
            quat_loss = x0_pred.new_tensor(0.0)
            continuity_loss = x0_pred.new_tensor(0.0)
        else:
            vel_loss = self.diffusion.velocity_consistency_loss(x0_aux)
            quat_loss = self.diffusion.quaternion_unit_loss(x0_aux)
            continuity_loss = self.diffusion.continuity_loss(x0_aux, cond_aux)
        total = (
            noise_loss
            + self.diffusion.velocity_loss_weight * vel_loss
            + self.diffusion.quaternion_loss_weight * quat_loss
            + self.diffusion.continuity_loss_weight * continuity_loss
        )
        return {
            "loss": total,
            "noise_loss": noise_loss,
            "velocity_loss": vel_loss,
            "quaternion_loss": quat_loss,
            "continuity_loss": continuity_loss,
        }

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.diffusion.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "val_loss": val_loss,
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

        current_data = self.cfg.get("data", {})
        previous_data = checkpoint_cfg.get("data", {})
        for key in ("frame_dim", "history_len", "pred_len"):
            if key in current_data and key in previous_data and int(current_data[key]) != int(previous_data[key]):
                raise RuntimeError(
                    f"Refusing to resume {path}: checkpoint data.{key}={previous_data[key]}, "
                    f"current data.{key}={current_data[key]}."
                )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.diffusion.train()
        totals: dict[str, float] = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "velocity_loss": 0.0,
            "quaternion_loss": 0.0,
            "continuity_loss": 0.0,
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
            }
        self.diffusion.eval()
        totals: dict[str, float] = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "velocity_loss": 0.0,
            "quaternion_loss": 0.0,
            "continuity_loss": 0.0,
        }
        for batch in self.val_loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self._compute_losses(batch["cond"], batch["target"])
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
        return {key: value / len(self.val_loader) for key, value in totals.items()}

    def fit(self) -> None:
        epochs = int(self.cfg["training"]["epochs"])
        for epoch in range(self.start_epoch + 1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            val_loss = val_metrics["loss"]
            latest_val = val_loss if not math.isnan(val_loss) else train_metrics["loss"]
            is_best = latest_val < self.best_val
            if is_best:
                self.best_val = latest_val
            self._save_checkpoint("latest.pt", epoch, latest_val)
            if is_best:
                self._save_checkpoint("best.pt", epoch, latest_val)

            msg = (
                f"epoch {epoch:04d} | train {train_metrics['loss']:.6f} "
                f"| val {val_metrics['loss']:.6f} | noise {train_metrics['noise_loss']:.6f} "
                f"| vel {train_metrics['velocity_loss']:.6f} "
                f"| quat {train_metrics['quaternion_loss']:.6f} "
                f"| cont {train_metrics['continuity_loss']:.6f}"
            )
            print(msg, flush=True)
            if self.wandb is not None:
                self.wandb.log({f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
