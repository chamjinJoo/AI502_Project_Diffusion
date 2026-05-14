"""Smoke tests for the minimal conditional DDIM planner."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.motion_chunk_dataset import MotionChunkDataset
from data_prep.convert_bones_seed_to_internal import convert_one_csv
from diffusion.scheduler_wrapper import DiffusionSchedulerWrapper
from models.denoiser import ConditionalDenoiser
from scripts.train import validate_processed_manifest_metadata
from training.trainer import Trainer
from utils.export_csv import export_reference_csv, reconstruct_joint_vel


class ZeroDenoiser(nn.Module):
    """Tiny model for shape tests."""

    def forward(self, xt: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(xt)


class TinyDiffusion(nn.Module):
    """Minimal diffusion-like module for checkpoint tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)

    def scheduler_config(self) -> dict[str, int]:
        return {"num_train_timesteps": 4}


def test_dataset_slicing(tmp_path: Path) -> None:
    seq = np.arange(30 * 65, dtype=np.float32).reshape(30, 65)
    path = tmp_path / "seq.npy"
    np.save(path, seq)

    dataset = MotionChunkDataset(path, history_len=4, pred_len=3, split="all", normalize=False)
    item = dataset[0]

    np.testing.assert_allclose(item["cond"].numpy(), seq[0:4])
    np.testing.assert_allclose(item["target"].numpy(), seq[4:7])
    assert item["cond"].shape == (4, 65)
    assert item["target"].shape == (3, 65)
    assert item["cond"].dtype == torch.float32
    assert item["target"].dtype == torch.float32


def test_dataset_uses_mmap_and_returns_float32(tmp_path: Path) -> None:
    seq = np.random.randn(40, 65).astype(np.float32)
    path = tmp_path / "seq.npy"
    np.save(path, seq)

    dataset = MotionChunkDataset(path, history_len=4, pred_len=3, split="all", normalize=False)
    item = dataset[0]

    assert isinstance(dataset.sequences[0], np.memmap)
    assert item["cond"].shape == (4, 65)
    assert item["target"].shape == (3, 65)
    assert item["cond"].dtype == torch.float32
    assert item["target"].dtype == torch.float32


def test_add_noise_output_shape() -> None:
    diffusion = DiffusionSchedulerWrapper(ZeroDenoiser(), num_train_timesteps=8, beta_schedule="linear")
    x0 = torch.randn(2, 3, 65)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 7], dtype=torch.long)
    xt = diffusion.add_noise(x0, noise, t)
    assert xt.shape == (2, 3, 65)


def test_ddim_sampler_output_shape() -> None:
    sampler = DiffusionSchedulerWrapper(ZeroDenoiser(), num_train_timesteps=8, beta_schedule="linear")
    cond = torch.randn(2, 4, 65)
    x_t = torch.randn(2, 3, 65)
    out = sampler.sample(cond, pred_len=3, frame_dim=65, num_inference_steps=4, x_T=x_t)
    assert out.shape == (2, 3, 65)


def test_ddim_sampler_output_shape_with_unet() -> None:
    model = ConditionalDenoiser(
        frame_dim=65,
        history_len=4,
        pred_len=10,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        condition_encoder="conv",
        architecture="unet",
        down_dims=(32, 64, 128),
    )
    sampler = DiffusionSchedulerWrapper(model, num_train_timesteps=8, beta_schedule="linear")
    cond = torch.randn(2, 4, 65)
    x_t = torch.randn(2, 10, 65)
    out = sampler.sample(cond, pred_len=10, frame_dim=65, num_inference_steps=4, x_T=x_t)
    assert out.shape == (2, 10, 65)


def test_ddim_sampler_external_xt_shape_check() -> None:
    sampler = DiffusionSchedulerWrapper(ZeroDenoiser(), num_train_timesteps=8, beta_schedule="linear")
    cond = torch.randn(2, 4, 65)
    bad_x_t = torch.randn(2, 4, 65)
    try:
        sampler.sample(cond, pred_len=3, frame_dim=65, num_inference_steps=4, x_T=bad_x_t)
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad x_T shape")


def test_conditional_unet_denoiser_output_shape() -> None:
    model = ConditionalDenoiser(
        frame_dim=65,
        history_len=4,
        pred_len=10,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        condition_encoder="conv",
        architecture="unet",
        down_dims=(32, 64, 128),
    )
    xt = torch.randn(2, 10, 65)
    cond = torch.randn(2, 4, 65)
    timesteps = torch.tensor([0, 7], dtype=torch.long)
    out = model(xt, cond, timesteps)
    assert out.shape == (2, 10, 65)


def test_transformer_denoiser_backward_compat_output_shape() -> None:
    model = ConditionalDenoiser(
        frame_dim=65,
        history_len=4,
        pred_len=3,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        condition_encoder="conv",
        architecture="transformer",
    )
    xt = torch.randn(2, 3, 65)
    cond = torch.randn(2, 4, 65)
    timesteps = torch.tensor([0, 7], dtype=torch.long)
    out = model(xt, cond, timesteps)
    assert out.shape == (2, 3, 65)


def test_reconstruct_joint_vel_uses_fps() -> None:
    chunk = np.zeros((3, 65), dtype=np.float32)
    chunk[:, 0] = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    vel = reconstruct_joint_vel(chunk, fps=50.0)
    assert vel.shape == (3, 29)
    np.testing.assert_allclose(vel[:, 0], np.array([50.0, 50.0, 100.0], dtype=np.float32))


def test_reconstruct_joint_vel_single_frame() -> None:
    chunk = np.ones((1, 65), dtype=np.float32)
    vel = reconstruct_joint_vel(chunk, fps=50.0)
    assert vel.shape == (1, 29)
    np.testing.assert_allclose(vel, np.zeros((1, 29), dtype=np.float32))


def test_fallback_velocity_uses_output_fps_after_resampling(tmp_path: Path) -> None:
    path = tmp_path / "motion.csv"
    columns = [f"qpos_{i:02d}" for i in range(29)]
    columns += ["root_rotatex", "root_rotatey", "root_rotatez", "root_pos_x", "root_pos_y", "root_pos_z"]
    rows = []
    source_fps = 120.0
    target_fps = 60.0
    for i in range(121):
        t = i / source_fps
        joint_pos = [t] + [0.0] * 28
        rows.append(joint_pos + [0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(value) for value in row) + "\n")

    cfg = {
        "target_fps": target_fps,
        "source_joint_unit": "radians",
        "source_root_pos_unit": "centimeters",
        "source_root_euler_unit": "degrees",
        "min_sequence_length": 2,
        "joint_pos_only_mode": False,
        "assume_fps_if_missing": source_fps,
        "allow_joint_vel_fallback": True,
        "allow_body_pos_zero_fallback": False,
        "allow_body_quat_identity_debug_fallback": False,
        "renormalize_quaternion": True,
        "expected_joint_count": 29,
        "quaternion_norm_tolerance": 0.05,
    }
    meta, sequence = convert_one_csv(path, "motion_000001", tmp_path / "out.npy", cfg)
    assert sequence is not None, meta.get("skip_reasons")
    assert meta["assumed_fps"] is True
    assert meta["fps_original"] == source_fps
    np.testing.assert_allclose(sequence[1:, 29], np.ones(sequence.shape[0] - 1), atol=1e-5)


def test_processed_manifest_stale_assumed_fps_guard(tmp_path: Path) -> None:
    manifest = tmp_path / "train_manifest.jsonl"
    manifest.write_text(
        '{"processed_npy_path":"processed_dataset/sequences/old.npy",'
        '"fps_output":50,"assumed_fps":true,"fps_original":50}\n',
        encoding="utf-8",
    )
    data_cfg = {"train_file_list": str(manifest), "fps": 50, "source_fps_if_assumed": 120}
    try:
        validate_processed_manifest_metadata(data_cfg, "train")
    except ValueError as exc:
        assert "stale" in str(exc)
        return
    raise AssertionError("expected stale manifest ValueError")


def test_csv_export_formatting(tmp_path: Path) -> None:
    chunk = np.arange(3 * 65, dtype=np.float32).reshape(3, 65)
    export_reference_csv(chunk, tmp_path, reconstruct_velocity=True)

    expected_files = ["joint_pos.csv", "joint_vel.csv", "body_quat.csv", "body_pos.csv"]
    for name in expected_files:
        assert (tmp_path / name).exists()

    with (tmp_path / "body_quat.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["w", "x", "y", "z"]
    assert len(rows) == 4

    with (tmp_path / "joint_vel.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [f"joint_vel_{i}" for i in range(29)]
    assert len(rows) == 4


def test_checkpoint_resume_preserves_best_val_and_scaler(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "device": "cpu",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "grad_clip": 1.0,
            "checkpoint_dir": str(tmp_path),
            "use_wandb": False,
            "resume": False,
        }
    }
    trainer = Trainer(TinyDiffusion(), DataLoader([]), None, cfg)
    trainer.best_val = 0.123
    trainer._save_checkpoint("latest.pt", epoch=1, val_loss=0.5)

    checkpoint = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
    assert checkpoint["best_val"] == 0.123
    assert "scaler" in checkpoint

    cfg["training"]["resume"] = True
    resumed = Trainer(TinyDiffusion(), DataLoader([]), None, cfg)
    assert resumed.start_epoch == 1
    assert resumed.best_val == 0.123


def test_checkpoint_resume_rejects_incompatible_architecture(tmp_path: Path) -> None:
    cfg = {
        "model": {"architecture": "unet"},
        "data": {"frame_dim": 65, "history_len": 20, "pred_len": 10},
        "training": {
            "device": "cpu",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "grad_clip": 1.0,
            "checkpoint_dir": str(tmp_path),
            "use_wandb": False,
            "resume": True,
        },
    }
    torch.save(
        {
            "epoch": 1,
            "model": TinyDiffusion().model.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "config": {"model": {"architecture": "transformer"}, "data": cfg["data"]},
        },
        tmp_path / "latest.pt",
    )
    try:
        Trainer(TinyDiffusion(), DataLoader([]), None, cfg)
    except RuntimeError as exc:
        assert "architecture" in str(exc)
        return
    raise AssertionError("expected incompatible architecture RuntimeError")
