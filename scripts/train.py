"""Train the conditional DDIM planner."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import MotionChunkDataset, find_motion_files
from diffusion.scheduler_wrapper import DiffusionSchedulerWrapper
from models.denoiser import ConditionalDenoiser
from training.trainer import Trainer
from utils.seed import set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _path_from_manifest_row(row: dict[str, Any]) -> str:
    """Extract a motion .npy path from a manifest row."""
    if "processed_npy_path" in row:
        return str(row["processed_npy_path"])
    if "path" in row:
        return str(row["path"])
    raise KeyError("manifest row must contain 'processed_npy_path' or 'path'")


def _paths_from_file_list(path: str | Path) -> list[str]:
    """Read .npy paths from JSON, JSONL, or plain text file lists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File list not found: {path}. "
            "If this is a processed_dataset manifest, run data_prep/build_dataset.py first "
            "and make sure processed_dataset is visible to the SLURM job."
        )
    if path.suffix == ".jsonl":
        rows: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                rows.append(_path_from_manifest_row(payload) if isinstance(payload, dict) else str(payload))
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload and isinstance(payload[0], dict):
            return [_path_from_manifest_row(item) for item in payload]
        return [str(item) for item in payload]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_processed_manifest_metadata(data_cfg: dict[str, Any], split: str) -> None:
    """Fail early when a processed manifest was built with stale fps assumptions."""
    file_list = data_cfg.get(f"{split}_file_list")
    if not file_list or Path(file_list).suffix != ".jsonl":
        return
    expected_output_fps = data_cfg.get("fps")
    expected_assumed_fps = data_cfg.get("source_fps_if_assumed")
    checked = 0
    bad_rows: list[str] = []
    with Path(file_list).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            checked += 1
            path = row.get("processed_npy_path", row.get("path", "<unknown>"))
            if expected_output_fps is not None and "fps_output" in row:
                if abs(float(row["fps_output"]) - float(expected_output_fps)) > 1e-6:
                    bad_rows.append(f"{path}: fps_output={row['fps_output']} expected={expected_output_fps}")
            if expected_assumed_fps is not None and row.get("assumed_fps", False):
                if abs(float(row.get("fps_original", -1.0)) - float(expected_assumed_fps)) > 1e-6:
                    bad_rows.append(
                        f"{path}: assumed fps_original={row.get('fps_original')} expected={expected_assumed_fps}"
                    )
            if len(bad_rows) >= 5:
                break
    if bad_rows:
        preview = "\n  ".join(bad_rows)
        raise ValueError(
            f"{split} manifest appears stale or incompatible with the training config:\n  {preview}\n"
            "Rebuild processed_dataset with data_prep/build_dataset.py --force_rebuild after changing fps assumptions."
        )
    if checked:
        print(f"[data] checked {checked} {split} manifest metadata rows", flush=True)


def resolve_data_paths(data_cfg: dict[str, Any], split: str) -> list[str]:
    """Resolve train/val paths from explicit paths, a JSON file list, or a data directory."""
    file_list_key = f"{split}_file_list"
    data_dir_key = f"{split}_data_dir"
    paths_key = f"{split}_paths"
    if data_cfg.get(file_list_key):
        paths = _paths_from_file_list(data_cfg[file_list_key])
    elif data_cfg.get(data_dir_key):
        paths = [str(path) for path in find_motion_files(data_cfg[data_dir_key], frame_dim=int(data_cfg["frame_dim"]))]
    else:
        paths = data_cfg.get(paths_key) or []
    if not paths:
        raise ValueError(
            f"No {split} motion files found. Set data.{paths_key}, data.{file_list_key}, "
            f"or data.{data_dir_key}. Files must be .npy shaped [T, {data_cfg['frame_dim']}]."
        )
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        preview = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} {split} motion file(s) were not found. First missing paths:\n  {preview}\n"
            "For SLURM, use paths that exist after stage-in, or absolute paths visible from compute nodes."
        )
    return paths


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs in DataLoader workers from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def expand_checkpoint_dir(cfg: dict[str, Any]) -> None:
    """Expand date tokens in training.checkpoint_dir at launch time."""
    train_cfg = cfg["training"]
    checkpoint_dir = str(train_cfg["checkpoint_dir"])
    now = datetime.now()
    expanded = (
        checkpoint_dir
        .replace("{date}", now.strftime("%Y%m%d"))
        .replace("{datetime}", now.strftime("%Y%m%d_%H%M%S"))
    )
    train_cfg["checkpoint_dir"] = expanded
    if expanded != checkpoint_dir:
        print(f"[config] checkpoint_dir={expanded}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Optional device override.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    expand_checkpoint_dir(cfg)
    set_seed(int(cfg["seed"]))

    data_cfg = cfg["data"]
    validate_processed_manifest_metadata(data_cfg, "train")
    if data_cfg.get("val_file_list"):
        validate_processed_manifest_metadata(data_cfg, "val")
    train_paths = resolve_data_paths(data_cfg, "train")
    val_paths = resolve_data_paths(data_cfg, "val") if data_cfg.get("val_paths") or data_cfg.get("val_file_list") else train_paths
    train_dataset = MotionChunkDataset(
        train_paths,
        history_len=int(data_cfg["history_len"]),
        pred_len=int(data_cfg["pred_len"]),
        split="all" if data_cfg.get("train_file_list") else "train",
        val_split=float(data_cfg["val_split"]),
        stats_path=data_cfg["stats_path"],
        frame_dim=int(data_cfg["frame_dim"]),
        samples_per_epoch=cfg["training"].get("train_samples_per_epoch"),
        random_window_sampling=bool(cfg["training"].get("random_window_sampling", False)),
    )
    val_dataset = MotionChunkDataset(
        val_paths,
        history_len=int(data_cfg["history_len"]),
        pred_len=int(data_cfg["pred_len"]),
        split="all" if data_cfg.get("val_file_list") or data_cfg.get("val_paths") else "val",
        val_split=float(data_cfg["val_split"]),
        stats_path=data_cfg["stats_path"],
        frame_dim=int(data_cfg["frame_dim"]),
        samples_per_epoch=cfg["training"].get("val_samples_per_epoch"),
        random_window_sampling=False,
    )
    print(
        "[data] "
        f"train_files={len(train_paths)} train_windows={train_dataset.total_windows} train_epoch_samples={len(train_dataset)} "
        f"val_files={len(val_paths)} val_windows={val_dataset.total_windows} val_epoch_samples={len(val_dataset)}",
        flush=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    random_window_sampling = bool(cfg["training"].get("random_window_sampling", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=not random_window_sampling,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model_cfg = cfg["model"]
    model = ConditionalDenoiser(
        frame_dim=int(data_cfg["frame_dim"]),
        history_len=int(data_cfg["history_len"]),
        pred_len=int(data_cfg["pred_len"]),
        model_dim=int(model_cfg["dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        condition_encoder=str(model_cfg["condition_encoder"]),
        architecture=str(model_cfg.get("architecture", "unet")),
        down_dims=tuple(int(dim) for dim in model_cfg.get("down_dims", [256, 512, 1024])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        n_groups=int(model_cfg.get("n_groups", 8)),
        cond_predict_scale=bool(model_cfg.get("cond_predict_scale", False)),
        condition_summary=str(model_cfg.get("condition_summary", "flatten")),
        use_time_token=bool(model_cfg.get("use_time_token", False)),
        use_segment_embedding=bool(model_cfg.get("use_segment_embedding", False)),
        use_local_condition=bool(model_cfg.get("use_local_condition", False)),
    )
    diffusion_cfg = cfg["diffusion"]
    diffusion = DiffusionSchedulerWrapper(
        model,
        num_train_timesteps=int(diffusion_cfg["num_train_timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
        prediction_type=str(diffusion_cfg["prediction_type"]),
        clip_sample=bool(diffusion_cfg["clip_sample"]),
        velocity_loss_weight=float(cfg["training"]["velocity_loss_weight"]),
        quaternion_loss_weight=float(cfg["training"]["quaternion_loss_weight"]),
        continuity_loss_weight=float(cfg["training"].get("continuity_loss_weight", 0.0)),
        fps=float(data_cfg.get("fps", 50.0)),
    )
    diffusion.set_normalization_stats(torch.from_numpy(train_dataset.mean), torch.from_numpy(train_dataset.std))

    init_checkpoint = cfg["training"].get("init_checkpoint")
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    will_resume = bool(cfg["training"].get("resume", False)) and (checkpoint_dir / "latest.pt").exists()
    if init_checkpoint and not will_resume:
        try:
            checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print(f"[init] loaded model weights from {init_checkpoint}", flush=True)
    Trainer(diffusion, train_loader, val_loader, cfg).fit()


if __name__ == "__main__":
    main()
