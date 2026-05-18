"""Evaluate generated future references with tracking-oriented diagnostics."""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import load_stats
from scripts.evaluate import compute_metrics


def _load_chunk(path: str | Path) -> np.ndarray:
    """Load [K, 65] or [N, K, 65] motion chunks as [N, K, 65]."""
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != 65:
        raise ValueError(f"{path} must have shape [K, 65] or [N, K, 65], got {array.shape}")
    return array


def _load_history(path: str | Path) -> np.ndarray:
    """Load [H, 65] or [N, H, 65] histories as [N, H, 65]."""
    array = _load_chunk(path)
    return array


def reference_quality_metrics(
    pred: np.ndarray,
    target: np.ndarray | None = None,
    cond: np.ndarray | None = None,
    fps: float = 50.0,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MSE plus smoothness and history/future seam diagnostics."""
    metrics: dict[str, float] = {}
    if target is not None:
        metrics.update(compute_metrics(pred, target, mean=mean, std=std))

    joint_pos = pred[:, :, :29]  # [N, K, 29]
    joint_vel = pred[:, :, 29:58]  # [N, K, 29]
    if pred.shape[1] >= 2:
        finite_diff_vel = (joint_pos[:, 1:] - joint_pos[:, :-1]) * float(fps)
        metrics["vel_fd_mse"] = float(np.mean((joint_vel[:, 1:] - finite_diff_vel) ** 2))
        metrics["mean_step_joint_pos_rmse"] = float(np.sqrt(np.mean((joint_pos[:, 1:] - joint_pos[:, :-1]) ** 2)))
    else:
        metrics["vel_fd_mse"] = 0.0
        metrics["mean_step_joint_pos_rmse"] = 0.0

    quat = pred[:, :, 58:62]
    metrics["quat_norm_min"] = float(np.min(np.linalg.norm(quat, axis=-1)))
    metrics["quat_norm_max"] = float(np.max(np.linalg.norm(quat, axis=-1)))
    metrics["finite"] = float(np.isfinite(pred).all())
    metrics["max_abs"] = float(np.max(np.abs(pred)))

    if cond is not None:
        if cond.shape[0] != pred.shape[0]:
            raise ValueError(f"cond batch {cond.shape[0]} must match pred batch {pred.shape[0]}")
        last = cond[:, -1]
        seam_pos = pred[:, 0, :29] - last[:, :29]
        expected_pos = last[:, :29] + last[:, 29:58] / float(fps)
        expected_seam = pred[:, 0, :29] - expected_pos
        seam_vel = (pred[:, 0, :29] - last[:, :29]) * float(fps)
        metrics["seam_pos_rmse"] = float(np.sqrt(np.mean(seam_pos**2)))
        metrics["seam_expected_rmse"] = float(np.sqrt(np.mean(expected_seam**2)))
        metrics["seam_velocity_mse"] = float(np.mean((pred[:, 0, 29:58] - seam_vel) ** 2))
        metrics["first_velocity_mse_to_history"] = float(np.mean((pred[:, 0, 29:58] - last[:, 29:58]) ** 2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Predicted future chunk .npy")
    parser.add_argument("--target", default=None, help="Optional target future chunk .npy")
    parser.add_argument("--cond", default=None, help="Optional condition history .npy")
    parser.add_argument("--stats", default=None, help="Optional normalization stats JSON/NPZ")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of key/value lines")
    args = parser.parse_args()

    pred = _load_chunk(args.pred)
    target = _load_chunk(args.target) if args.target else None
    cond = _load_history(args.cond) if args.cond else None
    mean = std = None
    if args.stats:
        mean, std = load_stats(args.stats)

    metrics = reference_quality_metrics(pred, target=target, cond=cond, fps=args.fps, mean=mean, std=std)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        for key, value in metrics.items():
            print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
