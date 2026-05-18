"""Evaluate predicted future chunks against target chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import load_stats


_COMPONENTS: dict[str, slice] = {
    "joint_pos": slice(0, 29),
    "joint_vel": slice(29, 58),
    "body_quat": slice(58, 62),
    "body_pos": slice(62, 65),
}


def _load(path: str | Path) -> np.ndarray:
    """Load a future chunk as [N, K, 65]."""
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None]  # [1, K, 65]
    if array.ndim != 3 or array.shape[-1] != 65:
        raise ValueError(f"{path} must have shape [K, 65] or [N, K, 65], got {array.shape}")
    return array


def _component_mse(pred: np.ndarray, target: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute full and per-component MSE for [N, K, 65] arrays."""
    metrics = {f"{prefix}full_mse": float(np.mean((pred - target) ** 2))}
    for name, slc in _COMPONENTS.items():
        metrics[f"{prefix}{name}_mse"] = float(np.mean((pred[:, :, slc] - target[:, :, slc]) ** 2))
    return metrics


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute physical-space metrics and optional normalized-space MSE.

    Args:
        pred: Predicted future chunks with shape [N, K, 65].
        target: Target future chunks with shape [N, K, 65].
        mean: Optional normalization mean with shape [65].
        std: Optional normalization std with shape [65].
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {pred.shape} and {target.shape}")

    metrics = _component_mse(pred, target)
    quat = pred[:, :, 58:62]  # [N, K, 4], body_quat in (w, x, y, z)
    metrics["quaternion_norm_error"] = float(np.mean(np.abs(np.linalg.norm(quat, axis=-1) - 1.0)))

    if mean is not None and std is not None:
        mean = mean.astype(np.float32).reshape(1, 1, 65)
        std = np.maximum(std.astype(np.float32).reshape(1, 1, 65), 1e-6)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        metrics.update(_component_mse(pred_norm, target_norm, prefix="norm_"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--stats", type=str, default=None, help="Optional normalization stats JSON/NPZ")
    args = parser.parse_args()

    pred = _load(args.pred)
    target = _load(args.target)
    mean = std = None
    if args.stats is not None:
        mean, std = load_stats(args.stats)

    metrics = compute_metrics(pred, target, mean=mean, std=std)
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
