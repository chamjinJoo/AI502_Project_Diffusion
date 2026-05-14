"""Evaluate predicted future chunks against target chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: str | Path) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None]  # [1, K, 65]
    if array.ndim != 3 or array.shape[-1] != 65:
        raise ValueError(f"{path} must have shape [K, 65] or [N, K, 65], got {array.shape}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()

    pred = _load(args.pred)
    target = _load(args.target)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {pred.shape} and {target.shape}")

    full_mse = float(np.mean((pred - target) ** 2))
    joint_pos_mse = float(np.mean((pred[:, :, :29] - target[:, :, :29]) ** 2))
    joint_vel_mse = float(np.mean((pred[:, :, 29:58] - target[:, :, 29:58]) ** 2))
    quat_norm_error = float(np.mean(np.abs(np.linalg.norm(pred[:, :, 58:62], axis=-1) - 1.0)))

    print(f"full_mse: {full_mse:.8f}")
    print(f"joint_pos_mse: {joint_pos_mse:.8f}")
    print(f"joint_vel_mse: {joint_vel_mse:.8f}")
    print(f"quaternion_norm_error: {quat_norm_error:.8f}")


if __name__ == "__main__":
    main()
