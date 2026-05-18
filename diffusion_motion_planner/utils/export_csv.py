"""Export GR00T tracking-reference chunks to CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _write_csv(path: Path, header: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data.tolist())


def reconstruct_joint_vel(chunk: np.ndarray, fps: float = 50.0) -> np.ndarray:
    """Reconstruct joint velocity from finite differences of joint positions.

    The first timestep uses the same velocity as the second timestep so the
    output keeps shape [K, 29].
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    joint_pos = chunk[:, :29]  # [K, 29]
    if joint_pos.shape[0] == 1:
        return np.zeros_like(joint_pos)
    vel = np.zeros_like(joint_pos)
    dt = 1.0 / fps
    vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt  # [K-1, 29]
    vel[0] = vel[1]
    return vel


def export_reference_csv(
    chunk: np.ndarray,
    output_dir: str | Path,
    reconstruct_velocity: bool = False,
    fps: float = 50.0,
) -> None:
    """Export a [K, 65] chunk into GR00T-compatible CSV groups."""
    if chunk.ndim != 2 or chunk.shape[1] != 65:
        raise ValueError(f"chunk must have shape [K, 65], got {chunk.shape}")
    output_dir = Path(output_dir)

    joint_pos = chunk[:, :29]  # [K, 29]
    joint_vel = reconstruct_joint_vel(chunk, fps=fps) if reconstruct_velocity else chunk[:, 29:58]  # [K, 29]
    body_quat = chunk[:, 58:62]  # [K, 4], order: w, x, y, z
    body_pos = chunk[:, 62:65]  # [K, 3]

    _write_csv(output_dir / "joint_pos.csv", [f"joint_pos_{i}" for i in range(29)], joint_pos)
    _write_csv(output_dir / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], joint_vel)
    _write_csv(output_dir / "body_quat.csv", ["w", "x", "y", "z"], body_quat)
    _write_csv(output_dir / "body_pos.csv", ["x", "y", "z"], body_pos)
