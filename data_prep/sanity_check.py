"""Sanity checks for processed [T, 65] training sequences."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.finite_difference import finite_difference
from utils.motion_io import load_yaml, read_jsonl, write_json
from utils.quaternion_utils import quat_norm_error


def _summarize_sequence(path: str, fps: float) -> dict[str, Any]:
    sequence = np.load(path).astype(np.float32)  # [T, 65]
    joint_pos = sequence[:, :29]
    joint_vel = sequence[:, 29:58]
    quat = sequence[:, 58:62]
    fd_vel = finite_difference(joint_pos, dt=1.0 / fps)
    return {
        "path": path,
        "shape": list(sequence.shape),
        "min": float(np.min(sequence)),
        "max": float(np.max(sequence)),
        "finite": bool(np.isfinite(sequence).all()),
        "max_quaternion_norm_error": float(np.max(quat_norm_error(quat))),
        "joint_vel_fd_mse": float(np.mean(np.square(joint_vel[1:] - fd_vel[1:])) if sequence.shape[0] > 1 else 0.0),
    }


def sanity_check(config_path: str | Path, num_samples: int = 8, plot: bool = False) -> dict[str, Any]:
    """Run lightweight checks and write sanity_report.json."""
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_root"])
    train_rows = read_jsonl(output_root / "manifests" / "train_manifest.jsonl")
    val_rows = read_jsonl(output_root / "manifests" / "val_manifest.jsonl")
    all_rows = train_rows + val_rows
    rng = random.Random(int(cfg["split_seed"]))
    sample_rows = rng.sample(all_rows, k=min(num_samples, len(all_rows))) if all_rows else []
    summaries = [_summarize_sequence(row["processed_npy_path"], float(row["fps_output"])) for row in sample_rows]

    report = {
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_checked": len(summaries),
        "samples": summaries,
    }
    write_json(output_root / "reports" / "sanity_report.json", report)

    for item in summaries:
        print(
            f"{item['path']} shape={item['shape']} range=({item['min']:.4f}, {item['max']:.4f}) "
            f"quat_err={item['max_quaternion_norm_error']:.6f} vel_fd_mse={item['joint_vel_fd_mse']:.6f}"
        )

    if plot and summaries:
        import matplotlib.pyplot as plt

        seq = np.load(summaries[0]["path"])
        plt.plot(seq[:, : min(5, seq.shape[1])])
        plt.title("First few joint_pos dimensions")
        plt.show()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    sanity_check(args.config, num_samples=args.num_samples, plot=args.plot)


if __name__ == "__main__":
    main()
