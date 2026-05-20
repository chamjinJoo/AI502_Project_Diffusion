"""Denormalize and export a predicted chunk as GR00T-compatible CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import decode_future_model_space, load_stats
from utils.export_csv import export_reference_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, required=True, help=".npy predicted chunk shaped [K, 65]")
    parser.add_argument("--output_dir", type=str, default="exports")
    parser.add_argument("--stats", type=str, default=None, help="Normalization stats JSON/NPZ for denormalization")
    parser.add_argument("--already_denormalized", action="store_true")
    parser.add_argument("--reconstruct_velocity", action="store_true", help="Export joint_vel from finite differences of joint_pos")
    parser.add_argument("--fps", type=float, default=50.0, help="FPS used when reconstructing velocity")
    parser.add_argument("--joint_vel_mode", type=str, default="source", choices=["source", "finite_difference"])
    parser.add_argument("--body_pos_mode", type=str, default="relative", choices=["relative", "delta"])
    args = parser.parse_args()

    chunk = np.load(args.chunk).astype(np.float32)  # [K, 65]
    if not args.already_denormalized:
        if args.stats is None:
            raise ValueError("--stats is required unless --already_denormalized is set")
        mean, std = load_stats(args.stats)
        chunk = chunk * std + mean
    chunk = decode_future_model_space(
        chunk,
        fps=args.fps,
        joint_vel_mode=args.joint_vel_mode,
        body_pos_mode=args.body_pos_mode,
    )
    export_reference_csv(chunk, args.output_dir, reconstruct_velocity=args.reconstruct_velocity, fps=args.fps)
    print(f"exported CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
