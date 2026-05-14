"""Compute per-dimension normalization stats over training sequences only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.motion_io import ensure_dir, load_yaml, read_jsonl, write_json


def compute_stats(config_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std over train_manifest processed arrays."""
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_root"])
    train_rows = read_jsonl(output_root / "manifests" / "train_manifest.jsonl")
    if not train_rows:
        raise ValueError("train_manifest.jsonl is empty; run make_splits.py first")

    total_count = 0
    total_sum = np.zeros(65, dtype=np.float64)
    total_sq = np.zeros(65, dtype=np.float64)
    progress_every = int(cfg.get("progress_every", 100))
    start_time = time.time()
    print(f"[stats] computing over {len(train_rows)} train sequences", flush=True)
    for index, row in enumerate(train_rows, start=1):
        sequence = np.load(row["processed_npy_path"]).astype(np.float64)  # [T, 65]
        total_count += sequence.shape[0]
        total_sum += sequence.sum(axis=0)
        total_sq += np.square(sequence).sum(axis=0)
        if index == 1 or index % progress_every == 0 or index == len(train_rows):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[stats] {index}/{len(train_rows)} sequences | frames={total_count} "
                f"| {index / elapsed:.2f} seq/s",
                flush=True,
            )

    mean = total_sum / total_count
    var = np.maximum(total_sq / total_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), float(cfg["std_epsilon"]))

    stats_dir = ensure_dir(output_root / "stats")
    np.save(stats_dir / "mean.npy", mean.astype(np.float32))
    np.save(stats_dir / "std.npy", std.astype(np.float32))
    write_json(
        stats_dir / "stats.json",
        {
            "num_train_sequences": len(train_rows),
            "num_train_frames": int(total_count),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "std_epsilon": float(cfg["std_epsilon"]),
        },
    )
    print(f"[stats] done num_train_frames={total_count}", flush=True)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    args = parser.parse_args()
    mean, std = compute_stats(args.config)
    print(f"computed stats with shape mean={mean.shape}, std={std.shape}")


if __name__ == "__main__":
    main()
