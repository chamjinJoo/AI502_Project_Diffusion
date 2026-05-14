"""Compute and save normalization statistics for motion .npy files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import compute_mean_std, find_motion_files, save_stats


def _paths_from_json(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload and isinstance(payload[0], dict):
        return [item["path"] for item in payload]
    return [str(item) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--file_list", type=str, default=None, help="JSON from preprocess_dataset.py")
    parser.add_argument("--output", type=str, default="checkpoints/normalization_stats.json")
    parser.add_argument("--frame_dim", type=int, default=65)
    args = parser.parse_args()

    if args.file_list is not None:
        paths = _paths_from_json(args.file_list)
    elif args.data_dir is not None:
        paths = [str(path) for path in find_motion_files(args.data_dir, frame_dim=args.frame_dim)]
    else:
        raise ValueError("Provide --data_dir or --file_list")

    if not paths:
        raise ValueError("No valid files found for stats computation")
    mean, std = compute_mean_std(paths, frame_dim=args.frame_dim)
    save_stats(args.output, mean, std)
    print(f"computed stats from {len(paths)} files")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
