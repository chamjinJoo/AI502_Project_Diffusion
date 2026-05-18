"""Scan a directory for valid [T, 65] motion .npy files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import find_motion_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/motion_files.json")
    parser.add_argument("--frame_dim", type=int, default=65)
    args = parser.parse_args()

    paths = find_motion_files(args.data_dir, frame_dim=args.frame_dim)
    payload = []
    for path in paths:
        array = np.load(path, mmap_mode="r")
        payload.append({"path": str(path), "shape": list(array.shape)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"found {len(payload)} valid .npy files")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
