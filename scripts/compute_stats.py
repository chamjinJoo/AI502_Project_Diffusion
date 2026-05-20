"""Compute and save normalization statistics for motion .npy files or windows."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import MotionChunkDataset, compute_mean_std, find_motion_files, save_stats


def _path_from_manifest_row(row: dict[str, Any]) -> str:
    """Extract a processed motion path from a manifest row."""
    if "processed_npy_path" in row:
        return str(row["processed_npy_path"])
    if "path" in row:
        return str(row["path"])
    raise KeyError("manifest row must contain 'processed_npy_path' or 'path'")


def _paths_from_file_list(path: str | Path) -> list[str]:
    """Read .npy paths from JSON, JSONL, or plain text file lists."""
    path = Path(path)
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


def compute_window_stats(
    paths: list[str],
    output: str | Path,
    history_len: int,
    pred_len: int,
    frame_dim: int,
    root_relative: bool,
    fps: float,
    joint_vel_mode: str,
    body_pos_mode: str,
    max_windows: int | None,
    seed: int,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stats over sampled windows after optional root-relative conversion.

    Both cond [H, 65] and target [K, 65] frames contribute, matching the
    normalized tensors seen by training. The implementation streams selected
    files instead of keeping every sequence memmap open at once.
    """
    from datasets.motion_chunk_dataset import apply_model_space_transforms, make_root_relative

    rng = random.Random(seed)
    selected_paths = list(paths)
    if max_files is not None and max_files < len(selected_paths):
        selected_paths = rng.sample(selected_paths, max_files)
    rng.shuffle(selected_paths)

    target_windows = max_windows
    windows_per_file = 1
    if target_windows is not None and selected_paths:
        windows_per_file = max(1, int(np.ceil(target_windows / len(selected_paths))))

    total_count = 0
    total_windows = 0
    total_sum = np.zeros(frame_dim, dtype=np.float64)
    total_sq = np.zeros(frame_dim, dtype=np.float64)
    start_time = time.time()
    print(
        f"[stats] streaming files={len(selected_paths)}/{len(paths)} root_relative={root_relative} "
        f"joint_vel_mode={joint_vel_mode} body_pos_mode={body_pos_mode} "
        f"max_windows={target_windows} windows_per_file={windows_per_file}",
        flush=True,
    )

    for file_index, path in enumerate(selected_paths, start=1):
        sequence = np.load(path, mmap_mode="r")
        if sequence.ndim != 2 or sequence.shape[1] != frame_dim:
            continue
        max_t = len(sequence) - pred_len - 1
        min_t = history_len - 1
        available = max(0, max_t - min_t + 1)
        if available <= 0:
            continue
        sample_count = min(windows_per_file, available)
        if sample_count == available:
            offsets = list(range(available))
        else:
            offsets = rng.sample(range(available), sample_count)
        for offset in offsets:
            t = min_t + offset
            cond = np.asarray(sequence[t - history_len + 1 : t + 1], dtype=np.float32)  # [H, 65]
            target = np.asarray(sequence[t + 1 : t + 1 + pred_len], dtype=np.float32)  # [K, 65]
            if root_relative:
                cond, target = make_root_relative(cond, target)
            cond, target = apply_model_space_transforms(
                cond,
                target,
                fps=fps,
                joint_vel_mode=joint_vel_mode,
                body_pos_mode=body_pos_mode,
            )
            frames = np.concatenate([cond, target], axis=0).astype(np.float64)
            total_count += frames.shape[0]
            total_windows += 1
            total_sum += frames.sum(axis=0)
            total_sq += np.square(frames).sum(axis=0)
            if target_windows is not None and total_windows >= target_windows:
                break
        if file_index == 1 or file_index % 1000 == 0 or file_index == len(selected_paths) or (
            target_windows is not None and total_windows >= target_windows
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[stats] files={file_index}/{len(selected_paths)} windows={total_windows} "
                f"frames={total_count} | {total_windows / elapsed:.2f} win/s",
                flush=True,
            )
        if target_windows is not None and total_windows >= target_windows:
            break

    if total_count == 0:
        raise ValueError("No frames accumulated for stats")
    mean = total_sum / total_count
    var = np.maximum(total_sq / total_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    save_stats(output, mean.astype(np.float32), std.astype(np.float32))

    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    payload.update(
        {
            "stats_type": "sampled_window",
            "root_relative": bool(root_relative),
            "fps": float(fps),
            "joint_vel_mode": str(joint_vel_mode),
            "body_pos_mode": str(body_pos_mode),
            "history_len": int(history_len),
            "pred_len": int(pred_len),
            "frame_dim": int(frame_dim),
            "num_source_files": int(len(paths)),
            "num_files_used": int(file_index if 'file_index' in locals() else 0),
            "max_files": None if max_files is None else int(max_files),
            "num_windows_used": int(total_windows),
            "num_frames_accumulated": int(total_count),
            "seed": int(seed),
        }
    )
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--file_list", type=str, default=None, help="JSON/JSONL/plain-text file list")
    parser.add_argument("--output", type=str, default="checkpoints/normalization_stats.json")
    parser.add_argument("--frame_dim", type=int, default=65)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--root_relative", action="store_true")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--joint_vel_mode", type=str, default="source", choices=["source", "finite_difference"])
    parser.add_argument("--body_pos_mode", type=str, default="relative", choices=["relative", "delta"])
    parser.add_argument("--max_windows", type=int, default=None, help="Optional random window subset for window stats")
    parser.add_argument("--max_files", type=int, default=None, help="Optional random file subset for window stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.file_list is not None:
        paths = _paths_from_file_list(args.file_list)
    elif args.data_dir is not None:
        paths = [str(path) for path in find_motion_files(args.data_dir, frame_dim=args.frame_dim)]
    else:
        raise ValueError("Provide --data_dir or --file_list")

    if not paths:
        raise ValueError("No valid files found for stats computation")

    use_window_stats = args.root_relative or args.joint_vel_mode != "source" or args.body_pos_mode != "relative"
    if use_window_stats:
        compute_window_stats(
            paths=paths,
            output=args.output,
            history_len=args.history_len,
            pred_len=args.pred_len,
            frame_dim=args.frame_dim,
            root_relative=args.root_relative,
            fps=args.fps,
            joint_vel_mode=args.joint_vel_mode,
            body_pos_mode=args.body_pos_mode,
            max_windows=args.max_windows,
            seed=args.seed,
            max_files=args.max_files,
        )
    else:
        mean, std = compute_mean_std(paths, frame_dim=args.frame_dim)
        save_stats(args.output, mean, std)
        print(f"computed stats from {len(paths)} files")
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
