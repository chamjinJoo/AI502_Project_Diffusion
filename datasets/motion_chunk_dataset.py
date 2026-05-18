"""Sliding-window dataset for GR00T tracking-reference motion chunks."""

from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.quaternion_utils import normalize_quat, quat_conjugate, quat_multiply, quat_rotate_inverse


FRAME_DIM = 65


def _as_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path) for path in paths]


def _load_sequences(paths: list[Path], frame_dim: int = FRAME_DIM) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Motion file not found: {path}. "
                "Update data.train_paths/data.val_paths or use data.train_file_list in the config."
            )
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != frame_dim:
            raise ValueError(f"{path} must have shape [T, {frame_dim}], got {array.shape}")
        sequences.append(array.astype(np.float32, copy=False))
    return sequences


def make_root_relative(cond: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express body pose channels relative to the last history frame.

    The input chunks are source-global [H, 65] and [K, 65]. Joint position and
    velocity channels are left unchanged; only body_quat/body_pos are converted.
    """
    cond_rel = np.asarray(cond, dtype=np.float32).copy()
    target_rel = np.asarray(target, dtype=np.float32).copy()
    anchor_quat = normalize_quat(cond_rel[-1:, 58:62])[0][0]  # [4], wxyz
    anchor_pos = cond_rel[-1, 62:65].copy()  # [3]
    anchor_inv = quat_conjugate(anchor_quat)  # [4]

    for chunk in (cond_rel, target_rel):
        if chunk.shape[0] == 0:
            continue
        chunk[:, 62:65] = quat_rotate_inverse(anchor_quat, chunk[:, 62:65] - anchor_pos)
        chunk[:, 58:62] = normalize_quat(quat_multiply(anchor_inv, chunk[:, 58:62]))[0]
    return cond_rel.astype(np.float32), target_rel.astype(np.float32)


def find_motion_files(data_dir: str | Path, frame_dim: int = FRAME_DIM) -> list[Path]:
    """Recursively find .npy files shaped [T, frame_dim]."""
    valid_paths: list[Path] = []
    for path in sorted(Path(data_dir).rglob("*.npy")):
        try:
            array = np.load(path, mmap_mode="r")
        except ValueError:
            continue
        if array.ndim == 2 and array.shape[1] == frame_dim:
            valid_paths.append(path)
    return valid_paths


def compute_mean_std(paths: str | Path | list[str | Path], frame_dim: int = FRAME_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension z-score statistics over one or more [T, 65] files."""
    sequences = _load_sequences(_as_paths(paths), frame_dim=frame_dim)
    all_frames = np.concatenate(sequences, axis=0)  # [sum_T, 65]
    mean = all_frames.mean(axis=0).astype(np.float32)  # [65]
    std = all_frames.std(axis=0).astype(np.float32)  # [65]
    std = np.maximum(std, 1e-6)
    return mean, std


def save_stats(path: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    """Save normalization statistics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalization statistics from JSON or NPZ."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)


class MotionChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Return history-conditioned future chunks from [T, 65] motion sequences."""

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        history_len: int = 20,
        pred_len: int = 10,
        split: str = "train",
        val_split: float = 0.1,
        stats_path: str | Path | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        normalize: bool = True,
        frame_dim: int = FRAME_DIM,
        samples_per_epoch: int | None = None,
        random_window_sampling: bool = False,
        root_relative: bool = False,
    ) -> None:
        """Build sliding windows.

        Sample index t yields:
        cond = sequence[t-H+1:t+1] with shape [H, 65]
        target = sequence[t+1:t+1+K] with shape [K, 65]

        For large motion corpora, set samples_per_epoch and random_window_sampling=True
        so each epoch draws a fresh subset without materializing every window index.
        """
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")

        self.paths = _as_paths(paths)
        self.history_len = history_len
        self.pred_len = pred_len
        self.frame_dim = frame_dim
        self.normalize = normalize
        self.samples_per_epoch = samples_per_epoch
        self.random_window_sampling = random_window_sampling
        self.root_relative = root_relative
        self.sequences = _load_sequences(self.paths, frame_dim=frame_dim)

        if mean is None or std is None:
            if stats_path is not None and Path(stats_path).exists():
                mean, std = load_stats(stats_path)
            else:
                mean, std = compute_mean_std(self.paths, frame_dim=frame_dim)
                if stats_path is not None:
                    save_stats(stats_path, mean, std)

        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

        self.windows: list[tuple[int, int, int]] = []
        self.cumulative_counts: list[int] = []
        total_windows = 0
        for seq_idx, sequence in enumerate(self.sequences):
            max_t = len(sequence) - pred_len - 1
            min_t = history_len - 1
            count = max(0, max_t - min_t + 1)
            start_t = min_t
            if split != "all" and val_split > 0.0:
                cutoff = int(count * (1.0 - val_split))
                if split == "train":
                    count = cutoff
                else:
                    start_t = min_t + cutoff
                    count = count - cutoff
            if count > 0:
                self.windows.append((seq_idx, start_t, count))
                total_windows += count
                self.cumulative_counts.append(total_windows)

        if total_windows == 0:
            raise ValueError("No valid motion chunks found. Check sequence lengths, H, and K.")
        self.total_windows = total_windows
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive when set")
        self.epoch_length = min(self.samples_per_epoch, total_windows) if self.samples_per_epoch else total_windows

    def __len__(self) -> int:
        return self.epoch_length

    def _index_to_window(self, index: int) -> tuple[int, int]:
        """Map a flat window index to (sequence index, timestep t)."""
        window_idx = bisect.bisect_right(self.cumulative_counts, index)
        previous_count = self.cumulative_counts[window_idx - 1] if window_idx > 0 else 0
        seq_idx, start_t, _ = self.windows[window_idx]
        return seq_idx, start_t + (index - previous_count)

    def _normalize(self, chunk: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return chunk.astype(np.float32)
        return ((chunk - self.mean) / self.std).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.random_window_sampling:
            index = random.randrange(self.total_windows)
        seq_idx, t = self._index_to_window(index % self.total_windows)
        sequence = self.sequences[seq_idx]
        cond = sequence[t - self.history_len + 1 : t + 1]  # [H, 65]
        target = sequence[t + 1 : t + 1 + self.pred_len]  # [K, 65]
        if self.root_relative:
            cond, target = make_root_relative(cond, target)
        return {
            "cond": torch.from_numpy(self._normalize(cond)),  # [H, 65]
            "target": torch.from_numpy(self._normalize(target)),  # [K, 65]
        }
