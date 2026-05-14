"""Simple temporal resampling helpers."""

from __future__ import annotations

import numpy as np

from .quaternion_utils import normalize_quat


def resample_array(values: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Linearly resample [T, D] values from fps_in to fps_out."""
    if fps_in <= 0 or fps_out <= 0 or values.shape[0] < 2 or abs(fps_in - fps_out) < 1e-6:
        return values.astype(np.float32)
    duration = (values.shape[0] - 1) / fps_in
    out_frames = max(2, int(round(duration * fps_out)) + 1)
    t_in = np.arange(values.shape[0], dtype=np.float32) / fps_in
    t_out = np.arange(out_frames, dtype=np.float32) / fps_out
    out = np.empty((out_frames, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(t_out, t_in, values[:, dim])
    return out


def resample_motion_parts(
    joint_pos: np.ndarray,
    body_quat: np.ndarray,
    body_pos: np.ndarray,
    fps_in: float,
    fps_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resample motion components and renormalize quaternion after interpolation."""
    joint_pos_out = resample_array(joint_pos, fps_in, fps_out)  # [T, 29]
    body_quat_out = resample_array(body_quat, fps_in, fps_out)  # [T, 4]
    body_quat_out, quat_changed = normalize_quat(body_quat_out)
    body_pos_out = resample_array(body_pos, fps_in, fps_out)  # [T, 3]
    return joint_pos_out, body_quat_out, body_pos_out, quat_changed
