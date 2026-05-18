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


def _slerp_single(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions with shape [4]."""
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q1 + t * (q2 - q1)).astype(np.float32)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s2 = np.sin(theta) / sin_theta_0
    return (s1 * q1 + s2 * q2).astype(np.float32)


def resample_quat_slerp(quat: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """SLERP-resample [T, 4] unit quaternions from fps_in to fps_out."""
    T = quat.shape[0]
    if fps_in <= 0 or fps_out <= 0 or T < 2 or abs(fps_in - fps_out) < 1e-6:
        return quat.astype(np.float32)
    duration = (T - 1) / fps_in
    out_frames = max(2, int(round(duration * fps_out)) + 1)
    t_in = np.arange(T, dtype=np.float64) / fps_in
    t_out = np.arange(out_frames, dtype=np.float64) / fps_out
    out = np.empty((out_frames, 4), dtype=np.float32)
    for i, t in enumerate(t_out):
        idx = int(np.searchsorted(t_in, t, side="right")) - 1
        idx = max(0, min(idx, T - 2))
        dt = t_in[idx + 1] - t_in[idx]
        alpha = float((t - t_in[idx]) / dt) if dt > 1e-12 else 0.0
        out[i] = _slerp_single(quat[idx], quat[idx + 1], alpha)
    return out


def resample_motion_parts(
    joint_pos: np.ndarray,
    body_quat: np.ndarray,
    body_pos: np.ndarray,
    fps_in: float,
    fps_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resample motion components. Quaternions use SLERP; others use linear interpolation."""
    joint_pos_out = resample_array(joint_pos, fps_in, fps_out)  # [T, 29]
    body_quat_out = resample_quat_slerp(body_quat, fps_in, fps_out)  # [T, 4]
    body_quat_out, quat_changed = normalize_quat(body_quat_out)
    body_pos_out = resample_array(body_pos, fps_in, fps_out)  # [T, 3]
    return joint_pos_out, body_quat_out, body_pos_out, quat_changed
