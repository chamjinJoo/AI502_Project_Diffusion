"""Quaternion helpers using (w, x, y, z) order."""

from __future__ import annotations

import numpy as np


def normalize_quat(quat: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Normalize quaternions with shape [T, 4]."""
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    changed = bool(np.nanmax(np.abs(norms - 1.0)) > 1e-5)
    return quat / np.maximum(norms, eps), changed


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return quaternion conjugates in (w, x, y, z) order."""
    q = np.asarray(quat, dtype=np.float32)
    out = q.copy()
    out[..., 1:4] *= -1.0
    return out.astype(np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (w, x, y, z) order."""
    a = np.asarray(q1, dtype=np.float32)
    b = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors by inverse quaternion rotation.

    Args:
        quat: Quaternion(s) with shape [..., 4] in (w, x, y, z).
        vec: Vector(s) with shape [..., 3].
    """
    q = normalize_quat(np.asarray(quat, dtype=np.float32))[0]
    q_inv = quat_conjugate(q)
    v = np.asarray(vec, dtype=np.float32)
    zeros = np.zeros(v.shape[:-1] + (1,), dtype=np.float32)
    v_quat = np.concatenate([zeros, v], axis=-1)
    rotated = quat_multiply(quat_multiply(q_inv, v_quat), q)
    return rotated[..., 1:4].astype(np.float32)


def quat_norm_error(quat: np.ndarray) -> np.ndarray:
    """Return absolute unit-norm error for [T, 4] quaternions."""
    return np.abs(np.linalg.norm(quat, axis=-1) - 1.0)


def identity_quat(num_frames: int) -> np.ndarray:
    """Create [T, 4] identity quaternions in (w, x, y, z)."""
    quat = np.zeros((num_frames, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat


def euler_xyz_to_quat(euler_xyz: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert XYZ Euler angles with shape [T, 3] to quaternions in (w, x, y, z)."""
    angles = np.deg2rad(euler_xyz) if degrees else euler_xyz
    roll = angles[:, 0] * 0.5
    pitch = angles[:, 1] * 0.5
    yaw = angles[:, 2] * 0.5

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    quat = np.empty((euler_xyz.shape[0], 4), dtype=np.float32)
    quat[:, 0] = cr * cp * cy + sr * sp * sy
    quat[:, 1] = sr * cp * cy - cr * sp * sy
    quat[:, 2] = cr * sp * cy + sr * cp * sy
    quat[:, 3] = cr * cp * sy - sr * sp * cy
    return normalize_quat(quat)[0]
