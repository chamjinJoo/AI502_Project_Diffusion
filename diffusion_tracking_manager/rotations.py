"""Small rotation wrapper for SONIC-compatible wxyz quaternion tensors."""

from __future__ import annotations

import torch

try:
    # Reuse the repository's rotation code so the manager follows SONIC's rot6 convention.
    from gear_sonic.trl.utils import torch_transform as _sonic_tf
except Exception:  # pragma: no cover - fallback is for minimal import-only environments.
    _sonic_tf = None


def normalize_quat(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize wxyz quaternions while preserving batched dimensions."""
    _assert_last_dim(quat, 4, "quat")
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a normalized wxyz quaternion."""
    _assert_last_dim(quat, 4, "quat")
    if _sonic_tf is not None:
        return _sonic_tf.quat_inv(quat)
    quat = normalize_quat(quat)
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two wxyz quaternions with broadcast-compatible shapes."""
    _assert_last_dim(a, 4, "a")
    _assert_last_dim(b, 4, "b")
    if _sonic_tf is not None:
        return _sonic_tf.quat_mul(a, b)

    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    out = torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
    return normalize_quat(out)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to SONIC's first-two-rotation-columns rot6 format."""
    _assert_last_dim(quat, 4, "quat")
    quat = normalize_quat(quat)
    if _sonic_tf is not None:
        return _sonic_tf.quat_to_rot6d(quat)
    return matrix_to_rot6d(quat_to_matrix(quat))


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the first two columns of a rotation matrix."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape [..., 3, 3], got {tuple(matrix.shape)}")
    if _sonic_tf is not None:
        return _sonic_tf.rotmat_to_rot6d(matrix)
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def relative_quat_from_world(current_quat_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
    """Compute target orientation in the current robot frame."""
    if current_quat_w.dim() == target_quat_w.dim() - 1:
        current_quat_w = current_quat_w.unsqueeze(1).expand_as(target_quat_w)
    return quat_mul(quat_inv(current_quat_w), target_quat_w)


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Fallback quaternion-to-matrix conversion for wxyz tensors."""
    quat = normalize_quat(quat)
    w, x, y, z = torch.unbind(quat, dim=-1)
    two_s = 2.0 / (quat * quat).sum(dim=-1).clamp_min(1e-8)
    matrix = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return matrix.reshape(quat.shape[:-1] + (3, 3))


def _assert_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} must have last dimension {expected}, got {tuple(tensor.shape)}")
