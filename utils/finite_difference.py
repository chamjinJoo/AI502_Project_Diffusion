"""Finite-difference utilities for motion preprocessing."""

from __future__ import annotations

import numpy as np


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Compute first-order velocity with the same shape as values.

    Args:
        values: Array with shape [T, D].
        dt: Seconds per frame.
    """
    if values.shape[0] < 2:
        return np.zeros_like(values)
    vel = np.zeros_like(values, dtype=np.float32)
    vel[1:] = (values[1:] - values[:-1]) / dt
    vel[0] = vel[1]
    return vel
