"""Column-schema inspection and heuristic mapping for humanoid CSV trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


AXES = ("x", "y", "z")
QUAT_AXES = ("w", "x", "y", "z")


def norm_name(name: str) -> str:
    """Normalize a column name for heuristic matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _contains_any(name: str, tokens: tuple[str, ...]) -> bool:
    return any(token in name for token in tokens)


def infer_fps(columns: dict[str, np.ndarray], path: str | Path | None = None) -> tuple[float | None, str | None]:
    """Infer fps from time/dt columns or filename hints."""
    normalized = {norm_name(key): key for key in columns}
    for key in ("time", "timestamp", "t", "sec", "seconds"):
        if key in normalized:
            values = columns[normalized[key]]
            diffs = np.diff(values[np.isfinite(values)])
            diffs = diffs[diffs > 0]
            if len(diffs):
                return float(1.0 / np.median(diffs)), normalized[key]
    for key in ("dt", "delta_t", "timestep"):
        if key in normalized:
            values = columns[normalized[key]]
            finite = values[np.isfinite(values)]
            finite = finite[finite > 0]
            if len(finite):
                return float(1.0 / np.median(finite)), normalized[key]
    if path is not None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*fps", str(path).lower())
        if match:
            return float(match.group(1)), "filename"
    return None, None


def infer_joint_columns(column_names: list[str], expected_count: int = 29, velocity: bool = False) -> list[str]:
    """Infer joint position or velocity columns from names."""
    scored: list[tuple[int, int, str]] = []
    for idx, original in enumerate(column_names):
        name = norm_name(original)
        is_vel = _contains_any(name, ("vel", "velocity", "dof_vel", "qvel", "qd"))
        is_pos = _contains_any(name, ("joint", "dof", "motor", "qpos", "angle"))
        if velocity and not is_vel:
            continue
        if not velocity and is_vel:
            continue
        if not velocity and not is_pos:
            continue
        score = 0
        if "joint" in name:
            score += 4
        if "dof" in name:
            score += 3
        if "qpos" in name or "qvel" in name:
            score += 3
        if "angle" in name:
            score += 2
        if velocity:
            score += 2 if is_vel else 0
        scored.append((-score, idx, original))
    scored.sort()
    return [item[2] for item in scored[:expected_count]]


def infer_vector_columns(column_names: list[str], kind: str, axes: tuple[str, ...]) -> list[str]:
    """Infer ordered vector columns for root position or root quaternion."""
    normalized = {norm_name(name): name for name in column_names}
    prefixes = {
        "body_pos": ("root_pos", "root_translate", "base_pos", "pelvis_pos", "torso_pos", "body_pos", "root_position", "position"),
        "body_quat": ("root_quat", "base_quat", "pelvis_quat", "torso_quat", "body_quat", "root_orientation", "quat"),
    }[kind]
    for prefix in prefixes:
        found: list[str] = []
        for axis in axes:
            candidates = (
                f"{prefix}_{axis}",
                f"{prefix}{axis}",
                f"{axis}_{prefix}",
            )
            matched = next((normalized[key] for key in candidates if key in normalized), None)
            if matched is not None:
                found.append(matched)
        if len(found) == len(axes):
            return found

    axis_matches: list[str] = []
    required = ("pos", "position") if kind == "body_pos" else ("quat", "orientation")
    for axis in axes:
        matches = [
            original
            for original in column_names
            if axis in norm_name(original).split("_") and _contains_any(norm_name(original), required)
        ]
        if not matches:
            return []
        axis_matches.append(matches[0])
    return axis_matches


def inspect_schema(path: str | Path, columns: dict[str, np.ndarray], expected_joint_count: int = 29) -> dict[str, Any]:
    """Inspect one CSV file and report candidate source schema facts."""
    names = list(columns.keys())
    fps, fps_source = infer_fps(columns, path)
    joint_pos_cols = infer_joint_columns(names, expected_joint_count, velocity=False)
    joint_vel_cols = infer_joint_columns(names, expected_joint_count, velocity=True)
    quat_cols = infer_vector_columns(names, "body_quat", QUAT_AXES)
    root_euler_cols = infer_vector_columns(names, "body_pos", ("rotatex", "rotatey", "rotatez"))
    if not root_euler_cols:
        normalized = {norm_name(name): name for name in names}
        candidates = ["root_rotatex", "root_rotatey", "root_rotatez"]
        root_euler_cols = [normalized[key] for key in candidates if key in normalized]
    pos_cols = infer_vector_columns(names, "body_pos", AXES)
    frame_count = len(next(iter(columns.values()))) if columns else 0
    return {
        "path": str(path),
        "column_names": names,
        "num_columns": len(names),
        "frame_count": frame_count,
        "joint_pos_columns": joint_pos_cols,
        "joint_vel_columns": joint_vel_cols,
        "body_quat_columns": quat_cols,
        "root_euler_xyz_columns": root_euler_cols if len(root_euler_cols) == 3 else [],
        "body_pos_columns": pos_cols,
        "has_29_joint_positions": len(joint_pos_cols) == expected_joint_count,
        "has_joint_velocities": len(joint_vel_cols) == expected_joint_count,
        "has_root_quaternion": len(quat_cols) == 4,
        "has_root_euler_xyz": len(root_euler_cols) == 3,
        "has_root_position": len(pos_cols) == 3,
        "fps": fps,
        "fps_source": fps_source,
    }


def source_info_from_path(path: str | Path) -> dict[str, str | None]:
    """Infer lightweight date/category metadata from path parts."""
    parts = Path(path).parts
    date = next((part for part in parts if re.match(r"\d{4}[-_]?\d{2}[-_]?\d{2}", part)), None)
    category = parts[-2] if len(parts) >= 2 else None
    return {"date": date, "category": category}
