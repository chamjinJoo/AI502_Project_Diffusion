"""Convert BONES-SEED Unitree G1 CSV trajectories to internal [T, 65] arrays."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.finite_difference import finite_difference
from utils.motion_io import write_jsonl, ensure_dir, find_files, load_yaml, read_jsonl, read_numeric_csv, save_sequence, write_json
from utils.quaternion_utils import euler_xyz_to_quat, identity_quat, normalize_quat, quat_norm_error
from utils.resample_utils import resample_array, resample_motion_parts
from utils.schema_utils import inspect_schema, source_info_from_path
from utils.source_filter import filter_source_paths


def preprocessing_assumptions(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return assumptions that affect converted [T, 65] arrays."""
    return {
        "target_fps": float(cfg["target_fps"]),
        "assume_fps_if_missing": cfg.get("assume_fps_if_missing"),
        "source_joint_unit": cfg.get("source_joint_unit", "radians"),
        "source_root_pos_unit": cfg.get("source_root_pos_unit", "meters"),
        "source_root_euler_unit": cfg.get("source_root_euler_unit", "degrees"),
        "joint_pos_only_mode": bool(cfg.get("joint_pos_only_mode", False)),
        "allow_joint_vel_fallback": bool(cfg.get("allow_joint_vel_fallback", False)),
        "allow_body_pos_zero_fallback": bool(cfg.get("allow_body_pos_zero_fallback", False)),
        "allow_body_quat_identity_debug_fallback": bool(cfg.get("allow_body_quat_identity_debug_fallback", False)),
        "renormalize_quaternion": bool(cfg.get("renormalize_quaternion", True)),
        "expected_joint_count": int(cfg["expected_joint_count"]),
    }


def _assumptions_match(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Check whether a reusable manifest row was built with current assumptions."""
    expected = preprocessing_assumptions(cfg)
    previous = row.get("preprocessing_assumptions")
    if previous is None:
        return False
    return previous == expected


def _stack_columns(columns: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Stack named scalar columns into [T, D]."""
    return np.stack([columns[name] for name in names], axis=1).astype(np.float32)


def _convert_joint_units(joint_pos: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Convert joint angles to radians for GR00T/IsaacLab compatibility."""
    if str(cfg.get("source_joint_unit", "radians")).lower() in {"degree", "degrees", "deg"}:
        return np.deg2rad(joint_pos).astype(np.float32)
    return joint_pos.astype(np.float32)


def _convert_root_pos_units(body_pos: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Convert root position to meters for GR00T/IsaacLab compatibility."""
    unit = str(cfg.get("source_root_pos_unit", "meters")).lower()
    if unit in {"centimeter", "centimeters", "cm"}:
        return (body_pos * 0.01).astype(np.float32)
    if unit in {"millimeter", "millimeters", "mm"}:
        return (body_pos * 0.001).astype(np.float32)
    return body_pos.astype(np.float32)


def _validate_sequence(sequence: np.ndarray, cfg: dict[str, Any]) -> list[str]:
    """Return validation errors for a processed [T, 65] sequence."""
    reasons: list[str] = []
    min_len = int(cfg["min_sequence_length"])
    if sequence.ndim != 2 or sequence.shape[1] != 65:
        reasons.append(f"bad_shape:{sequence.shape}")
        return reasons
    if sequence.shape[0] < min_len:
        reasons.append(f"too_short:{sequence.shape[0]}<{min_len}")
    if not np.isfinite(sequence).all():
        reasons.append("non_finite")
    quat_err = quat_norm_error(sequence[:, 58:62])
    if float(np.max(quat_err)) > float(cfg["quaternion_norm_tolerance"]):
        reasons.append(f"quat_norm_error:{float(np.max(quat_err)):.6f}")
    if float(np.max(np.abs(sequence[:, :58]))) > 1e4:
        reasons.append("joint_values_out_of_range")
    if float(np.max(np.abs(sequence[:, 62:65]))) > 1e5:
        reasons.append("body_pos_out_of_range")
    return reasons


def convert_one_csv(path: Path, sequence_id: str, output_path: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None]:
    """Convert one CSV file to [T, 65], returning manifest metadata and sequence."""
    columns = read_numeric_csv(path)
    schema = inspect_schema(path, columns, expected_joint_count=int(cfg["expected_joint_count"]))
    joint_pos_only_mode = bool(cfg.get("joint_pos_only_mode", False))
    assumed_fps = cfg.get("assume_fps_if_missing")
    meta: dict[str, Any] = {
        "sequence_id": sequence_id,
        "source_type": "bones_seed_csv",
        "source_path": str(path),
        "processed_npy_path": str(output_path),
        "fps_original": schema.get("fps") if schema.get("fps") is not None else assumed_fps,
        "fps_output": float(cfg["target_fps"]),
        "preprocessing_assumptions": preprocessing_assumptions(cfg),
        "joint_pos_only_mode": joint_pos_only_mode,
        "assumed_fps": schema.get("fps") is None and assumed_fps is not None,
        "source_joint_unit": cfg.get("source_joint_unit", "radians"),
        "source_root_pos_unit": cfg.get("source_root_pos_unit", "meters"),
        "source_root_euler_unit": cfg.get("source_root_euler_unit", "degrees"),
        "used_joint_vel_fallback": False,
        "used_body_pos_fallback": False,
        "used_body_quat_debug_fallback": False,
        "converted_root_euler_to_quat": False,
        "quaternion_renormalized": False,
        "validation_passed": False,
        "split": None,
        **source_info_from_path(path),
    }

    skip_reasons: list[str] = []
    if not schema["has_29_joint_positions"]:
        skip_reasons.append("missing_29_joint_positions")
    if schema["fps"] is None and assumed_fps is None:
        skip_reasons.append("missing_fps")
    if not schema["has_joint_velocities"] and not joint_pos_only_mode and not bool(cfg.get("allow_joint_vel_fallback", False)):
        skip_reasons.append("missing_joint_velocities")
    has_orientation = schema["has_root_quaternion"] or schema.get("has_root_euler_xyz", False)
    if not has_orientation and not joint_pos_only_mode and not bool(cfg["allow_body_quat_identity_debug_fallback"]):
        skip_reasons.append("missing_root_quaternion")
    if not schema["has_root_position"] and not joint_pos_only_mode and not bool(cfg["allow_body_pos_zero_fallback"]):
        skip_reasons.append("missing_root_position")
    if skip_reasons:
        meta["skip_reasons"] = skip_reasons
        return meta, None

    fps_in = float(schema["fps"] or assumed_fps)
    joint_pos = _convert_joint_units(_stack_columns(columns, schema["joint_pos_columns"]), cfg)  # [T, 29], radians
    if schema["has_joint_velocities"]:
        joint_vel = _stack_columns(columns, schema["joint_vel_columns"])  # [T, 29]
    else:
        joint_vel = finite_difference(joint_pos, dt=1.0 / fps_in)  # [T, 29], required synthetic channel in joint_pos_only_mode
        meta["used_joint_vel_fallback"] = True

    if schema["has_root_quaternion"]:
        body_quat = _stack_columns(columns, schema["body_quat_columns"])  # [T, 4], wxyz
    elif schema.get("has_root_euler_xyz", False):
        root_euler = _stack_columns(columns, schema["root_euler_xyz_columns"])  # [T, 3], XYZ
        degrees = str(cfg.get("source_root_euler_unit", "degrees")).lower() in {"degree", "degrees", "deg"}
        body_quat = euler_xyz_to_quat(root_euler, degrees=degrees)  # [T, 4], wxyz
        meta["converted_root_euler_to_quat"] = True
    else:
        body_quat = identity_quat(joint_pos.shape[0])  # [T, 4], synthetic channel in joint_pos_only_mode
        meta["used_body_quat_debug_fallback"] = True

    if schema["has_root_position"]:
        body_pos = _convert_root_pos_units(_stack_columns(columns, schema["body_pos_columns"]), cfg)  # [T, 3], meters
    else:
        body_pos = np.zeros((joint_pos.shape[0], 3), dtype=np.float32)
        meta["used_body_pos_fallback"] = True

    if bool(cfg["renormalize_quaternion"]):
        body_quat, changed = normalize_quat(body_quat)
        meta["quaternion_renormalized"] = changed

    if abs(fps_in - float(cfg["target_fps"])) > 1e-6:
        joint_pos, body_quat, body_pos, quat_changed = resample_motion_parts(
            joint_pos, body_quat, body_pos, fps_in=fps_in, fps_out=float(cfg["target_fps"])
        )
        if meta["used_joint_vel_fallback"]:
            joint_vel = finite_difference(joint_pos, dt=1.0 / float(cfg["target_fps"]))  # [T, 29]
        else:
            joint_vel = resample_array(joint_vel, fps_in=fps_in, fps_out=float(cfg["target_fps"]))  # [T, 29]
        meta["quaternion_renormalized"] = bool(meta["quaternion_renormalized"] or quat_changed)

    sequence = np.concatenate([joint_pos, joint_vel, body_quat, body_pos], axis=1).astype(np.float32)  # [T, 65]
    validation_errors = _validate_sequence(sequence, cfg)
    meta.update({"num_frames": int(sequence.shape[0]), "shape": list(sequence.shape)})
    if validation_errors:
        meta["skip_reasons"] = validation_errors
        return meta, None

    meta["validation_passed"] = True
    save_sequence(output_path, sequence)
    return meta, sequence


def convert_bones_seed(config_path: str | Path, force_rebuild: bool = False) -> list[dict[str, Any]]:
    """Convert all CSV sources under configured roots."""
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_root"])
    sequences_dir = ensure_dir(output_root / "sequences")
    manifests_dir = ensure_dir(output_root / "manifests")
    reports_dir = ensure_dir(output_root / "reports")
    all_csv_paths = find_files(cfg["source_roots"], (".csv",))
    csv_paths = filter_source_paths(all_csv_paths, cfg)
    progress_every = int(cfg.get("progress_every", 100))
    skip_existing = bool(cfg.get("skip_existing", True)) and not force_rebuild
    start_time = time.time()
    print(f"[convert] found {len(all_csv_paths)} CSV files, selected {len(csv_paths)} after filters", flush=True)

    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    reused = 0
    existing_by_source: dict[str, dict[str, Any]] = {}
    manifest_path = manifests_dir / "all_manifest.jsonl"
    if skip_existing and manifest_path.exists():
        for row in read_jsonl(manifest_path):
            processed = row.get("processed_npy_path")
            source = row.get("source_path")
            if (
                source
                and processed
                and Path(processed).exists()
                and row.get("validation_passed", False)
                and _assumptions_match(row, cfg)
            ):
                existing_by_source[str(source)] = row
    if skip_existing and manifest_path.exists() and not existing_by_source:
        print(
            "[convert] existing manifest found, but no rows match current preprocessing assumptions; "
            "rebuilding selected files",
            flush=True,
        )

    for index, path in enumerate(csv_paths, start=1):
        sequence_id = f"motion_{index:06d}"
        output_path = sequences_dir / f"{sequence_id}.npy"
        try:
            existing = existing_by_source.get(str(path))
            if existing is not None:
                manifest_rows.append(existing)
                reused += 1
            else:
                meta, sequence = convert_one_csv(path, sequence_id, output_path, cfg)
                if sequence is None:
                    skipped.append(meta)
                else:
                    manifest_rows.append(meta)
        except Exception as exc:
            skipped.append({"sequence_id": sequence_id, "source_path": str(path), "skip_reasons": [str(exc)]})
        if index == 1 or index % progress_every == 0 or index == len(csv_paths):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[convert] {index}/{len(csv_paths)} files | converted={len(manifest_rows)} "
                f"reused={reused} skipped={len(skipped)} | {index / elapsed:.2f} files/s",
                flush=True,
            )

    write_jsonl(manifests_dir / "all_manifest.jsonl", manifest_rows)
    report = {
        "preprocessing_assumptions": preprocessing_assumptions(cfg),
        "converted_or_reused": len(manifest_rows),
        "reused_existing": reused,
        "newly_converted": len(manifest_rows) - reused,
        "skipped": len(skipped),
        "skipped_items": skipped,
    }
    write_json(reports_dir / "conversion_report.json", report)
    print(
        f"[convert] done converted_or_reused={len(manifest_rows)} reused={reused} "
        f"new={len(manifest_rows) - reused} skipped={len(skipped)}",
        flush=True,
    )
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    rows = convert_bones_seed(args.config, force_rebuild=args.force_rebuild)
    print(f"converted {len(rows)} valid sequences")


if __name__ == "__main__":
    main()
