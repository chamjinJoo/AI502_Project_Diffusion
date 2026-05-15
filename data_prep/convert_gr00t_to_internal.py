"""Convert already-existing GR00T-style [T, 65] .npy folders to processed format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.motion_io import write_jsonl, ensure_dir, find_files, load_yaml, save_sequence, write_json
from utils.quaternion_utils import normalize_quat, quat_norm_error
from utils.schema_utils import source_info_from_path
from utils.source_filter import filter_source_paths


def convert_gr00t(config_path: str | Path) -> list[dict[str, Any]]:
    """Copy valid [T, 65] .npy sequences into processed_dataset/sequences."""
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_root"])
    sequences_dir = ensure_dir(output_root / "sequences")
    manifests_dir = ensure_dir(output_root / "manifests")
    reports_dir = ensure_dir(output_root / "reports")
    npy_paths = filter_source_paths(find_files(cfg["source_roots"], (".npy",)), cfg)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, path in enumerate(npy_paths, start=1):
        sequence_id = f"motion_{index:06d}"
        out_path = sequences_dir / f"{sequence_id}.npy"
        try:
            sequence = np.load(path).astype(np.float32)
            reasons: list[str] = []
            if sequence.ndim != 2 or sequence.shape[1] != 65:
                reasons.append(f"bad_shape:{sequence.shape}")
            elif sequence.shape[0] < int(cfg["min_sequence_length"]):
                reasons.append("too_short")
            elif not np.isfinite(sequence).all():
                reasons.append("non_finite")
            elif float(np.max(quat_norm_error(sequence[:, 58:62]))) > float(cfg["quaternion_norm_tolerance"]):
                if bool(cfg["renormalize_quaternion"]):
                    sequence[:, 58:62], _ = normalize_quat(sequence[:, 58:62])
                else:
                    reasons.append("quat_norm_error")
            if reasons:
                skipped.append({"source_path": str(path), "skip_reasons": reasons})
                continue
            save_sequence(out_path, sequence)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "source_type": "gr00t_npy",
                    "source_path": str(path),
                    "processed_npy_path": str(out_path),
                    "num_frames": int(sequence.shape[0]),
                    "fps_original": None,
                    "fps_output": float(cfg["target_fps"]),
                    "shape": list(sequence.shape),
                    "used_joint_vel_fallback": False,
                    "used_body_pos_fallback": False,
                    "used_body_quat_debug_fallback": False,
                    "quaternion_renormalized": bool(cfg["renormalize_quaternion"]),
                    "validation_passed": True,
                    "split": None,
                    **source_info_from_path(path),
                }
            )
        except Exception as exc:
            skipped.append({"source_path": str(path), "skip_reasons": [str(exc)]})

    write_jsonl(manifests_dir / "all_manifest.jsonl", rows)
    write_json(reports_dir / "conversion_report.json", {"converted": len(rows), "skipped": len(skipped), "skipped_items": skipped})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    args = parser.parse_args()
    rows = convert_gr00t(args.config)
    print(f"converted {len(rows)} GR00T-style sequences")


if __name__ == "__main__":
    main()
