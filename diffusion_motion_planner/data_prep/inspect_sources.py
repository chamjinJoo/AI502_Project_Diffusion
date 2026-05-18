"""Inspect source motion files before writing conversion assumptions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.motion_io import ensure_dir, find_files, load_yaml, read_numeric_csv, write_json
from utils.schema_utils import inspect_schema
from utils.source_filter import filter_source_paths


def inspect_sources(config_path: str | Path) -> list[dict]:
    """Inspect CSV source files and write schema reports."""
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_root"])
    reports_dir = ensure_dir(output_root / "reports")
    all_csv_paths = find_files(cfg["source_roots"], (".csv",))
    csv_paths = filter_source_paths(all_csv_paths, cfg)
    progress_every = int(cfg.get("progress_every", 100))
    start_time = time.time()
    print(f"[inspect] found {len(all_csv_paths)} CSV files, selected {len(csv_paths)} after filters", flush=True)
    if not csv_paths:
        print("[inspect] no CSV files found under source_roots")

    reports: list[dict] = []
    for index, path in enumerate(csv_paths, start=1):
        try:
            columns = read_numeric_csv(path)
            report = inspect_schema(path, columns, expected_joint_count=int(cfg["expected_joint_count"]))
            report["source_type"] = "bones_seed_csv"
            reports.append(report)
        except Exception as exc:
            reports.append({"path": str(path), "error": str(exc), "source_type": "bones_seed_csv"})
        if index == 1 or index % progress_every == 0 or index == len(csv_paths):
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"[inspect] {index}/{len(csv_paths)} files ({index / elapsed:.2f} files/s)", flush=True)

    write_json(reports_dir / "schema_report.json", reports)
    summary_lines = [
        f"inspected_csv_files: {len(reports)}",
        f"valid_29_joint_pos: {sum(1 for row in reports if row.get('has_29_joint_positions'))}",
        f"has_joint_vel: {sum(1 for row in reports if row.get('has_joint_velocities'))}",
        f"has_quat: {sum(1 for row in reports if row.get('has_root_quaternion'))}",
        f"has_root_euler_xyz: {sum(1 for row in reports if row.get('has_root_euler_xyz'))}",
        f"has_body_pos: {sum(1 for row in reports if row.get('has_root_position'))}",
        f"has_fps: {sum(1 for row in reports if row.get('fps') is not None)}",
    ]
    if bool(cfg.get("joint_pos_only_mode", False)):
        summary_lines.append("joint_pos_only_mode: true")
        summary_lines.append(f"assume_fps_if_missing: {cfg.get('assume_fps_if_missing')}")
    (reports_dir / "schema_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("[inspect] " + " | ".join(summary_lines), flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    args = parser.parse_args()
    reports = inspect_sources(args.config)
    print(f"wrote schema report for {len(reports)} CSV files")


if __name__ == "__main__":
    main()
