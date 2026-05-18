"""End-to-end dataset build orchestration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data_prep.compute_stats import compute_stats
from data_prep.convert_bones_seed_to_internal import convert_bones_seed
from data_prep.inspect_sources import inspect_sources
from data_prep.make_splits import make_splits
from data_prep.sanity_check import sanity_check
from utils.motion_io import find_files, load_yaml
from utils.source_filter import filter_source_paths


def preflight_sources(config_path: str | Path) -> None:
    """Fail early when configured source roots contain no source files."""
    cfg = load_yaml(config_path)
    roots = [Path(root) for root in cfg["source_roots"]]
    missing = [str(root) for root in roots if not root.exists()]
    csv_paths = filter_source_paths(find_files(cfg["source_roots"], (".csv",)), cfg)
    npy_paths = filter_source_paths(find_files(cfg["source_roots"], (".npy",)), cfg)
    if missing:
        print("[warn] missing source_roots:")
        for root in missing:
            print(f"  {root}")
    if not csv_paths and not npy_paths:
        raise FileNotFoundError(
            "No source motion files found. Update configs/dataset_build.yaml source_roots "
            "to the BONES-SEED/G1 CSV directory or an existing GR00T-style .npy directory. "
            "Expected files: .csv or .npy."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    parser.add_argument("--skip_inspect", action="store_true")
    parser.add_argument("--skip_sanity", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    total_start = time.time()
    print("[build] preflight start", flush=True)
    preflight_sources(args.config)
    print("[build] preflight done", flush=True)
    if not args.skip_inspect:
        print("[build] inspect start", flush=True)
        inspect_sources(args.config)
        print("[build] inspect done", flush=True)
    print("[build] convert start", flush=True)
    rows = convert_bones_seed(args.config, force_rebuild=args.force_rebuild)
    if not rows:
        raise ValueError("No valid sequences were converted. Check reports/conversion_report.json.")
    print("[build] split start", flush=True)
    make_splits(args.config)
    print("[build] split done", flush=True)
    print("[build] stats start", flush=True)
    compute_stats(args.config)
    print("[build] stats done", flush=True)
    if not args.skip_sanity:
        print("[build] sanity start", flush=True)
        sanity_check(args.config)
        print("[build] sanity done", flush=True)
    print(f"[build] dataset build complete in {time.time() - total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
