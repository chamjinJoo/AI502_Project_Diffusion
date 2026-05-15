"""Create train/val JSONL manifests from all_manifest.jsonl."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.motion_io import write_jsonl, load_yaml, read_jsonl


def make_splits(config_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split all_manifest into train and validation manifests."""
    cfg = load_yaml(config_path)
    manifests_dir = Path(cfg["output_root"]) / "manifests"
    rows = read_jsonl(manifests_dir / "all_manifest.jsonl")
    rows = [row for row in rows if row.get("validation_passed", False)]
    rng = random.Random(int(cfg["split_seed"]))

    if bool(cfg["grouped_split"]):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group = str(row.get("date") or row.get("category") or Path(row["source_path"]).parent)
            groups[group].append(row)
        group_keys = list(groups)
        rng.shuffle(group_keys)
        val_group_count = max(1, int(round(len(group_keys) * float(cfg["val_ratio"])))) if group_keys else 0
        val_groups = set(group_keys[:val_group_count])
        train_rows = [row for key in group_keys if key not in val_groups for row in groups[key]]
        val_rows = [row for key in group_keys if key in val_groups for row in groups[key]]
    else:
        rng.shuffle(rows)
        val_count = int(round(len(rows) * float(cfg["val_ratio"])))
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]

    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"

    write_jsonl(manifests_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(manifests_dir / "val_manifest.jsonl", val_rows)
    write_jsonl(manifests_dir / "all_manifest.jsonl", train_rows + val_rows)
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset_build.yaml")
    args = parser.parse_args()
    train_rows, val_rows = make_splits(args.config)
    print(f"train sequences: {len(train_rows)}")
    print(f"val sequences: {len(val_rows)}")


if __name__ == "__main__":
    main()
