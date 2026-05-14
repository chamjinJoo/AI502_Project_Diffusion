"""Small IO helpers for motion dataset preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    """Write pretty JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Read JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_files(roots: list[str | Path], suffixes: tuple[str, ...]) -> list[Path]:
    """Recursively find files under one or more roots."""
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix.lower() in suffixes:
            paths.append(root_path)
        elif root_path.exists():
            for suffix in suffixes:
                paths.extend(root_path.rglob(f"*{suffix}"))
    return sorted(set(paths))


def read_csv_header(path: str | Path) -> list[str]:
    """Read only the CSV header."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader)


def read_numeric_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a CSV into numeric column arrays, coercing non-numeric values to NaN."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        raw: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    raw[name].append(float(value))
                except (TypeError, ValueError):
                    raw[name].append(float("nan"))
    return {name: np.asarray(values, dtype=np.float32) for name, values in raw.items()}


def save_sequence(path: str | Path, sequence: np.ndarray) -> None:
    """Save a [T, 65] sequence."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, sequence.astype(np.float32))
