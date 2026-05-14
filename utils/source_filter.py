"""Source-file filtering helpers for dataset builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def filter_source_paths(paths: list[Path], cfg: dict[str, Any]) -> list[Path]:
    """Filter paths by include/exclude keywords from config."""
    include = [str(item).lower() for item in cfg.get("include_keywords", []) if str(item)]
    exclude = [str(item).lower() for item in cfg.get("exclude_keywords", []) if str(item)]
    filtered: list[Path] = []
    for path in paths:
        text = str(path).lower()
        if include and not any(token in text for token in include):
            continue
        if exclude and any(token in text for token in exclude):
            continue
        filtered.append(path)
    return filtered
