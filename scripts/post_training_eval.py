"""Run checkpoint evaluation and MuJoCo-FK visualization on the head node.

This script intentionally stays outside the SLURM training path. It evaluates a
finished checkpoint, exports visual sample chunks, and renders GIFs locally so
compute-node workdirs do not accumulate post-training artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_visual_samples(summary_path: Path) -> list[dict[str, Any]]:
    """Read visual sample metadata written by evaluate_checkpoint_windows.py."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = payload.get("visual_samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"visual_samples must be a list in {summary_path}")
    return samples


def _safe_stem(path: str | Path) -> str:
    """Return a compact filesystem-safe stem for a source motion path."""
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "motion"


def _run(cmd: list[str]) -> None:
    """Run a subprocess while echoing the command for reproducibility."""
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/latest.pt checkpoint")
    parser.add_argument("--output_dir", required=True, help="Head-node output directory under samples/ or tmp/")
    parser.add_argument("--manifest", default=None, help="Optional manifest override")
    parser.add_argument("--stats", default=None, help="Optional stats override for old checkpoints")
    parser.add_argument("--num_eval", type=int, default=256)
    parser.add_argument("--num_visual", type=int, default=18, help="Number of motion windows to export and render")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument(
        "--xml",
        default="AI502TermProject-main/gear_sonic/data/robots/g1/g1_29dof_old.xml",
        help="G1 MJCF for MuJoCo FK rendering",
    )
    parser.add_argument("--gif_fps", type=int, default=8)
    parser.add_argument("--base_height", type=float, default=0.8)
    parser.add_argument("--skip_gifs", action="store_true", help="Only run numeric evaluation/export")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable,
        "scripts/evaluate_checkpoint_windows.py",
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        str(out),
        "--num_eval",
        str(args.num_eval),
        "--num_visual",
        str(args.num_visual),
        "--seed",
        str(args.seed),
        "--num_inference_steps",
        str(args.num_inference_steps),
    ]
    if args.manifest is not None:
        eval_cmd += ["--manifest", args.manifest]
    if args.stats is not None:
        eval_cmd += ["--stats", args.stats]
    _run(eval_cmd)

    summary_path = out / "evaluation_summary.json"
    if args.skip_gifs:
        print(f"[done] numeric evaluation saved to {summary_path}", flush=True)
        return

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {xml_path}. Use --skip_gifs for metrics only, "
            "or pass --xml to the G1 MJCF path."
        )

    gif_dir = out / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_manifest: list[dict[str, str]] = []
    for sample in _load_visual_samples(summary_path):
        label = str(sample["label"])
        source_stem = _safe_stem(sample.get("path", "motion"))
        gif_name = f"{label}_{source_stem}.gif"
        gif_path = gif_dir / gif_name
        meta_path = gif_path.with_suffix(".json")
        _run(
            [
                sys.executable,
                "scripts/visualize_reference_mujoco_fk.py",
                "--cond",
                str(out / f"{label}_cond.npy"),
                "--target",
                str(out / f"{label}_target.npy"),
                "--pred",
                str(out / f"{label}_pred.npy"),
                "--xml",
                str(xml_path),
                "--output",
                str(gif_path),
                "--meta_output",
                str(meta_path),
                "--fps",
                str(args.gif_fps),
                "--base_height",
                str(args.base_height),
            ]
        )
        gif_manifest.append({"label": label, "source_path": str(sample.get("path", "")), "gif": str(gif_path)})

    (gif_dir / "gif_manifest.json").write_text(json.dumps(gif_manifest, indent=2), encoding="utf-8")
    print(f"[done] evaluation saved to {summary_path}", flush=True)
    print(f"[done] gifs saved to {gif_dir}", flush=True)


if __name__ == "__main__":
    main()
