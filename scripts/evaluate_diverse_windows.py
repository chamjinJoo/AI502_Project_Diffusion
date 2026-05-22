"""Evaluate and export diverse motion windows for visualization.

This is a small diagnostic helper. It selects validation windows by actual
motion score, skipping near-static clips, then exports easy/medium/hard chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import decode_future_model_space, load_checkpoint_stats_or_file
from scripts.evaluate_checkpoint_windows import _load_model, _manifest_paths, _select_windows, _window_to_arrays
from scripts.evaluate_reference_quality import reference_quality_metrics


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint with compatibility across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _choose_diverse_windows(
    candidates: list[tuple[Path, int, float]],
    min_motion_score: float,
) -> list[tuple[Path, int, float]]:
    """Choose three easy, three medium, and three hard non-static windows."""
    moving = [item for item in candidates if item[2] >= min_motion_score]
    moving.sort(key=lambda item: item[2])
    if len(moving) < 9:
        raise ValueError(f"Need at least 9 moving candidates, got {len(moving)}")

    quantiles = [0.08, 0.16, 0.24, 0.42, 0.50, 0.58, 0.76, 0.86, 0.96]
    chosen: list[tuple[Path, int, float]] = []
    used: set[int] = set()
    for quantile in quantiles:
        center = int(round(quantile * (len(moving) - 1)))
        selected = None
        for radius in range(len(moving)):
            for idx in (center - radius, center + radius):
                if 0 <= idx < len(moving) and idx not in used:
                    selected = idx
                    break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError("Could not select a diverse visual window")
        used.add(selected)
        chosen.append(moving[selected])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_count", type=int, default=512)
    parser.add_argument("--min_motion_score", type=float, default=0.04)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=149)
    args = parser.parse_args()

    ckpt = _load_checkpoint(Path(args.checkpoint))
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    manifest = data_cfg.get("val_file_list") or data_cfg.get("train_file_list")
    paths = _manifest_paths(manifest)
    mean, std = load_checkpoint_stats_or_file(ckpt, data_cfg["stats_path"])
    mean = mean.astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)

    candidates = _select_windows(paths, cfg, args.candidate_count, args.seed)
    chosen = _choose_diverse_windows(candidates, args.min_motion_score)
    labels = [f"easy_{i:02d}" for i in range(1, 4)]
    labels += [f"medium_{i:02d}" for i in range(1, 4)]
    labels += [f"hard_{i:02d}" for i in range(1, 4)]

    cond_norms: list[np.ndarray] = []
    cond_refs: list[np.ndarray] = []
    cond_eval_refs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for label, (path, timestep, score) in zip(labels, chosen, strict=True):
        sequence = np.load(path, mmap_mode="r")
        cond_norm, cond_eval, target_eval, cond_ref = _window_to_arrays(sequence, timestep, cfg, mean, std)
        cond_norms.append(cond_norm)
        cond_eval_refs.append(cond_eval)
        cond_refs.append(cond_ref)
        targets.append(target_eval)
        metadata.append(
            {
                "label": label,
                "path": str(path),
                "t": int(timestep),
                "motion_score": float(score),
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, sampler = _load_model(ckpt, device)
    pred_len = int(data_cfg["pred_len"])
    frame_dim = int(data_cfg["frame_dim"])
    cond_tensor = torch.from_numpy(np.stack(cond_norms)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x_t = torch.randn(cond_tensor.shape[0], pred_len, frame_dim, device=device, generator=generator)
    with torch.no_grad():
        pred_norm = sampler.sample(cond_tensor, pred_len, frame_dim, args.num_inference_steps, x_T=x_t, eta=0.0)
    pred_model = pred_norm.detach().cpu().numpy() * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

    preds: list[np.ndarray] = []
    for pred, cond_ref in zip(pred_model, cond_refs, strict=True):
        preds.append(
            decode_future_model_space(
                pred,
                fps=float(data_cfg.get("fps", 50.0)),
                joint_vel_mode=str(data_cfg.get("joint_vel_mode", "source")),
                body_pos_mode=str(data_cfg.get("body_pos_mode", "relative")),
                prev_joint_pos=cond_ref[-1, :29],
            )
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cond_batch = np.stack(cond_eval_refs).astype(np.float32)
    target_batch = np.stack(targets).astype(np.float32)
    pred_batch = np.stack(preds).astype(np.float32)
    np.save(out / "cond_batch.npy", cond_batch)
    np.save(out / "target_batch.npy", target_batch)
    np.save(out / "pred_batch.npy", pred_batch)

    for idx, item in enumerate(metadata):
        stem = f"{item['label']}_score_{item['motion_score']:.3f}"
        np.save(out / f"{stem}_cond.npy", cond_batch[idx])
        np.save(out / f"{stem}_target.npy", target_batch[idx])
        np.save(out / f"{stem}_pred.npy", pred_batch[idx])

    metrics = reference_quality_metrics(
        pred_batch,
        target=target_batch,
        cond=cond_batch,
        fps=float(data_cfg.get("fps", 50.0)),
    )
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["num_eval"] = int(len(metadata))
    metrics["num_inference_steps"] = int(args.num_inference_steps)
    payload = {"metrics": metrics, "visual_samples": metadata}
    (out / "evaluation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
