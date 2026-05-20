"""Evaluate a checkpoint on an existing reference-space condition/target batch."""

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

from datasets.motion_chunk_dataset import (  # noqa: E402
    apply_model_space_transforms,
    decode_future_model_space,
    load_stats,
)
from scripts.evaluate_checkpoint_windows import _load_model  # noqa: E402
from scripts.evaluate_reference_quality import reference_quality_metrics  # noqa: E402


DEFAULT_LABELS = [
    "easy_01_jump_ff_270_r_002",
    "easy_02_moonwalk_r_001",
    "easy_03_jog_ff_stop_180_r_001",
    "medium_01_big_heavy_two_hands_walk_ff_loop_270_r_001",
    "medium_02_baby_full_diaper_turn_walk_ff_270_start_r_003",
    "medium_03_walk_forward_loop_003",
    "hard_01_jog_ff_loop_180_r_very_fast_002",
    "hard_02_turn_run_360_r_002",
    "hard_03_turn_run_270_r_003",
]


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint while remaining compatible with old torch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cond_batch", required=True, help="Reference-space histories [N,H,65]")
    parser.add_argument("--target_batch", required=True, help="Reference-space targets [N,K,65]")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_named", type=int, default=9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(args.checkpoint)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    fps = float(data_cfg.get("fps", 50.0))
    joint_vel_mode = str(data_cfg.get("joint_vel_mode", "source"))
    body_pos_mode = str(data_cfg.get("body_pos_mode", "relative"))
    pred_len = int(data_cfg["pred_len"])
    frame_dim = int(data_cfg["frame_dim"])
    steps = int(args.num_inference_steps or cfg["diffusion"].get("num_inference_steps", 20))

    cond_ref = np.load(args.cond_batch).astype(np.float32)  # [N, H, 65], already reference-space
    target_ref = np.load(args.target_batch).astype(np.float32)  # [N, K, 65], already reference-space
    if cond_ref.ndim != 3 or target_ref.ndim != 3 or cond_ref.shape[-1] != 65 or target_ref.shape[-1] != 65:
        raise ValueError(f"expected [N,H,65] and [N,K,65], got {cond_ref.shape} and {target_ref.shape}")
    if target_ref.shape[1] != pred_len:
        raise ValueError(f"target horizon {target_ref.shape[1]} does not match checkpoint pred_len {pred_len}")

    mean, std = load_stats(data_cfg["stats_path"])
    mean = mean.astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)

    cond_model = []
    for cond in cond_ref:
        cond_t, _ = apply_model_space_transforms(
            cond,
            np.empty((0, frame_dim), dtype=np.float32),
            fps=fps,
            joint_vel_mode=joint_vel_mode,
            body_pos_mode=body_pos_mode,
        )
        cond_model.append(cond_t)
    cond_norm = ((np.stack(cond_model) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)

    _, sampler = _load_model(ckpt, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    cond_tensor = torch.from_numpy(cond_norm).to(device)
    x_t = torch.randn(cond_tensor.shape[0], pred_len, frame_dim, device=device, generator=generator)
    with torch.no_grad():
        pred_norm = sampler.sample(cond_tensor, pred_len, frame_dim, steps, x_T=x_t, eta=0.0)
    pred_model = pred_norm.detach().cpu().numpy() * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

    preds = []
    for pred, cond in zip(pred_model, cond_ref, strict=True):
        preds.append(
            decode_future_model_space(
                pred,
                fps=fps,
                joint_vel_mode=joint_vel_mode,
                body_pos_mode=body_pos_mode,
                prev_joint_pos=cond[-1, :29],
            )
        )
    pred_ref = np.stack(preds).astype(np.float32)
    metrics = reference_quality_metrics(pred_ref, target=target_ref, cond=cond_ref, fps=fps)
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["num_eval"] = int(cond_ref.shape[0])
    metrics["num_inference_steps"] = int(steps)
    metrics["source_cond_batch"] = str(args.cond_batch)
    metrics["source_target_batch"] = str(args.target_batch)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cond_batch.npy", cond_ref)
    np.save(out / "target_batch.npy", target_ref)
    np.save(out / "pred_batch.npy", pred_ref)

    labels = DEFAULT_LABELS[: int(args.num_named)]
    for idx, label in enumerate(labels):
        if idx >= cond_ref.shape[0]:
            break
        np.save(out / f"{label}_cond.npy", cond_ref[idx])
        np.save(out / f"{label}_target.npy", target_ref[idx])
        np.save(out / f"{label}_pred.npy", pred_ref[idx])

    (out / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
