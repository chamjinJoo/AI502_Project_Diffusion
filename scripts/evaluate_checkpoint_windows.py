"""Sample and evaluate a checkpoint on deterministic validation windows."""

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
    load_checkpoint_stats_or_file,
    make_root_relative,
)
from diffusion.scheduler_wrapper import DiffusionSchedulerWrapper  # noqa: E402
from models.denoiser import ConditionalDenoiser  # noqa: E402
from scripts.evaluate import compute_metrics  # noqa: E402
from scripts.evaluate_reference_quality import reference_quality_metrics  # noqa: E402


def _manifest_paths(path: str | Path) -> list[Path]:
    """Load processed .npy paths from a JSONL manifest."""
    paths: list[Path] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            paths.append(Path(row.get("processed_npy_path", row.get("path"))))
    return paths


def _load_model(ckpt: dict[str, Any], device: torch.device) -> tuple[ConditionalDenoiser, DiffusionSchedulerWrapper]:
    """Recreate model and scheduler wrapper from a checkpoint config."""
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    conditioning_mode = str(model_cfg.get("conditioning_mode", cfg.get("conditioning_mode", "history")))
    architecture = str(model_cfg.get("architecture", "transformer"))
    model = ConditionalDenoiser(
        frame_dim=int(data_cfg["frame_dim"]),
        history_len=int(data_cfg["history_len"]),
        pred_len=int(data_cfg["pred_len"]),
        model_dim=int(model_cfg["dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        condition_encoder=str(model_cfg["condition_encoder"]),
        architecture=architecture,
        down_dims=tuple(int(dim) for dim in model_cfg.get("down_dims", [256, 512, 1024])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        n_groups=int(model_cfg.get("n_groups", 8)),
        cond_predict_scale=bool(model_cfg.get("cond_predict_scale", False)),
        condition_summary=str(model_cfg.get("condition_summary", "flatten")),
        use_time_token=bool(model_cfg.get("use_time_token", False)),
        use_segment_embedding=bool(model_cfg.get("use_segment_embedding", False)),
        use_local_condition=bool(model_cfg.get("use_local_condition", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sampler = DiffusionSchedulerWrapper(
        model,
        num_train_timesteps=int(diffusion_cfg["num_train_timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
        prediction_type=str(diffusion_cfg["prediction_type"]),
        clip_sample=bool(diffusion_cfg["clip_sample"]),
        fps=float(data_cfg.get("fps", 50.0)),
        conditioning_mode=conditioning_mode,
        objective=str(diffusion_cfg.get("objective", "epsilon")),
        flow_solver=str(diffusion_cfg.get("flow_solver", "euler")),
        joint_vel_mode=str(data_cfg.get("joint_vel_mode", "source")),
        body_pos_mode=str(data_cfg.get("body_pos_mode", "relative")),
    ).to(device)
    return model, sampler


def _window_to_arrays(
    sequence: np.ndarray,
    start_t: int,
    cfg: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return model input condition plus reference-space condition/target."""
    data_cfg = cfg["data"]
    h = int(data_cfg["history_len"])
    k = int(data_cfg["pred_len"])
    frame_dim = int(data_cfg["frame_dim"])
    fps = float(data_cfg.get("fps", 50.0))
    joint_vel_mode = str(data_cfg.get("joint_vel_mode", "source"))
    body_pos_mode = str(data_cfg.get("body_pos_mode", "relative"))

    cond = np.asarray(sequence[start_t - h + 1 : start_t + 1], dtype=np.float32)  # [H, 65]
    target = np.asarray(sequence[start_t + 1 : start_t + 1 + k], dtype=np.float32)  # [K, 65]
    if bool(data_cfg.get("root_relative", False)):
        cond_ref, target_ref = make_root_relative(cond, target)
    else:
        cond_ref, target_ref = cond.copy(), target.copy()

    # Reference-space eval arrays: root-relative body_pos, finite-difference velocity if configured.
    cond_vel, target_model = apply_model_space_transforms(
        cond_ref,
        target_ref,
        fps=fps,
        joint_vel_mode=joint_vel_mode,
        body_pos_mode=body_pos_mode,
    )
    target_eval = decode_future_model_space(
        target_model,
        fps=fps,
        joint_vel_mode=joint_vel_mode,
        body_pos_mode=body_pos_mode,
        prev_joint_pos=cond_ref[-1, :29],
    )
    cond_eval, _ = apply_model_space_transforms(
        cond_ref,
        np.empty((0, frame_dim), dtype=np.float32),
        fps=fps,
        joint_vel_mode=joint_vel_mode,
        body_pos_mode="relative",
    )

    cond_norm = ((cond_vel - mean) / std).astype(np.float32)  # [H, 65], model-space normalized
    return cond_norm, cond_eval.astype(np.float32), target_eval.astype(np.float32), cond_ref.astype(np.float32)


def _select_windows(paths: list[Path], cfg: dict[str, Any], count: int, seed: int) -> list[tuple[Path, int, float]]:
    """Select deterministic validation windows spread by motion magnitude."""
    rng = np.random.default_rng(seed)
    h = int(cfg["data"]["history_len"])
    k = int(cfg["data"]["pred_len"])
    candidates: list[tuple[Path, int, float]] = []
    for path in rng.choice(paths, size=min(len(paths), max(2000, count * 80)), replace=False):
        seq = np.load(path, mmap_mode="r")
        min_t = h - 1
        max_t = len(seq) - k - 1
        if max_t < min_t:
            continue
        for _ in range(2):
            t = int(rng.integers(min_t, max_t + 1))
            target_pos = np.asarray(seq[t + 1 : t + 1 + k, :29], dtype=np.float32)
            root_pos = np.asarray(seq[t : t + 1 + k, 62:65], dtype=np.float32)
            joint_motion = float(np.mean(np.abs(target_pos[-1] - target_pos[0])))
            root_motion = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
            candidates.append((Path(path), t, joint_motion + root_motion))
    if len(candidates) <= count:
        return candidates
    candidates.sort(key=lambda item: item[2])
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(i)] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", default="samples/checkpoint_eval")
    parser.add_argument("--stats", default=None, help="Optional stats path for old checkpoints without embedded stats")
    parser.add_argument("--num_eval", type=int, default=64)
    parser.add_argument("--num_visual", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    manifest = args.manifest or data_cfg.get("val_file_list") or data_cfg.get("train_file_list")
    paths = _manifest_paths(manifest)
    mean, std = load_checkpoint_stats_or_file(ckpt, args.stats or data_cfg["stats_path"])
    std = np.maximum(std.astype(np.float32), 1e-6)
    mean = mean.astype(np.float32)

    _, sampler = _load_model(ckpt, device)
    pred_len = int(data_cfg["pred_len"])
    frame_dim = int(data_cfg["frame_dim"])
    steps = int(args.num_inference_steps or cfg["diffusion"].get("num_inference_steps", 20))

    windows = _select_windows(paths, cfg, max(args.num_eval, args.num_visual), args.seed)
    eval_windows = windows[: args.num_eval]
    if args.num_visual > 0 and eval_windows:
        vis_indices = np.linspace(0, len(eval_windows) - 1, min(args.num_visual, len(eval_windows)), dtype=int)
        vis_windows = [eval_windows[int(i)] for i in vis_indices]
    else:
        vis_windows = []

    cond_norms: list[np.ndarray] = []
    cond_refs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cond_eval_refs: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for idx, (path, t, score) in enumerate(eval_windows):
        seq = np.load(path, mmap_mode="r")
        cond_norm, cond_eval, target_eval, cond_ref = _window_to_arrays(seq, t, cfg, mean, std)
        cond_norms.append(cond_norm)
        cond_refs.append(cond_ref)
        cond_eval_refs.append(cond_eval)
        targets.append(target_eval)
        metadata.append({"index": idx, "path": str(path), "t": int(t), "motion_score": float(score)})

    cond_tensor = torch.from_numpy(np.stack(cond_norms)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x_t = torch.randn(cond_tensor.shape[0], pred_len, frame_dim, device=device, generator=generator)
    with torch.no_grad():
        pred_norm = sampler.sample(cond_tensor, pred_len, frame_dim, steps, x_T=x_t, eta=0.0)
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

    pred_batch = np.stack(preds).astype(np.float32)
    target_batch = np.stack(targets).astype(np.float32)
    cond_batch = np.stack(cond_eval_refs).astype(np.float32)
    metrics = reference_quality_metrics(
        pred_batch,
        target=target_batch,
        cond=cond_batch,
        fps=float(data_cfg.get("fps", 50.0)),
    )
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["num_eval"] = int(len(eval_windows))
    metrics["num_inference_steps"] = int(steps)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cond_batch.npy", cond_batch)
    np.save(out / "target_batch.npy", target_batch)
    np.save(out / "pred_batch.npy", pred_batch)
    visual_metadata: list[dict[str, Any]] = []
    difficulty_names = ["easy", "medium", "hard"]
    for i, item in enumerate(vis_windows):
        src_idx = eval_windows.index(item)
        if len(vis_windows) == 9:
            label = f"{difficulty_names[i // 3]}_{i % 3 + 1:02d}"
        else:
            label = f"sample_{i:02d}"
        np.save(out / f"{label}_cond.npy", cond_batch[src_idx])
        np.save(out / f"{label}_target.npy", target_batch[src_idx])
        np.save(out / f"{label}_pred.npy", pred_batch[src_idx])
        visual_metadata.append({"label": label, **metadata[src_idx]})
    (out / "evaluation_summary.json").write_text(
        json.dumps({"metrics": metrics, "windows": metadata, "visual_samples": visual_metadata}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
