"""Sample a future chunk from a trained DDIM planner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.motion_chunk_dataset import load_stats
from diffusion.scheduler_wrapper import DiffusionSchedulerWrapper
from models.denoiser import ConditionalDenoiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--cond", type=str, required=True, help=".npy array shaped [H, 65] or [T, 65]")
    parser.add_argument("--output", type=str, default="samples/predicted_chunk.npy")
    parser.add_argument("--x_T", type=str, default=None, help="Optional .npy initial noise shaped [K, 65] or [B, K, 65]")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--ddim_steps", type=int, default=None, help="Deprecated alias for --num_inference_steps")
    parser.add_argument("--denormalize", action="store_true")
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = torch.device("cuda" if cfg["training"].get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")

    model = ConditionalDenoiser(
        frame_dim=int(data_cfg["frame_dim"]),
        history_len=int(data_cfg["history_len"]),
        pred_len=int(data_cfg["pred_len"]),
        model_dim=int(model_cfg["dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        condition_encoder=str(model_cfg["condition_encoder"]),
        architecture=str(model_cfg.get("architecture", "transformer")),
        down_dims=tuple(int(dim) for dim in model_cfg.get("down_dims", [256, 512, 1024])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        n_groups=int(model_cfg.get("n_groups", 8)),
        cond_predict_scale=bool(model_cfg.get("cond_predict_scale", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cond_np = np.load(args.cond).astype(np.float32)
    if cond_np.ndim != 2 or cond_np.shape[1] != int(data_cfg["frame_dim"]):
        raise ValueError(f"condition must have shape [H, 65] or [T, 65], got {cond_np.shape}")
    cond_np = cond_np[-int(data_cfg["history_len"]) :]  # [H, 65]
    mean, std = load_stats(data_cfg["stats_path"])
    cond_np = (cond_np - mean) / std
    cond = torch.from_numpy(cond_np[None]).to(device)  # [1, H, 65]

    x_T = None
    if args.x_T is not None:
        x_np = np.load(args.x_T).astype(np.float32)
        if x_np.ndim == 2:
            x_np = x_np[None]  # [1, K, 65]
        x_T = torch.from_numpy(x_np).to(device)

    diffusion_cfg = cfg["diffusion"]
    sampler = DiffusionSchedulerWrapper(
        model,
        num_train_timesteps=int(diffusion_cfg["num_train_timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
        prediction_type=str(diffusion_cfg["prediction_type"]),
        clip_sample=bool(diffusion_cfg["clip_sample"]),
    ).to(device)
    with torch.no_grad():
        pred = sampler.sample(
            cond=cond,
            pred_len=int(data_cfg["pred_len"]),
            frame_dim=int(data_cfg["frame_dim"]),
            num_inference_steps=args.num_inference_steps or args.ddim_steps or int(diffusion_cfg["num_inference_steps"]),
            x_T=x_T,
            eta=0.0,
        )
    pred_np = pred.squeeze(0).cpu().numpy()  # [K, 65]
    if args.denormalize:
        pred_np = pred_np * std + mean
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, pred_np.astype(np.float32))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
