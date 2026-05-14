"""Run smoke tests without requiring pytest."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from test_smoke import (
    test_add_noise_output_shape,
    test_checkpoint_resume_rejects_incompatible_architecture,
    test_checkpoint_resume_preserves_best_val_and_scaler,
    test_conditional_unet_denoiser_output_shape,
    test_csv_export_formatting,
    test_dataset_slicing,
    test_dataset_uses_mmap_and_returns_float32,
    test_ddim_sampler_external_xt_shape_check,
    test_ddim_sampler_output_shape,
    test_ddim_sampler_output_shape_with_unet,
    test_fallback_velocity_uses_output_fps_after_resampling,
    test_processed_manifest_stale_assumed_fps_guard,
    test_reconstruct_joint_vel_single_frame,
    test_reconstruct_joint_vel_uses_fps,
    test_transformer_denoiser_backward_compat_output_shape,
)


def main() -> None:
    with TemporaryDirectory() as tmp:
        test_dataset_slicing(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_dataset_uses_mmap_and_returns_float32(Path(tmp))
    test_add_noise_output_shape()
    test_ddim_sampler_output_shape()
    test_ddim_sampler_output_shape_with_unet()
    test_ddim_sampler_external_xt_shape_check()
    test_conditional_unet_denoiser_output_shape()
    test_transformer_denoiser_backward_compat_output_shape()
    test_reconstruct_joint_vel_uses_fps()
    test_reconstruct_joint_vel_single_frame()
    with TemporaryDirectory() as tmp:
        test_fallback_velocity_uses_output_fps_after_resampling(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_processed_manifest_stale_assumed_fps_guard(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_csv_export_formatting(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_checkpoint_resume_preserves_best_val_and_scaler(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_checkpoint_resume_rejects_incompatible_architecture(Path(tmp))
    print("smoke tests passed")


if __name__ == "__main__":
    main()
