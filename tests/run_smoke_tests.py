"""Run smoke tests without requiring pytest."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from test_smoke import (
    test_add_noise_output_shape,
    test_auxiliary_losses_use_fps_and_stats,
    test_checkpoint_resume_rejects_incompatible_architecture,
    test_checkpoint_resume_preserves_best_val_and_scaler,
    test_conditional_unet_mean_summary_backward_compat_output_shape,
    test_conditional_unet_denoiser_output_shape,
    test_csv_export_formatting,
    test_dataset_slicing,
    test_dataset_model_space_modes_return_shapes,
    test_dataset_uses_mmap_and_returns_float32,
    test_ddim_sampler_external_xt_shape_check,
    test_ddim_sampler_output_shape,
    test_ddim_sampler_output_shape_with_unet,
    test_evaluate_reports_normalized_component_mse,
    test_fallback_velocity_uses_output_fps_after_resampling,
    test_processed_manifest_stale_assumed_fps_guard,
    test_reconstruct_joint_vel_single_frame,
    test_reconstruct_joint_vel_uses_fps,
    test_rectified_flow_interpolation_and_x0_recovery,
    test_rectified_flow_heun_sampler_output_shape,
    test_rectified_flow_sampler_output_shape,
    test_root_relative_current_frame_pose_is_identity,
    test_dataset_root_relative_shapes_and_values,
    test_model_space_finite_difference_velocity_and_body_delta,
    test_reference_quality_reports_seam_metrics,
    test_transformer_denoiser_backward_compat_output_shape,
    test_transformer_denoiser_raw_history_output_shape,
    test_velocity_consistency_uses_normalized_velocity_space,
    test_continuity_loss_penalizes_velocity_seam,
    test_joint_x0_and_acceleration_losses,
)


def main() -> None:
    with TemporaryDirectory() as tmp:
        test_dataset_slicing(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_dataset_uses_mmap_and_returns_float32(Path(tmp))
    test_root_relative_current_frame_pose_is_identity()
    with TemporaryDirectory() as tmp:
        test_dataset_root_relative_shapes_and_values(Path(tmp))
    test_model_space_finite_difference_velocity_and_body_delta()
    with TemporaryDirectory() as tmp:
        test_dataset_model_space_modes_return_shapes(Path(tmp))
    test_add_noise_output_shape()
    test_rectified_flow_interpolation_and_x0_recovery()
    test_rectified_flow_sampler_output_shape()
    test_rectified_flow_heun_sampler_output_shape()
    test_auxiliary_losses_use_fps_and_stats()
    test_ddim_sampler_output_shape()
    test_ddim_sampler_output_shape_with_unet()
    test_ddim_sampler_external_xt_shape_check()
    test_conditional_unet_denoiser_output_shape()
    test_conditional_unet_mean_summary_backward_compat_output_shape()
    test_transformer_denoiser_backward_compat_output_shape()
    test_transformer_denoiser_raw_history_output_shape()
    test_velocity_consistency_uses_normalized_velocity_space()
    test_continuity_loss_penalizes_velocity_seam()
    test_joint_x0_and_acceleration_losses()
    test_reconstruct_joint_vel_uses_fps()
    test_reconstruct_joint_vel_single_frame()
    test_evaluate_reports_normalized_component_mse()
    test_reference_quality_reports_seam_metrics()
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
