"""Depth calibration for metric distance accuracy.

This module provides calibration utilities to improve monocular depth
estimation accuracy using camera intrinsics and ground truth data.

Calibration strategies:
1. Scale factor calibration: Learn global scale from known distances
2. Per-zone calibration: Different scales for near/mid/far ranges
3. Camera intrinsics: Use focal length for metric conversion
4. Ground truth alignment: Calibrate against LiDAR or measured distances

Target accuracy (SPEC): ±0.5m @ 20m range
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class DepthCalibrator:
    """Depth calibration for metric accuracy.

    Calibrates monocular depth predictions using camera intrinsics
    and ground truth measurements.
    """

    def __init__(
        self,
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480
    ):
        """Initialize depth calibrator.

        Args:
            focal_length: Camera focal length in pixels
            baseline: Stereo baseline in meters (if available)
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height

        # Calibration parameters (learned from data)
        self.global_scale = 1.0
        self.zone_scales = {
            "near": 1.0,    # 0-10m
            "mid": 1.0,     # 10-20m
            "far": 1.0      # 20-50m
        }

    def load_camera_intrinsics(self, config_path: str):
        """Load camera intrinsics from config file.

        Args:
            config_path: Path to camera calibration JSON
        """
        with open(config_path, 'r') as f:
            intrinsics = json.load(f)

        self.focal_length = intrinsics.get('focal_length')
        self.image_width = intrinsics.get('image_width', self.image_width)
        self.image_height = intrinsics.get('image_height', self.image_height)

        # Load distortion coefficients if available
        self.distortion = intrinsics.get('distortion', None)

    def calibrate_from_ground_truth(
        self,
        predicted_depths: np.ndarray,
        ground_truth_depths: np.ndarray
    ) -> Dict[str, float]:
        """Calibrate scale factor from ground truth data.

        Args:
            predicted_depths: Model predicted depths [N]
            ground_truth_depths: Ground truth depths [N]

        Returns:
            Dictionary with calibration parameters
        """
        # Global scale factor (least squares)
        self.global_scale = np.median(ground_truth_depths / (predicted_depths + 1e-6))

        # Per-zone calibration
        near_mask = ground_truth_depths < 10.0
        mid_mask = (ground_truth_depths >= 10.0) & (ground_truth_depths < 20.0)
        far_mask = ground_truth_depths >= 20.0

        if near_mask.sum() > 0:
            self.zone_scales["near"] = np.median(
                ground_truth_depths[near_mask] / (predicted_depths[near_mask] + 1e-6)
            )

        if mid_mask.sum() > 0:
            self.zone_scales["mid"] = np.median(
                ground_truth_depths[mid_mask] / (predicted_depths[mid_mask] + 1e-6)
            )

        if far_mask.sum() > 0:
            self.zone_scales["far"] = np.median(
                ground_truth_depths[far_mask] / (predicted_depths[far_mask] + 1e-6)
            )

        return {
            "global_scale": float(self.global_scale),
            "zone_scales": self.zone_scales
        }

    def apply_calibration(
        self,
        depth: torch.Tensor,
        use_zone_calibration: bool = True
    ) -> torch.Tensor:
        """Apply calibration to depth predictions.

        Args:
            depth: Predicted depth map [H, W] or [B, H, W]
            use_zone_calibration: Use per-zone scales

        Returns:
            Calibrated depth map
        """
        if use_zone_calibration:
            # Apply zone-specific scales
            calibrated = torch.zeros_like(depth)

            near_mask = depth < 10.0
            mid_mask = (depth >= 10.0) & (depth < 20.0)
            far_mask = depth >= 20.0

            calibrated[near_mask] = depth[near_mask] * self.zone_scales["near"]
            calibrated[mid_mask] = depth[mid_mask] * self.zone_scales["mid"]
            calibrated[far_mask] = depth[far_mask] * self.zone_scales["far"]

            return calibrated
        else:
            # Apply global scale
            return depth * self.global_scale

    def save_calibration(self, save_path: str):
        """Save calibration parameters.

        Args:
            save_path: Path to save calibration JSON
        """
        calibration_data = {
            "focal_length": self.focal_length,
            "baseline": self.baseline,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "global_scale": float(self.global_scale),
            "zone_scales": {k: float(v) for k, v in self.zone_scales.items()}
        }

        with open(save_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

    def load_calibration(self, load_path: str):
        """Load calibration parameters.

        Args:
            load_path: Path to calibration JSON
        """
        with open(load_path, 'r') as f:
            calibration_data = json.load(f)

        self.focal_length = calibration_data.get('focal_length')
        self.baseline = calibration_data.get('baseline')
        self.image_width = calibration_data.get('image_width', self.image_width)
        self.image_height = calibration_data.get('image_height', self.image_height)
        self.global_scale = calibration_data.get('global_scale', 1.0)
        self.zone_scales = calibration_data.get('zone_scales', self.zone_scales)


def create_calibrator(
    focal_length: Optional[float] = None,
    config_path: Optional[str] = None
) -> DepthCalibrator:
    """Create depth calibrator.

    Args:
        focal_length: Camera focal length
        config_path: Path to camera config (overrides focal_length)

    Returns:
        DepthCalibrator instance

    Example:
        >>> calibrator = create_calibrator(focal_length=615.0)
        >>> calibrated_depth = calibrator.apply_calibration(depth_map)
    """
    calibrator = DepthCalibrator(focal_length=focal_length)

    if config_path is not None:
        calibrator.load_camera_intrinsics(config_path)

    return calibrator


def main():
    """Test depth calibrator."""
    print("Testing depth calibrator...")

    # Create calibrator
    calibrator = create_calibrator(focal_length=615.0)

    # Simulate calibration data
    np.random.seed(42)
    predicted = np.random.uniform(5, 40, 100)
    ground_truth = predicted * 1.2 + np.random.normal(0, 0.5, 100)

    print("\nCalibrating from ground truth...")
    calib_params = calibrator.calibrate_from_ground_truth(predicted, ground_truth)

    print(f"Global scale: {calib_params['global_scale']:.3f}")
    print(f"Zone scales:")
    for zone, scale in calib_params['zone_scales'].items():
        print(f"  {zone}: {scale:.3f}")

    # Test calibration application
    test_depth = torch.tensor([[5.0, 15.0, 30.0],
                               [8.0, 18.0, 45.0]])

    print("\nOriginal depth:")
    print(test_depth)

    calibrated = calibrator.apply_calibration(test_depth)
    print("\nCalibrated depth:")
    print(calibrated)

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        save_path = f.name

    calibrator.save_calibration(save_path)
    print(f"\nCalibration saved to: {save_path}")

    new_calibrator = create_calibrator()
    new_calibrator.load_calibration(save_path)
    print("Calibration loaded successfully")

    import os
    os.remove(save_path)

    print("\nDepth calibrator test successful!")


if __name__ == "__main__":
    main()
