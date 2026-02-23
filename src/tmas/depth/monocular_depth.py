"""Monocular depth estimation using ZoeDepth.

ZoeDepth (Zero-shot Depth) is a state-of-the-art monocular depth estimation
model that provides metric depth prediction with strong zero-shot generalization.

Key advantages:
- Metric depth output (not relative) - critical for TTC calculation
- Strong zero-shot performance across indoor/outdoor scenes
- Based on MiDaS v3 with metric depth bins
- ±5% accuracy in metric depth prediction

For TMAS obstacle detection:
- Distance accuracy target: ±0.5m @ 20m (SPEC requirement)
- Range: 0-50m for outdoor navigation
- Integration with RF-DETR obstacle detector for distance annotation
- Safety zone classification: critical (0-10m), warning (10-20m), observation (20-50m)

References:
- Paper: ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth
- GitHub: https://github.com/isl-org/ZoeDepth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MonocularDepthEstimator(nn.Module):
    """Monocular depth estimation using ZoeDepth.

    Estimates metric depth from single RGB images for obstacle distance
    calculation in autonomous navigation.
    """

    # Safety zones for obstacle detection (SPEC Section 3.4)
    ZONE_CRITICAL = (0.0, 10.0)      # Critical: immediate alert
    ZONE_WARNING = (10.0, 20.0)      # Warning: prepare to brake
    ZONE_OBSERVATION = (20.0, 50.0)  # Observation: monitor

    def __init__(
        self,
        model_type: str = "ZoeD_N",
        max_depth: float = 50.0,
        device: str = "cuda"
    ):
        """Initialize monocular depth estimator.

        Args:
            model_type: ZoeDepth model variant (ZoeD_N, ZoeD_K, ZoeD_NK)
            max_depth: Maximum depth range in meters
            device: Device to run inference (cuda/cpu)
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm not installed. Install with: pip install timm")

        self.model_type = model_type
        self.max_depth = max_depth
        self.device = device

        # Load ZoeDepth model using torch hub
        print(f"Loading ZoeDepth model: {model_type}...")
        try:
            self.model = torch.hub.load(
                "isl-org/ZoeDepth",
                model_type,
                pretrained=True,
                trust_repo=True
            )
            self.model = self.model.to(device)
            self.model.eval()
            print("ZoeDepth model loaded successfully")
        except Exception as e:
            print(f"Error loading ZoeDepth: {e}")
            print("Falling back to MiDaS v3 (relative depth)")
            # Fallback to MiDaS if ZoeDepth fails
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self.model = self.model.to(device)
            self.model.eval()
            self.is_metric = False
        else:
            self.is_metric = True

    def forward(
        self,
        images: torch.Tensor,
        return_numpy: bool = False
    ) -> torch.Tensor:
        """Forward pass for depth estimation.

        Args:
            images: Input RGB images [B, 3, H, W] normalized [0, 1]
            return_numpy: Return depth as numpy array

        Returns:
            Depth maps [B, H, W] in meters (if metric) or relative depth
        """
        with torch.no_grad():
            # ZoeDepth expects [0, 1] normalized RGB
            if images.max() > 1.0:
                images = images / 255.0

            # Inference
            if self.is_metric:
                # ZoeDepth outputs metric depth directly
                depth = self.model.infer(images)
            else:
                # MiDaS outputs inverse depth (disparity)
                depth = self.model(images)
                # Convert to approximate depth (not truly metric)
                depth = 1.0 / (depth + 1e-6)

            # Clip to max depth
            depth = torch.clamp(depth, 0.0, self.max_depth)

        if return_numpy:
            return depth.cpu().numpy()
        return depth

    def estimate_distance(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor,
        method: str = "median"
    ) -> torch.Tensor:
        """Estimate distance to detected obstacles.

        Args:
            images: Input RGB images [B, 3, H, W]
            bboxes: Bounding boxes [B, N, 4] in [x1, y1, x2, y2] pixel format
            method: Distance estimation method (median/mean/min)

        Returns:
            Distances [B, N] in meters for each bounding box
        """
        # Get depth map
        depth_map = self.forward(images)  # [B, H, W]

        B, H, W = depth_map.shape
        _, N, _ = bboxes.shape

        distances = torch.zeros((B, N), device=self.device)

        for b in range(B):
            for n in range(N):
                x1, y1, x2, y2 = bboxes[b, n]

                # Convert to integer coordinates
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Clip to image bounds
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W))
                y2 = max(0, min(y2, H))

                # Extract ROI depth
                roi_depth = depth_map[b, y1:y2, x1:x2]

                if roi_depth.numel() == 0:
                    distances[b, n] = self.max_depth
                    continue

                # Compute distance based on method
                if method == "median":
                    distances[b, n] = torch.median(roi_depth)
                elif method == "mean":
                    distances[b, n] = torch.mean(roi_depth)
                elif method == "min":
                    # Minimum (closest point) - conservative for safety
                    distances[b, n] = torch.min(roi_depth)
                else:
                    distances[b, n] = torch.median(roi_depth)

        return distances

    def classify_safety_zones(
        self,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """Classify obstacles into safety zones.

        Args:
            distances: Obstacle distances [N] in meters

        Returns:
            Zone classifications [N]:
                0 = critical (0-10m)
                1 = warning (10-20m)
                2 = observation (20-50m)
                3 = out of range (>50m)
        """
        zones = torch.zeros_like(distances, dtype=torch.long)

        # Critical zone: 0-10m
        zones[distances < self.ZONE_CRITICAL[1]] = 0

        # Warning zone: 10-20m
        mask_warning = (distances >= self.ZONE_WARNING[0]) & (distances < self.ZONE_WARNING[1])
        zones[mask_warning] = 1

        # Observation zone: 20-50m
        mask_obs = (distances >= self.ZONE_OBSERVATION[0]) & (distances < self.ZONE_OBSERVATION[1])
        zones[mask_obs] = 2

        # Out of range: >50m
        zones[distances >= self.max_depth] = 3

        return zones

    def predict(
        self,
        images: torch.Tensor,
        detections: Optional[Dict] = None
    ) -> Dict:
        """Predict depth with optional obstacle distance estimation.

        Args:
            images: Input RGB images [B, 3, H, W]
            detections: Optional detection dict with 'boxes' [B, N, 4]

        Returns:
            Dictionary with:
                - depth_map: [B, H, W] depth in meters
                - distances: [B, N] obstacle distances (if detections provided)
                - zones: [B, N] safety zone classifications
        """
        depth_map = self.forward(images)

        result = {"depth_map": depth_map}

        if detections is not None and "boxes" in detections:
            bboxes = detections["boxes"]
            distances = self.estimate_distance(images, bboxes)
            zones = self.classify_safety_zones(distances.flatten()).reshape(distances.shape)

            result["distances"] = distances
            result["zones"] = zones

        return result


def create_depth_estimator(
    model_type: str = "ZoeD_N",
    max_depth: float = 50.0,
    device: str = "cuda",
    **kwargs
) -> MonocularDepthEstimator:
    """Create monocular depth estimator.

    Args:
        model_type: ZoeDepth model type
        max_depth: Maximum depth range
        device: Device (cuda/cpu)
        **kwargs: Additional arguments

    Returns:
        MonocularDepthEstimator instance

    Example:
        >>> depth_estimator = create_depth_estimator()
        >>> images = torch.randn(2, 3, 480, 640).cuda()
        >>> result = depth_estimator.predict(images)
        >>> print(f"Depth map: {result['depth_map'].shape}")
    """
    return MonocularDepthEstimator(
        model_type=model_type,
        max_depth=max_depth,
        device=device,
        **kwargs
    )


def main():
    """Test monocular depth estimator."""
    print("Testing monocular depth estimator...")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    else:
        device = "cuda"

    # Create depth estimator
    print("\nCreating ZoeDepth estimator...")
    depth_estimator = create_depth_estimator(
        model_type="ZoeD_N",
        max_depth=50.0,
        device=device
    )

    # Test with dummy image
    print("\nTesting depth estimation...")
    test_image = torch.randn(1, 3, 480, 640)
    if device == "cuda":
        test_image = test_image.cuda()

    result = depth_estimator.predict(test_image)

    print(f"Depth map shape: {result['depth_map'].shape}")
    print(f"Depth range: [{result['depth_map'].min():.2f}, {result['depth_map'].max():.2f}] meters")

    # Test with dummy detections
    print("\nTesting obstacle distance estimation...")
    dummy_boxes = torch.tensor([
        [[100, 100, 200, 200],
         [300, 300, 400, 400]]
    ], dtype=torch.float32, device=device)

    detections = {"boxes": dummy_boxes}
    result = depth_estimator.predict(test_image, detections)

    print(f"Obstacle distances: {result['distances']}")
    print(f"Safety zones: {result['zones']}")
    print(f"  0=critical, 1=warning, 2=observation, 3=out of range")

    # Print zone thresholds
    print(f"\nSafety zone thresholds:")
    print(f"  Critical: {depth_estimator.ZONE_CRITICAL} meters")
    print(f"  Warning: {depth_estimator.ZONE_WARNING} meters")
    print(f"  Observation: {depth_estimator.ZONE_OBSERVATION} meters")

    print("\nMonocular depth estimator test successful!")


if __name__ == "__main__":
    main()
