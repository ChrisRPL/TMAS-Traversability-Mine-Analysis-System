"""Complete obstacle detection inference pipeline.

Integrates all obstacle detection components into unified real-time pipeline.
Target: >15 FPS end-to-end performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class ObstacleInferencePipeline:
    """End-to-end obstacle detection and collision avoidance pipeline.

    Orchestrates: Detection → Depth → Tracking → TTC → Alerts
    """

    def __init__(
        self,
        device: str = "cuda",
        fps: float = 20.0
    ):
        """Initialize inference pipeline.

        Args:
            device: Device for inference (cuda/cpu)
            fps: Frame rate for velocity and TTC calculation
        """
        self.device = device
        self.fps = fps

        # Components will be initialized separately
        self.detector = None
        self.depth_estimator = None
        self.tracker = None
        self.trajectory_predictor = None
        self.ttc_calculator = None
        self.sudden_detector = None

    def initialize_detector(
        self,
        model_size: str = "large",
        confidence_threshold: float = 0.5,
        confidence_critical: float = 0.3
    ) -> None:
        """Initialize RF-DETR obstacle detector.

        Args:
            model_size: Model size (small/medium/large)
            confidence_threshold: General confidence threshold
            confidence_critical: Lower threshold for critical zones
        """
        from .obstacle_detector import ObstacleDetector

        self.detector = ObstacleDetector(
            model_size=model_size,
            confidence_threshold=confidence_threshold,
            confidence_critical=confidence_critical
        ).to(self.device)
        self.detector.eval()
