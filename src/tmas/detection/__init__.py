"""Object detection models for mine and obstacle detection.

This module provides real-time object detection models optimized for
safety-critical mine detection and obstacle avoidance tasks.
"""

from .rtdetr import RTDETRHead, create_rtdetr_head
from .mine_detector import MineDetectionModel, create_mine_detector
from .obstacle_detector import ObstacleDetector, create_obstacle_detector
from .obstacle_inference import ObstacleInferencePipeline

__all__ = [
    "RTDETRHead",
    "create_rtdetr_head",
    "MineDetectionModel",
    "create_mine_detector",
    "ObstacleDetector",
    "create_obstacle_detector",
    "ObstacleInferencePipeline"
]
