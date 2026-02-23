"""Object detection models for mine and obstacle detection.

This module provides real-time object detection models optimized for
safety-critical mine detection tasks.
"""

from .rtdetr import RTDETRHead, create_rtdetr_head

__all__ = ["RTDETRHead", "create_rtdetr_head"]
