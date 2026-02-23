"""Loss functions for TMAS training.

This module provides loss functions for terrain segmentation,
mine detection, and obstacle detection tasks.
"""

from .detection_loss import (
    DetectionLoss,
    FocalLoss,
    GIoULoss,
    create_detection_loss
)

__all__ = [
    "DetectionLoss",
    "FocalLoss",
    "GIoULoss",
    "create_detection_loss"
]
