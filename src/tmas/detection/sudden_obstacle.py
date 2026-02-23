"""Sudden obstacle detection using frame differencing.

This module detects obstacles that appear suddenly in the field of view
using fast frame differencing techniques. Critical for emergency scenarios
where standard object detection may be too slow.

Target latency: <50ms (SPEC requirement)
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class FrameDifferencer:
    """Basic frame differencing for motion detection.

    Computes absolute difference between consecutive frames to detect motion.
    """

    def __init__(self, threshold: int = 25):
        """Initialize frame differencer.

        Args:
            threshold: Pixel difference threshold for motion detection
        """
        self.threshold = threshold
        self.previous_frame = None

    def compute_difference(
        self,
        current_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute frame difference.

        Args:
            current_frame: Current grayscale frame [H, W]

        Returns:
            Difference mask [H, W] or None if no previous frame
        """
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return None

        # Compute absolute difference
        diff = cv2.absdiff(current_frame, self.previous_frame)

        # Apply threshold
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Update previous frame
        self.previous_frame = current_frame.copy()

        return motion_mask


class MorphologicalFilter:
    """Morphological operations to reduce noise in motion masks."""

    def __init__(self, kernel_size: int = 5):
        """Initialize morphological filter.

        Args:
            kernel_size: Size of morphological kernel
        """
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

    def filter_noise(self, motion_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to reduce noise.

        Args:
            motion_mask: Binary motion mask [H, W]

        Returns:
            Filtered motion mask [H, W]
        """
        # Morphological opening (erosion followed by dilation)
        # Removes small noise
        opened = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel)

        # Morphological closing (dilation followed by erosion)
        # Fills small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)

        return closed
