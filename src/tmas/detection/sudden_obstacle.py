"""Sudden obstacle detection using frame differencing.

This module detects obstacles that appear suddenly in the field of view
using fast frame differencing techniques. Critical for emergency scenarios
where standard object detection may be too slow.

Target latency: <50ms (SPEC requirement)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict


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


class MotionRegionDetector:
    """Detect motion regions from filtered motion mask."""

    def __init__(self, min_area: int = 100):
        """Initialize motion region detector.

        Args:
            min_area: Minimum contour area to consider as valid motion
        """
        self.min_area = min_area

    def find_motion_regions(
        self,
        motion_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find motion regions using connected components.

        Args:
            motion_mask: Binary motion mask [H, W]

        Returns:
            Tuple of (bounding_boxes, areas)
                bounding_boxes: [N, 4] in [x, y, w, h] format
                areas: [N] area of each region
        """
        # Find contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        areas = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, w, h])
                areas.append(area)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)

        return np.array(boxes, dtype=np.int32), np.array(areas, dtype=np.float32)


class SuddenObstacleDetector:
    """Complete sudden obstacle detection pipeline.

    Combines frame differencing, morphological filtering, and motion region
    detection to identify obstacles appearing suddenly in critical zones.
    """

    def __init__(
        self,
        threshold: int = 25,
        kernel_size: int = 5,
        min_area: int = 100
    ):
        """Initialize sudden obstacle detector.

        Args:
            threshold: Frame difference threshold
            kernel_size: Morphological filter kernel size
            min_area: Minimum motion region area
        """
        self.differencer = FrameDifferencer(threshold)
        self.filter = MorphologicalFilter(kernel_size)
        self.detector = MotionRegionDetector(min_area)

    def detect(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect sudden obstacles in frame.

        Args:
            frame: RGB frame [H, W, 3]

        Returns:
            Tuple of (motion_boxes, areas)
                motion_boxes: [N, 4] bounding boxes
                areas: [N] motion region areas
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Compute frame difference
        motion_mask = self.differencer.compute_difference(gray)

        if motion_mask is None:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)

        # Filter noise
        filtered_mask = self.filter.filter_noise(motion_mask)

        # Detect motion regions
        boxes, areas = self.detector.find_motion_regions(filtered_mask)

        return boxes, areas

    def check_critical_zone(
        self,
        boxes: np.ndarray,
        image_height: int,
        critical_zone_ratio: float = 0.7
    ) -> List[Dict]:
        """Check if motion boxes are in critical zone.

        Args:
            boxes: Motion bounding boxes [N, 4]
            image_height: Image height in pixels
            critical_zone_ratio: Bottom fraction of image as critical zone

        Returns:
            List of alert dicts for boxes in critical zone
        """
        critical_threshold = image_height * critical_zone_ratio
        alerts = []

        for box in boxes:
            x, y, w, h = box
            box_bottom = y + h

            if box_bottom >= critical_threshold:
                alerts.append({
                    "bbox": box.tolist(),
                    "alert_level": "CRITICAL",
                    "action": "EMERGENCY_BRAKE",
                    "reason": "sudden_obstacle_in_critical_zone"
                })

        return alerts
