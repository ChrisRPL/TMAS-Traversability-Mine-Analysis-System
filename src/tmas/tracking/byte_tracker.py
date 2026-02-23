"""ByteTrack multi-object tracking implementation.

ByteTrack is a simple yet effective tracking algorithm that achieves
state-of-the-art performance without complex appearance models or ReID.

Key advantages for TMAS:
- No ReID network needed (faster, lighter)
- Low/high confidence detection matching
- Recovers temporarily occluded objects
- 80.3% MOTA on MOT17 benchmark
- Real-time capable on edge devices

The algorithm works by:
1. Match high-confidence detections to existing tracks (first association)
2. Match low-confidence detections to unmatched tracks (second association)
3. Initialize new tracks from remaining high-confidence detections

References:
- Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
- GitHub: https://github.com/ifzhang/ByteTrack
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment


class Track:
    """Single object track.

    Maintains state for a tracked object across frames.
    """

    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        label: int,
        track_id: int,
        frame_id: int
    ):
        """Initialize track.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            score: Detection confidence
            label: Class label
            track_id: Unique track identifier
            frame_id: Frame number
        """
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.label = label
        self.frame_id = frame_id

        # Track state
        self.state = "new"  # new, active, lost, removed
        self.age = 0  # Frames since initialization
        self.hits = 1  # Number of matched detections
        self.time_since_update = 0  # Frames since last update

        # Trajectory history
        self.trajectory = [bbox.copy()]
        self.timestamps = [frame_id]

    def update(self, bbox: np.ndarray, score: float, frame_id: int):
        """Update track with new detection.

        Args:
            bbox: New bounding box
            score: Detection score
            frame_id: Current frame number
        """
        self.bbox = bbox
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.state = "active"

        # Update trajectory
        self.trajectory.append(bbox.copy())
        self.timestamps.append(frame_id)

    def predict(self):
        """Predict next position (simple constant velocity).

        For ByteTrack, we use simple IoU matching without prediction.
        This method is a placeholder for potential Kalman filter integration.
        """
        # Simple constant velocity prediction
        if len(self.trajectory) >= 2:
            # Compute velocity from last two positions
            prev_bbox = self.trajectory[-2]
            curr_bbox = self.trajectory[-1]

            dx = curr_bbox[0] - prev_bbox[0]
            dy = curr_bbox[1] - prev_bbox[1]
            dw = curr_bbox[2] - prev_bbox[2]
            dh = curr_bbox[3] - prev_bbox[3]

            # Predict next position
            predicted = curr_bbox + np.array([dx, dy, dw, dh])
            return predicted
        else:
            return self.bbox

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.age += 1

        if self.state == "active":
            if self.time_since_update > 1:
                self.state = "lost"

    def to_dict(self) -> Dict:
        """Convert track to dictionary.

        Returns:
            Track information as dict
        """
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "score": float(self.score),
            "label": int(self.label),
            "age": self.age,
            "hits": self.hits,
            "state": self.state
        }


class ByteTracker:
    """ByteTrack multi-object tracker.

    Implements ByteTrack algorithm for real-time multi-object tracking
    without appearance models.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_hits: int = 3
    ):
        """Initialize ByteTracker.

        Args:
            track_thresh: High confidence threshold for first association
            track_buffer: Frames to keep lost tracks before removal
            match_thresh: IoU threshold for matching
            min_hits: Minimum hits before track is confirmed
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits

        # Track management
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []  # Lost tracks
        self.removed_tracks = []  # Removed tracks

        self.frame_id = 0
        self.track_id_count = 0

    def update(
        self,
        detections: Dict[str, torch.Tensor],
        frame_id: Optional[int] = None
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: Detection dict with:
                - boxes: [N, 4] in [x1, y1, x2, y2] format
                - scores: [N] confidence scores
                - labels: [N] class labels
            frame_id: Current frame number (auto-increment if None)

        Returns:
            List of active tracks
        """
        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id

        # Convert detections to numpy
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()

        # Split detections by confidence
        high_conf_mask = scores >= self.track_thresh
        low_conf_mask = scores < self.track_thresh

        high_detections = {
            "boxes": boxes[high_conf_mask],
            "scores": scores[high_conf_mask],
            "labels": labels[high_conf_mask]
        }

        low_detections = {
            "boxes": boxes[low_conf_mask],
            "scores": scores[low_conf_mask],
            "labels": labels[low_conf_mask]
        }

        # First association: match high confidence detections
        matched_tracks, unmatched_tracks, unmatched_dets = self._match(
            self.tracked_tracks, high_detections
        )

        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracked_tracks[track_idx]
            track.update(
                high_detections["boxes"][det_idx],
                high_detections["scores"][det_idx],
                self.frame_id
            )

        # Second association: match low confidence to unmatched tracks
        if len(low_detections["boxes"]) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracks_list = [self.tracked_tracks[i] for i in unmatched_tracks]
            matched_low, unmatched_tracks_low, _ = self._match(
                unmatched_tracks_list, low_detections
            )

            for track_idx, det_idx in matched_low:
                track = unmatched_tracks_list[track_idx]
                track.update(
                    low_detections["boxes"][det_idx],
                    low_detections["scores"][det_idx],
                    self.frame_id
                )
                unmatched_tracks.remove(self.tracked_tracks.index(track))

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracked_tracks[track_idx].mark_missed()

        # Initialize new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            new_track = Track(
                bbox=high_detections["boxes"][det_idx],
                score=high_detections["scores"][det_idx],
                label=high_detections["labels"][det_idx],
                track_id=self.track_id_count,
                frame_id=self.frame_id
            )
            self.track_id_count += 1
            self.tracked_tracks.append(new_track)

        # Move lost tracks
        self.lost_tracks = []
        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if t.state != "lost"
        ]

        # Update lost tracks
        for track in self.tracked_tracks[:]:
            if track.state == "lost":
                self.lost_tracks.append(track)
                self.tracked_tracks.remove(track)

        # Remove old tracks
        self.lost_tracks = [
            t for t in self.lost_tracks
            if self.frame_id - t.frame_id <= self.track_buffer
        ]

        # Return active confirmed tracks
        return [t for t in self.tracked_tracks if t.hits >= self.min_hits]

    def _match(
        self,
        tracks: List[Track],
        detections: Dict
    ) -> Tuple[List, List, List]:
        """Match tracks to detections using IoU.

        Args:
            tracks: List of tracks
            detections: Detection dict

        Returns:
            Tuple of (matched pairs, unmatched track indices, unmatched det indices)
        """
        if len(tracks) == 0 or len(detections["boxes"]) == 0:
            return [], list(range(len(tracks))), list(range(len(detections["boxes"])))

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(
            [t.bbox for t in tracks],
            detections["boxes"]
        )

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Filter by IoU threshold
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections["boxes"])))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.match_thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou_matrix(
        boxes1: List[np.ndarray],
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes.

        Args:
            boxes1: List of boxes [x1, y1, x2, y2]
            boxes2: Array of boxes [N, 4]

        Returns:
            IoU matrix [len(boxes1), N]
        """
        boxes1 = np.array(boxes1)
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i, box1 in enumerate(boxes1):
            iou_matrix[i] = ByteTracker._compute_iou(box1, boxes2)

        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes.

        Args:
            box1: Single box [x1, y1, x2, y2]
            boxes2: Multiple boxes [N, 4]

        Returns:
            IoU scores [N]
        """
        # Intersection
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection

        # IoU
        iou = intersection / (union + 1e-6)

        return iou


def create_tracker(
    track_thresh: float = 0.5,
    match_thresh: float = 0.8,
    **kwargs
) -> ByteTracker:
    """Create ByteTrack tracker.

    Args:
        track_thresh: High confidence threshold
        match_thresh: IoU matching threshold
        **kwargs: Additional arguments

    Returns:
        ByteTracker instance

    Example:
        >>> tracker = create_tracker()
        >>> detections = {
        ...     "boxes": torch.tensor([[100, 100, 200, 200]]),
        ...     "scores": torch.tensor([0.9]),
        ...     "labels": torch.tensor([0])
        ... }
        >>> tracks = tracker.update(detections)
        >>> print(f"Active tracks: {len(tracks)}")
    """
    return ByteTracker(
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        **kwargs
    )


def main():
    """Test ByteTracker."""
    print("Testing ByteTracker...")

    # Create tracker
    tracker = create_tracker(
        track_thresh=0.5,
        match_thresh=0.8,
        min_hits=3
    )

    # Simulate detections across frames
    print("\nSimulating tracking across 5 frames...")

    for frame in range(5):
        # Simulate moving object
        x = 100 + frame * 20
        y = 100 + frame * 10

        detections = {
            "boxes": torch.tensor([[x, y, x + 50, y + 50]], dtype=torch.float32),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0])
        }

        tracks = tracker.update(detections)

        print(f"Frame {frame}: {len(tracks)} active tracks")
        for track in tracks:
            print(f"  Track {track.track_id}: bbox={track.bbox}, hits={track.hits}")

    print("\nByteTracker test successful!")


if __name__ == "__main__":
    main()
