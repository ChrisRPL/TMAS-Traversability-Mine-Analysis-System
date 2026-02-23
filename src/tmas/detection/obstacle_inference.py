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

    def initialize_depth_estimator(self, model_type: str = "ZoeD_NK") -> None:
        """Initialize ZoeDepth monocular depth estimator.

        Args:
            model_type: ZoeDepth model variant
        """
        from ..depth.monocular_depth import MonocularDepthEstimator

        self.depth_estimator = MonocularDepthEstimator(
            model_type=model_type
        ).to(self.device)
        self.depth_estimator.eval()

    def initialize_tracker(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30
    ) -> None:
        """Initialize ByteTrack multi-object tracker.

        Args:
            track_thresh: High confidence threshold for tracks
            match_thresh: IOU threshold for matching
            track_buffer: Frames to keep lost tracks
        """
        from ..tracking.byte_tracker import ByteTracker

        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=self.fps
        )

    def initialize_trajectory_predictor(self) -> None:
        """Initialize Kalman filter trajectory predictor."""
        from ..tracking.trajectory_prediction import TrajectoryPredictor

        dt = 1.0 / self.fps
        self.trajectory_predictor = TrajectoryPredictor(dt=dt)

    def initialize_ttc_calculator(
        self,
        critical_ttc: float = 1.0,
        warning_ttc: float = 3.0
    ) -> None:
        """Initialize time-to-collision calculator.

        Args:
            critical_ttc: Critical TTC threshold (seconds)
            warning_ttc: Warning TTC threshold (seconds)
        """
        from .ttc import TTCCalculator

        self.ttc_calculator = TTCCalculator(
            critical_ttc=critical_ttc,
            warning_ttc=warning_ttc
        )

    def initialize_sudden_detector(
        self,
        threshold: int = 25,
        min_area: int = 500
    ) -> None:
        """Initialize sudden obstacle detector.

        Args:
            threshold: Motion detection threshold
            min_area: Minimum area for valid motion region
        """
        from .sudden_obstacle import SuddenObstacleDetector

        self.sudden_detector = SuddenObstacleDetector(
            threshold=threshold,
            min_area=min_area
        )

    @torch.no_grad()
    def process_frame(
        self,
        frame: np.ndarray,
        ego_velocity: Optional[np.ndarray] = None
    ) -> Dict:
        """Process single frame through complete pipeline.

        Args:
            frame: Input image (H, W, 3) BGR
            ego_velocity: Ego vehicle velocity [vx, vy] m/s

        Returns:
            Dictionary with detections, tracks, alerts
        """
        # 1. Obstacle detection
        detections = self.detector(frame)

        # 2. Depth estimation for each detection
        depths = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]
            depth_map = self.depth_estimator(roi)
            mean_depth = float(depth_map.mean())
            depths.append(mean_depth)

        # Add depth to detections
        for det, depth in zip(detections, depths):
            det["depth"] = depth
            det["safety_zone"] = self.depth_estimator.classify_zone(depth)

        # 3. Multi-object tracking
        tracks = self.tracker.update(detections)

        # 4. Trajectory prediction and TTC calculation
        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            depth = track["depth"]

            # Update Kalman filter
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.trajectory_predictor.update(track_id, center_x, center_y)

            # Predict trajectory
            future_trajectory = self.trajectory_predictor.predict_trajectory(
                track_id,
                horizon=3.0
            )
            track["predicted_trajectory"] = future_trajectory

            # Compute TTC
            velocity = self.trajectory_predictor.get_velocity(track_id)
            if velocity is not None:
                ttc_result = self.ttc_calculator.compute_ttc(
                    distance=depth,
                    velocity=np.linalg.norm(velocity),
                    ego_velocity=ego_velocity
                )
                track["ttc"] = ttc_result["ttc"]
                track["collision_risk"] = ttc_result["risk_level"]

        # 5. Sudden obstacle detection
        sudden_obstacles = self.sudden_detector.detect(frame)
        for obs in sudden_obstacles:
            if obs["in_critical_zone"]:
                obs["alert_type"] = "SUDDEN_CRITICAL"

        return {
            "detections": detections,
            "tracks": tracks,
            "sudden_obstacles": sudden_obstacles
        }

    def aggregate_alerts(self, results: Dict) -> List[Dict]:
        """Aggregate all alerts from pipeline results.

        Args:
            results: Output from process_frame

        Returns:
            List of alerts with priority and action recommendations
        """
        alerts = []

        # Critical TTC alerts
        for track in results["tracks"]:
            if track.get("collision_risk") == "CRITICAL":
                alerts.append({
                    "type": "CRITICAL_TTC",
                    "priority": 1,
                    "track_id": track["track_id"],
                    "ttc": track["ttc"],
                    "class": track["class"],
                    "action": "EMERGENCY_BRAKE"
                })

        # Warning TTC alerts
        for track in results["tracks"]:
            if track.get("collision_risk") == "WARNING":
                alerts.append({
                    "type": "WARNING_TTC",
                    "priority": 2,
                    "track_id": track["track_id"],
                    "ttc": track["ttc"],
                    "class": track["class"],
                    "action": "SLOW_DOWN"
                })

        # Sudden obstacle alerts
        for obs in results["sudden_obstacles"]:
            if obs.get("alert_type") == "SUDDEN_CRITICAL":
                alerts.append({
                    "type": "SUDDEN_CRITICAL",
                    "priority": 1,
                    "bbox": obs["bbox"],
                    "action": "EMERGENCY_BRAKE"
                })

        # Sort by priority
        alerts.sort(key=lambda x: x["priority"])

        return alerts
