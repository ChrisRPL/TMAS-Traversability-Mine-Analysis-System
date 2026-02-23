"""Time-to-Collision (TTC) estimation for collision avoidance.

TTC measures the time until a potential collision occurs between the
vehicle and a detected obstacle. Critical for safety-critical autonomous
navigation and emergency braking decisions.

TTC formula:
    TTC = distance / relative_velocity

Where:
- distance: Current distance to obstacle (from depth estimation)
- relative_velocity: obstacle_velocity - ego_velocity

Safety thresholds (SPEC Section 3.4):
- TTC < 1s + distance < 10m → CRITICAL (emergency brake)
- TTC < 3s + distance < 20m → WARNING (prepare to brake)
- distance < 50m → OBSERVATION (monitor)

Target accuracy: ±0.3s (SPEC requirement)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class TTCCalculator:
    """Time-to-Collision calculator for obstacle avoidance.

    Computes TTC from obstacle distance, velocity, and vehicle ego-motion.
    Generates safety alerts based on configurable thresholds.
    """

    # Alert levels
    ALERT_NONE = 0
    ALERT_OBSERVATION = 1
    ALERT_WARNING = 2
    ALERT_CRITICAL = 3

    # Safety thresholds (from SPEC)
    TTC_CRITICAL = 1.0  # seconds
    TTC_WARNING = 3.0   # seconds
    DIST_CRITICAL = 10.0  # meters
    DIST_WARNING = 20.0   # meters
    DIST_OBSERVATION = 50.0  # meters

    def __init__(
        self,
        fps: float = 20.0,
        min_velocity_threshold: float = 0.5
    ):
        """Initialize TTC calculator.

        Args:
            fps: Frame rate for velocity calculation
            min_velocity_threshold: Minimum velocity to compute TTC (m/s)
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames
        self.min_velocity_threshold = min_velocity_threshold

    def compute_velocity(
        self,
        trajectory: List[np.ndarray],
        timestamps: List[int]
    ) -> np.ndarray:
        """Compute velocity from trajectory.

        Args:
            trajectory: List of bounding boxes [[x1,y1,x2,y2], ...]
            timestamps: List of frame numbers

        Returns:
            Velocity vector [vx, vy] in pixels/second
        """
        if len(trajectory) < 2:
            return np.array([0.0, 0.0])

        # Get last two positions
        bbox1 = trajectory[-2]
        bbox2 = trajectory[-1]

        # Compute center positions
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2

        # Compute displacement
        dx = cx2 - cx1
        dy = cy2 - cy1

        # Compute time delta
        dt = (timestamps[-1] - timestamps[-2]) / self.fps

        if dt < 1e-6:
            return np.array([0.0, 0.0])

        # Velocity in pixels/second
        vx = dx / dt
        vy = dy / dt

        return np.array([vx, vy])

    def compute_ttc(
        self,
        distance: float,
        velocity: float,
        ego_velocity: Optional[float] = None
    ) -> Optional[float]:
        """Compute time-to-collision.

        Args:
            distance: Distance to obstacle in meters
            velocity: Obstacle velocity in m/s (positive = approaching)
            ego_velocity: Vehicle ego velocity in m/s (optional)

        Returns:
            TTC in seconds, or None if not approaching
        """
        # Account for ego-motion
        if ego_velocity is not None:
            relative_velocity = velocity + ego_velocity
        else:
            relative_velocity = velocity

        # Check if approaching
        if abs(relative_velocity) < self.min_velocity_threshold:
            # Stationary or moving away
            return None

        if relative_velocity < 0:
            # Moving away
            return None

        # Compute TTC
        ttc = distance / relative_velocity

        # Clamp to reasonable range
        if ttc < 0:
            return None
        if ttc > 100:  # More than 100 seconds is not relevant
            return None

        return ttc

    def classify_alert_level(
        self,
        distance: float,
        ttc: Optional[float],
        is_critical_class: bool = False
    ) -> int:
        """Classify safety alert level.

        Args:
            distance: Distance to obstacle in meters
            ttc: Time-to-collision in seconds (None if not approaching)
            is_critical_class: Whether obstacle is critical (person/vehicle)

        Returns:
            Alert level (0=none, 1=observation, 2=warning, 3=critical)
        """
        # Out of range
        if distance > self.DIST_OBSERVATION:
            return self.ALERT_NONE

        # Critical alert conditions
        if ttc is not None and ttc < self.TTC_CRITICAL:
            return self.ALERT_CRITICAL

        if distance < self.DIST_CRITICAL and is_critical_class:
            # Critical class very close even if not approaching
            return self.ALERT_CRITICAL

        # Warning alert conditions
        if ttc is not None and ttc < self.TTC_WARNING:
            return self.ALERT_WARNING

        if distance < self.DIST_WARNING:
            return self.ALERT_WARNING

        # Observation zone
        if distance < self.DIST_OBSERVATION:
            return self.ALERT_OBSERVATION

        return self.ALERT_NONE

    def get_action_recommendation(self, alert_level: int) -> str:
        """Get recommended action for alert level.

        Args:
            alert_level: Alert level (0-3)

        Returns:
            Action string
        """
        actions = {
            self.ALERT_NONE: "CONTINUE",
            self.ALERT_OBSERVATION: "MONITOR",
            self.ALERT_WARNING: "PREPARE_TO_BRAKE",
            self.ALERT_CRITICAL: "EMERGENCY_BRAKE"
        }
        return actions.get(alert_level, "CONTINUE")

    def process_tracks(
        self,
        tracks: List,
        distances: np.ndarray,
        ego_velocity: Optional[float] = None,
        critical_classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """Process tracks to compute TTC and alerts.

        Args:
            tracks: List of Track objects from ByteTracker
            distances: Distance to each track in meters [N]
            ego_velocity: Vehicle ego velocity in m/s
            critical_classes: List of critical class labels (persons, vehicles)

        Returns:
            List of processed track dicts with TTC and alerts
        """
        if critical_classes is None:
            critical_classes = [0, 1, 2, 3, 4, 6]  # Default: person, cars, trucks, bus

        processed_tracks = []

        for i, track in enumerate(tracks):
            distance = distances[i]

            # Compute velocity (simplified - assumes constant velocity)
            if len(track.trajectory) >= 2:
                velocity_pixels = self.compute_velocity(
                    track.trajectory,
                    track.timestamps
                )
                # Convert to m/s (requires calibration - placeholder)
                # This is simplified - real implementation needs pixel-to-meter conversion
                velocity_magnitude = float(np.linalg.norm(velocity_pixels))
                velocity_ms = velocity_magnitude * 0.01  # Placeholder conversion
            else:
                velocity_ms = 0.0

            # Compute TTC
            ttc = self.compute_ttc(distance, velocity_ms, ego_velocity)

            # Classify alert level
            is_critical = track.label in critical_classes
            alert_level = self.classify_alert_level(distance, ttc, is_critical)
            action = self.get_action_recommendation(alert_level)

            # Create processed track dict
            processed_track = {
                "track_id": track.track_id,
                "bbox": track.bbox.tolist(),
                "label": track.label,
                "score": track.score,
                "distance": float(distance),
                "velocity": float(velocity_ms),
                "ttc": float(ttc) if ttc is not None else None,
                "alert_level": alert_level,
                "action": action,
                "is_critical_class": is_critical
            }

            processed_tracks.append(processed_track)

        return processed_tracks


def create_ttc_calculator(
    fps: float = 20.0,
    **kwargs
) -> TTCCalculator:
    """Create TTC calculator.

    Args:
        fps: Frame rate for velocity estimation
        **kwargs: Additional arguments

    Returns:
        TTCCalculator instance

    Example:
        >>> ttc_calc = create_ttc_calculator(fps=20.0)
        >>> ttc = ttc_calc.compute_ttc(distance=15.0, velocity=5.0)
        >>> print(f"TTC: {ttc:.2f} seconds")
    """
    return TTCCalculator(fps=fps, **kwargs)


def main():
    """Test TTC calculator."""
    print("Testing TTC calculator...")

    # Create calculator
    ttc_calc = create_ttc_calculator(fps=20.0)

    # Test TTC computation
    print("\nTest 1: Approaching obstacle")
    distance = 15.0  # meters
    velocity = 5.0   # m/s approaching
    ttc = ttc_calc.compute_ttc(distance, velocity)
    print(f"Distance: {distance}m, Velocity: {velocity}m/s")
    print(f"TTC: {ttc:.2f}s")

    alert_level = ttc_calc.classify_alert_level(distance, ttc, is_critical_class=True)
    action = ttc_calc.get_action_recommendation(alert_level)
    print(f"Alert level: {alert_level}, Action: {action}")

    print("\nTest 2: Critical scenario")
    distance = 8.0   # meters
    velocity = 10.0  # m/s fast approaching
    ttc = ttc_calc.compute_ttc(distance, velocity)
    print(f"Distance: {distance}m, Velocity: {velocity}m/s")
    print(f"TTC: {ttc:.2f}s")

    alert_level = ttc_calc.classify_alert_level(distance, ttc, is_critical_class=True)
    action = ttc_calc.get_action_recommendation(alert_level)
    print(f"Alert level: {alert_level}, Action: {action}")

    print("\nTest 3: Moving away")
    distance = 20.0   # meters
    velocity = -3.0   # m/s moving away
    ttc = ttc_calc.compute_ttc(distance, velocity)
    print(f"Distance: {distance}m, Velocity: {velocity}m/s")
    print(f"TTC: {ttc}")

    alert_level = ttc_calc.classify_alert_level(distance, ttc, is_critical_class=False)
    action = ttc_calc.get_action_recommendation(alert_level)
    print(f"Alert level: {alert_level}, Action: {action}")

    # Test alert thresholds
    print("\nAlert thresholds:")
    print(f"  Critical: TTC < {ttc_calc.TTC_CRITICAL}s or distance < {ttc_calc.DIST_CRITICAL}m")
    print(f"  Warning: TTC < {ttc_calc.TTC_WARNING}s or distance < {ttc_calc.DIST_WARNING}m")
    print(f"  Observation: distance < {ttc_calc.DIST_OBSERVATION}m")

    print("\nTTC calculator test successful!")


if __name__ == "__main__":
    main()
