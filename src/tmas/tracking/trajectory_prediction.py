"""Trajectory prediction using Kalman filtering for moving obstacles.

This module implements Kalman filter-based trajectory prediction to forecast
future positions of moving obstacles. Critical for collision avoidance in
autonomous navigation.

Key features:
- Kalman filter for optimal state estimation under Gaussian noise
- Constant velocity motion model
- Position and velocity tracking: [x, y, vx, vy]
- Trajectory extrapolation 1-3 seconds ahead
- Collision zone detection with vehicle path

The Kalman filter provides:
- Smooth trajectory estimates despite noisy detections
- Uncertainty quantification for predictions
- Optimal fusion of motion model and measurements

Use case: Predict where a moving person/vehicle will be in 1-3 seconds
to enable proactive collision avoidance and path planning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Kalman filter state for a tracked object.

    State vector: [x, y, vx, vy]
    - x, y: Position in image coordinates (pixels)
    - vx, vy: Velocity in pixels/second
    """

    x: np.ndarray  # State vector [4]
    P: np.ndarray  # Covariance matrix [4, 4]
    track_id: int  # Associated track ID


class KalmanFilter:
    """Kalman filter for 2D position and velocity tracking.

    Implements the classic Kalman filter for constant velocity motion model.

    State: [x, y, vx, vy]
    Measurement: [x_obs, y_obs]

    Prediction:
        x_k = F @ x_{k-1} + w  (w ~ N(0, Q))

    Update:
        z_k = H @ x_k + v  (v ~ N(0, R))
    """

    def __init__(
        self,
        dt: float = 0.05,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0
    ):
        """Initialize Kalman filter.

        Args:
            dt: Time step in seconds (1/fps)
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        # x_k = x_{k-1} + vx * dt
        # y_k = y_{k-1} + vy * dt
        # vx_k = vx_{k-1}
        # vy_k = vy_{k-1}
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        q = process_noise ** 2
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float32)

        # Measurement noise covariance
        r = measurement_noise ** 2
        self.R = r * np.eye(2, dtype=np.float32)

    def initialize(self, measurement: np.ndarray, track_id: int) -> KalmanState:
        """Initialize Kalman filter state.

        Args:
            measurement: Initial measurement [x, y]
            track_id: Track identifier

        Returns:
            Initialized KalmanState
        """
        # Initialize state with zero velocity
        x = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)

        # Initialize covariance with high uncertainty
        P = np.diag([100, 100, 100, 100]).astype(np.float32)

        return KalmanState(x=x, P=P, track_id=track_id)

    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state.

        Args:
            state: Current Kalman state

        Returns:
            Predicted state
        """
        # Predict state
        x_pred = self.F @ state.x

        # Predict covariance
        P_pred = self.F @ state.P @ self.F.T + self.Q

        return KalmanState(x=x_pred, P=P_pred, track_id=state.track_id)

    def update(self, state: KalmanState, measurement: np.ndarray) -> KalmanState:
        """Update state with measurement.

        Args:
            state: Predicted state
            measurement: Measurement [x, y]

        Returns:
            Updated state
        """
        # Innovation (measurement residual)
        y = measurement - (self.H @ state.x)

        # Innovation covariance
        S = self.H @ state.P @ self.H.T + self.R

        # Kalman gain
        K = state.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        x_upd = state.x + K @ y

        # Update covariance
        P_upd = (np.eye(4) - K @ self.H) @ state.P

        return KalmanState(x=x_upd, P=P_upd, track_id=state.track_id)

    def predict_trajectory(
        self,
        state: KalmanState,
        n_steps: int
    ) -> np.ndarray:
        """Predict future trajectory.

        Args:
            state: Current state
            n_steps: Number of time steps to predict

        Returns:
            Predicted positions [n_steps, 2] (x, y coordinates)
        """
        trajectory = []
        current_state = state

        for _ in range(n_steps):
            current_state = self.predict(current_state)
            position = current_state.x[:2]
            trajectory.append(position)

        return np.array(trajectory)


class TrajectoryPredictor:
    """Trajectory predictor for multiple tracked objects.

    Manages Kalman filters for all active tracks and predicts
    future trajectories for collision avoidance.
    """

    def __init__(
        self,
        fps: float = 20.0,
        prediction_horizon: float = 3.0,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0
    ):
        """Initialize trajectory predictor.

        Args:
            fps: Frame rate in Hz
            prediction_horizon: How far to predict (seconds)
            process_noise: Process noise for Kalman filter
            measurement_noise: Measurement noise for Kalman filter
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.prediction_horizon = prediction_horizon
        self.n_prediction_steps = int(prediction_horizon * fps)

        # Kalman filter
        self.kf = KalmanFilter(
            dt=self.dt,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

        # Track states (track_id -> KalmanState)
        self.states: Dict[int, KalmanState] = {}

    def update(self, tracks: List) -> Dict[int, KalmanState]:
        """Update predictor with new track measurements.

        Args:
            tracks: List of Track objects from ByteTracker

        Returns:
            Dictionary of track_id -> KalmanState
        """
        current_track_ids = set()

        for track in tracks:
            track_id = track.track_id

            # Get bbox center as measurement
            bbox = track.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            measurement = np.array([cx, cy], dtype=np.float32)

            if track_id not in self.states:
                # Initialize new track
                self.states[track_id] = self.kf.initialize(measurement, track_id)
            else:
                # Predict and update existing track
                predicted_state = self.kf.predict(self.states[track_id])
                self.states[track_id] = self.kf.update(predicted_state, measurement)

            current_track_ids.add(track_id)

        # Remove states for tracks that no longer exist
        removed_ids = set(self.states.keys()) - current_track_ids
        for track_id in removed_ids:
            del self.states[track_id]

        return self.states

    def predict_trajectories(
        self,
        track_ids: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """Predict trajectories for specified tracks.

        Args:
            track_ids: Track IDs to predict (None = all tracks)

        Returns:
            Dictionary of track_id -> predicted_trajectory [N, 2]
        """
        if track_ids is None:
            track_ids = list(self.states.keys())

        trajectories = {}

        for track_id in track_ids:
            if track_id in self.states:
                trajectory = self.kf.predict_trajectory(
                    self.states[track_id],
                    self.n_prediction_steps
                )
                trajectories[track_id] = trajectory

        return trajectories

    def get_velocity(self, track_id: int) -> Optional[np.ndarray]:
        """Get velocity estimate for a track.

        Args:
            track_id: Track ID

        Returns:
            Velocity vector [vx, vy] or None if track not found
        """
        if track_id in self.states:
            return self.states[track_id].x[2:4].copy()
        return None

    def check_collision_zone(
        self,
        trajectory: np.ndarray,
        zone_polygon: np.ndarray,
        threshold: float = 0.5
    ) -> bool:
        """Check if predicted trajectory intersects a collision zone.

        Args:
            trajectory: Predicted trajectory [N, 2]
            zone_polygon: Collision zone as polygon [M, 2]
            threshold: Distance threshold for intersection (normalized)

        Returns:
            True if trajectory intersects zone
        """
        # Simple implementation: check if any trajectory point is inside polygon
        # More sophisticated version would use actual polygon intersection

        from matplotlib.path import Path

        # Create path from polygon
        path = Path(zone_polygon)

        # Check each predicted position
        for position in trajectory:
            if path.contains_point(position):
                return True

        return False


def create_trajectory_predictor(
    fps: float = 20.0,
    prediction_horizon: float = 3.0,
    **kwargs
) -> TrajectoryPredictor:
    """Create trajectory predictor.

    Args:
        fps: Frame rate in Hz
        prediction_horizon: Prediction horizon in seconds
        **kwargs: Additional arguments

    Returns:
        TrajectoryPredictor instance

    Example:
        >>> predictor = create_trajectory_predictor(fps=20.0)
        >>> states = predictor.update(tracks)
        >>> trajectories = predictor.predict_trajectories()
    """
    return TrajectoryPredictor(
        fps=fps,
        prediction_horizon=prediction_horizon,
        **kwargs
    )


def main():
    """Test trajectory predictor."""
    print("Testing Kalman filter trajectory predictor...")

    # Create predictor
    predictor = create_trajectory_predictor(
        fps=20.0,
        prediction_horizon=3.0
    )

    # Simulate a moving track
    print("\nSimulating moving object...")

    # Create mock track class
    class MockTrack:
        def __init__(self, track_id, bbox):
            self.track_id = track_id
            self.bbox = bbox

    # Simulate 10 frames of a moving object
    for frame in range(10):
        # Moving diagonally
        x = 100 + frame * 10
        y = 100 + frame * 5

        track = MockTrack(
            track_id=1,
            bbox=np.array([x, y, x + 50, y + 50])
        )

        # Update predictor
        states = predictor.update([track])

        # Get current state
        state = states[1]
        print(f"Frame {frame}: pos=({state.x[0]:.1f}, {state.x[1]:.1f}), "
              f"vel=({state.x[2]:.1f}, {state.x[3]:.1f})")

    # Predict trajectory
    print("\nPredicting 3-second trajectory...")
    trajectories = predictor.predict_trajectories()

    trajectory = trajectories[1]
    print(f"Predicted {len(trajectory)} positions")
    print(f"First 5 positions:")
    for i in range(min(5, len(trajectory))):
        print(f"  t+{i*0.05:.2f}s: ({trajectory[i, 0]:.1f}, {trajectory[i, 1]:.1f})")

    # Get velocity
    velocity = predictor.get_velocity(1)
    print(f"\nEstimated velocity: ({velocity[0]:.2f}, {velocity[1]:.2f}) pixels/s")

    # Test collision zone check
    print("\nTesting collision zone detection...")
    zone = np.array([[150, 120], [200, 120], [200, 170], [150, 170]])
    intersects = predictor.check_collision_zone(trajectory, zone)
    print(f"Trajectory intersects zone: {intersects}")

    print("\nTrajectory predictor test successful!")


if __name__ == "__main__":
    main()
