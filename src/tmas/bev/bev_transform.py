"""Bird's Eye View transformation module.

Implements Inverse Perspective Mapping (IPM) for transforming camera view
to Bird's Eye View representation.
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional
from pathlib import Path


class CameraCalibration:
    """Camera calibration parameters for BEV transformation.

    Manages intrinsic and extrinsic camera parameters required for
    inverse perspective mapping.
    """

    def __init__(
        self,
        # Intrinsic parameters
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        # Extrinsic parameters
        height: float,
        pitch: float,
        roll: float = 0.0,
        yaw: float = 0.0,
        # Distortion coefficients (optional)
        k1: float = 0.0,
        k2: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0
    ):
        """Initialize camera calibration.

        Args:
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)
            height: Camera height above ground (meters)
            pitch: Camera pitch angle (radians, positive = looking down)
            roll: Camera roll angle (radians)
            yaw: Camera yaw angle (radians)
            k1, k2: Radial distortion coefficients
            p1, p2: Tangential distortion coefficients
        """
        # Intrinsic parameters
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Extrinsic parameters
        self.height = height
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

        # Distortion coefficients
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

        # Construct intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'CameraCalibration':
        """Load calibration from YAML file.

        Args:
            yaml_path: Path to calibration YAML file

        Returns:
            CameraCalibration instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        intrinsics = config['intrinsics']
        extrinsics = config['extrinsics']
        distortion = config.get('distortion', {})

        return cls(
            # Intrinsics
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
            # Extrinsics
            height=extrinsics['height'],
            pitch=extrinsics['pitch'],
            roll=extrinsics.get('roll', 0.0),
            yaw=extrinsics.get('yaw', 0.0),
            # Distortion
            k1=distortion.get('k1', 0.0),
            k2=distortion.get('k2', 0.0),
            p1=distortion.get('p1', 0.0),
            p2=distortion.get('p2', 0.0)
        )

    def get_rotation_matrix(self) -> np.ndarray:
        """Compute 3D rotation matrix from extrinsic parameters.

        Returns:
            3x3 rotation matrix
        """
        # Rotation around X axis (pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ])

        # Rotation around Y axis (yaw)
        Ry = np.array([
            [np.cos(self.yaw), 0, np.sin(self.yaw)],
            [0, 1, 0],
            [-np.sin(self.yaw), 0, np.cos(self.yaw)]
        ])

        # Rotation around Z axis (roll)
        Rz = np.array([
            [np.cos(self.roll), -np.sin(self.roll), 0],
            [np.sin(self.roll), np.cos(self.roll), 0],
            [0, 0, 1]
        ])

        # Combined rotation: Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        return R

    def get_translation_vector(self) -> np.ndarray:
        """Get translation vector (camera position).

        Returns:
            3x1 translation vector
        """
        return np.array([[0], [0], [self.height]], dtype=np.float32)
