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


class BEVTransform:
    """Bird's Eye View transformation using Inverse Perspective Mapping.

    Transforms image coordinates to bird's eye view coordinates using
    homography matrix computed from camera calibration.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        grid_size: int = 400,
        resolution: float = 0.05
    ):
        """Initialize BEV transformation.

        Args:
            calibration: Camera calibration parameters
            grid_size: BEV grid size (grid_size × grid_size)
            resolution: Resolution in meters per pixel (default: 5cm/pixel)
        """
        self.calib = calibration
        self.grid_size = grid_size
        self.resolution = resolution

        # Coverage in meters (20m × 20m for 400×400 grid at 5cm/pixel)
        self.coverage = grid_size * resolution

        # Compute homography matrix for ground plane
        self.H = self._compute_homography()

    def _compute_homography(self) -> np.ndarray:
        """Compute homography matrix for ground plane transformation.

        Uses camera intrinsics, extrinsics, and ground plane assumption
        to compute inverse perspective mapping matrix.

        Returns:
            3x3 homography matrix
        """
        # Get rotation and translation
        R = self.calib.get_rotation_matrix()
        t = self.calib.get_translation_vector()

        # Ground plane: Z = 0 in world coordinates
        # Homography for ground plane: H = K * [r1, r2, t]
        # where r1, r2 are first two columns of rotation matrix

        r1 = R[:, 0].reshape(3, 1)
        r2 = R[:, 1].reshape(3, 1)

        # Construct homography matrix
        H = self.calib.K @ np.hstack([r1, r2, t])

        return H

    def image_to_bev(
        self,
        image_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform image coordinates to BEV coordinates.

        Args:
            image_points: Nx2 array of image coordinates (x, y)

        Returns:
            Tuple of (bev_points, valid_mask):
            - bev_points: Nx2 array of BEV coordinates (x, y) in meters
            - valid_mask: N boolean array indicating valid transformations
        """
        if len(image_points) == 0:
            return np.array([]), np.array([], dtype=bool)

        # Convert to homogeneous coordinates
        points_h = np.hstack([
            image_points,
            np.ones((len(image_points), 1))
        ]).T  # 3×N

        # Apply inverse homography
        H_inv = np.linalg.inv(self.H)
        world_points_h = H_inv @ points_h  # 3×N

        # Convert from homogeneous coordinates
        world_points = world_points_h[:2, :] / world_points_h[2, :]  # 2×N
        world_points = world_points.T  # N×2

        # Check valid points (positive depth, within coverage)
        valid_mask = (
            (world_points_h[2, :] > 0) &  # Positive depth
            (world_points[:, 0] >= -self.coverage / 2) &
            (world_points[:, 0] <= self.coverage / 2) &
            (world_points[:, 1] >= 0) &
            (world_points[:, 1] <= self.coverage)
        )

        return world_points, valid_mask

    def bbox_to_bev_footprint(
        self,
        bbox: np.ndarray,
        depth: float
    ) -> Optional[np.ndarray]:
        """Project 2D bounding box to BEV footprint using depth.

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in image coordinates
            depth: Distance to object in meters

        Returns:
            4×2 array of BEV footprint corners, or None if invalid
        """
        x1, y1, x2, y2 = bbox

        # Get bottom center and corners of bbox (ground contact points)
        image_points = np.array([
            [x1, y2],  # Bottom-left
            [x2, y2],  # Bottom-right
            [(x1 + x2) / 2, y2]  # Bottom-center
        ], dtype=np.float32)

        # Transform to BEV
        bev_points, valid = self.image_to_bev(image_points)

        if not valid.all():
            return None

        # Estimate footprint size based on bbox width and depth
        bbox_width_pixels = x2 - x1
        # Approximate width in meters (rough estimate)
        approx_width = bbox_width_pixels * depth / self.calib.fx

        # Create rectangular footprint
        center = bev_points[2]  # Bottom-center point
        half_width = approx_width / 2

        footprint = np.array([
            [center[0] - half_width, center[1]],
            [center[0] + half_width, center[1]],
            [center[0] + half_width, center[1] + 0.5],  # Assume 0.5m depth
            [center[0] - half_width, center[1] + 0.5]
        ])

        return footprint

    def world_to_grid(self, world_points: np.ndarray) -> np.ndarray:
        """Convert world coordinates (meters) to grid coordinates (pixels).

        Args:
            world_points: Nx2 array of world coordinates (x, y) in meters

        Returns:
            Nx2 array of grid coordinates (col, row)
        """
        # Origin at center-bottom of grid
        # X: [-coverage/2, coverage/2] → [0, grid_size]
        # Y: [0, coverage] → [grid_size, 0] (inverted for image coordinates)

        grid_x = (world_points[:, 0] + self.coverage / 2) / self.resolution
        grid_y = self.grid_size - (world_points[:, 1] / self.resolution)

        grid_coords = np.stack([grid_x, grid_y], axis=-1)

        return grid_coords

    def grid_to_world(self, grid_points: np.ndarray) -> np.ndarray:
        """Convert grid coordinates (pixels) to world coordinates (meters).

        Args:
            grid_points: Nx2 array of grid coordinates (col, row)

        Returns:
            Nx2 array of world coordinates (x, y) in meters
        """
        world_x = (grid_points[:, 0] * self.resolution) - self.coverage / 2
        world_y = (self.grid_size - grid_points[:, 1]) * self.resolution

        world_coords = np.stack([world_x, world_y], axis=-1)

        return world_coords


class BEVGrid:
    """Bird's Eye View grid management and visualization.

    Manages 400×400 grid at 5cm/pixel resolution (20m × 20m coverage)
    as specified in TMAS spec.
    """

    def __init__(
        self,
        grid_size: int = 400,
        resolution: float = 0.05,
        dtype: np.dtype = np.float32
    ):
        """Initialize BEV grid.

        Args:
            grid_size: Grid dimensions (grid_size × grid_size)
            resolution: Resolution in meters per pixel (5cm/pixel)
            dtype: Data type for grid values
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.dtype = dtype

        # Coverage in meters (20m × 20m for default)
        self.coverage = grid_size * resolution

        # Create empty grid
        self.grid = self._create_empty_grid()

    def _create_empty_grid(self) -> np.ndarray:
        """Create empty grid with zeros.

        Returns:
            grid_size × grid_size array of zeros
        """
        return np.zeros((self.grid_size, self.grid_size), dtype=self.dtype)

    def reset(self) -> None:
        """Reset grid to zeros."""
        self.grid.fill(0)

    def get_grid(self) -> np.ndarray:
        """Get current grid.

        Returns:
            grid_size × grid_size array
        """
        return self.grid

    def set_value(self, row: int, col: int, value: float) -> None:
        """Set value at grid cell.

        Args:
            row: Row index
            col: Column index
            value: Value to set
        """
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.grid[row, col] = value

    def get_value(self, row: int, col: int) -> float:
        """Get value at grid cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            Value at cell, or 0 if out of bounds
        """
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return float(self.grid[row, col])
        return 0.0

    def fill_polygon(
        self,
        polygon: np.ndarray,
        value: float,
        transform: Optional[BEVTransform] = None
    ) -> None:
        """Fill polygon region with value.

        Args:
            polygon: Nx2 array of polygon vertices in world coordinates
            value: Value to fill
            transform: BEVTransform for coordinate conversion
        """
        if transform is not None:
            # Convert world coordinates to grid coordinates
            grid_coords = transform.world_to_grid(polygon)
        else:
            grid_coords = polygon

        # Round to integer grid coordinates
        grid_coords = np.round(grid_coords).astype(int)

        # Use scanline fill algorithm (simplified)
        # For now, just mark the bounding box
        min_row = max(0, int(grid_coords[:, 1].min()))
        max_row = min(self.grid_size - 1, int(grid_coords[:, 1].max()))
        min_col = max(0, int(grid_coords[:, 0].min()))
        max_col = min(self.grid_size - 1, int(grid_coords[:, 0].max()))

        self.grid[min_row:max_row + 1, min_col:max_col + 1] = value

    def fill_circle(
        self,
        center_world: Tuple[float, float],
        radius: float,
        value: float,
        transform: BEVTransform
    ) -> None:
        """Fill circular region with value.

        Args:
            center_world: Center coordinates (x, y) in meters
            radius: Radius in meters
            value: Value to fill
            transform: BEVTransform for coordinate conversion
        """
        # Convert center to grid coordinates
        center_array = np.array([center_world])
        center_grid = transform.world_to_grid(center_array)[0]
        center_col, center_row = int(center_grid[0]), int(center_grid[1])

        # Radius in pixels
        radius_pixels = int(radius / self.resolution)

        # Fill circle using distance check
        for r in range(max(0, center_row - radius_pixels),
                      min(self.grid_size, center_row + radius_pixels + 1)):
            for c in range(max(0, center_col - radius_pixels),
                          min(self.grid_size, center_col + radius_pixels + 1)):
                dist = np.sqrt((r - center_row)**2 + (c - center_col)**2)
                if dist <= radius_pixels:
                    self.grid[r, c] = value

    def visualize(self, colormap: str = 'viridis') -> np.ndarray:
        """Create visualization of grid.

        Args:
            colormap: Matplotlib colormap name

        Returns:
            RGB image (H, W, 3) for visualization
        """
        # Normalize grid to [0, 1] for visualization
        grid_norm = self.grid.copy()

        # Handle infinite values
        finite_mask = np.isfinite(grid_norm)
        if finite_mask.any():
            grid_norm[~finite_mask] = grid_norm[finite_mask].max()

        # Normalize to [0, 1]
        grid_min = grid_norm.min()
        grid_max = grid_norm.max()
        if grid_max > grid_min:
            grid_norm = (grid_norm - grid_min) / (grid_max - grid_min)
        else:
            grid_norm = np.zeros_like(grid_norm)

        # Convert to RGB (simple grayscale for now)
        # In production, use matplotlib colormap
        rgb = np.stack([grid_norm] * 3, axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        return rgb
