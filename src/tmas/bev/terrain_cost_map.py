"""Terrain cost map generation for BEV navigation.

Projects terrain segmentation to BEV and assigns traversability costs
based on TMAS SPEC requirements.
"""

import numpy as np
import cv2
from typing import Optional, Dict
import torch

from .bev_transform import BEVTransform, BEVGrid


# Terrain cost assignment based on TMAS SPEC (Lines 213-225)
# Maps RELLIS-3D class indices to traversability costs [0.0-1.0]
TERRAIN_COSTS = {
    0: 0.0,    # void/unknown - treat as traversable
    1: 0.6,    # dirt - moderate difficulty
    2: 0.15,   # grass - low difficulty, good mine visibility
    3: 0.7,    # tree - obstacle, high difficulty
    4: 0.8,    # pole - obstacle
    5: 0.8,    # water - difficult, mines may shift
    6: 0.7,    # sky - invalid, treat as difficult
    7: 0.7,    # vehicle - obstacle
    8: 0.8,    # object - obstacle
    9: 0.8,    # asphalt - paved road (note: lower is better)
    10: 0.0,   # building - obstacle (will be marked separately)
    11: 0.8,   # log - obstacle
    12: 0.9,   # person - obstacle (critical)
    13: 0.3,   # fence - partial obstacle
    14: 0.7,   # bush - dense vegetation
    15: 0.5,   # concrete - hard surface
    16: 0.8,   # barrier - obstacle
    17: 0.8,   # puddle - water hazard
    18: 0.8,   # mud - difficult terrain
    19: 0.8,   # rubble - high IED risk
}

# TMAS-specific cost mapping (from SPEC requirements)
TMAS_TERRAIN_COSTS = {
    "road": 0.0,        # Paved road - preferred
    "gravel": 0.1,      # Gravel road - good visibility
    "low_grass": 0.15,  # Low grass - good mine visibility
    "dirt": 0.2,        # Packed dirt - possible buried AT mines
    "sand": 0.4,        # Sand - easy mine concealment
    "high_grass": 0.5,  # High grass - limited visibility, AP risk
    "water": 0.6,       # Wet terrain - mines may shift
    "brush": 0.7,       # Dense brush - very limited visibility
    "rubble": 0.8,      # Rubble/ruins - high IED/trap risk
}


class TerrainCostMap:
    """Project terrain segmentation to BEV cost map.

    Assigns traversability costs based on terrain type and geometry,
    following TMAS SPEC requirements for safe navigation.
    """

    def __init__(
        self,
        bev_transform: BEVTransform,
        terrain_segmentor: Optional[torch.nn.Module] = None,
        terrain_costs: Optional[Dict[int, float]] = None,
        temporal_alpha: float = 0.7
    ):
        """Initialize terrain cost map generator.

        Args:
            bev_transform: BEV transformation module
            terrain_segmentor: Terrain segmentation model (optional)
            terrain_costs: Custom cost mapping (uses TERRAIN_COSTS if None)
            temporal_alpha: Temporal averaging factor [0-1] (0=no avg, 1=full avg)
        """
        self.bev_transform = bev_transform
        self.segmentor = terrain_segmentor
        self.terrain_costs = terrain_costs if terrain_costs else TERRAIN_COSTS
        self.temporal_alpha = temporal_alpha

        # Initialize BEV grid for cost map
        self.cost_grid = BEVGrid(
            grid_size=bev_transform.grid_size,
            resolution=bev_transform.resolution,
            dtype=np.float32
        )

        # Previous cost map for temporal averaging
        self.prev_cost_map: Optional[np.ndarray] = None

    def assign_terrain_costs(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """Assign base traversability costs from segmentation mask.

        Args:
            segmentation_mask: HxW segmentation mask with class indices

        Returns:
            HxW cost map with values [0.0-1.0]
        """
        cost_map = np.zeros_like(segmentation_mask, dtype=np.float32)

        # Map each terrain class to its cost
        for class_id, cost in self.terrain_costs.items():
            mask = segmentation_mask == class_id
            cost_map[mask] = cost

        return cost_map

    def project_to_bev(
        self,
        image_cost_map: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """Project image-space cost map to BEV grid.

        Args:
            image_cost_map: HxW cost map in image space
            interpolation: OpenCV interpolation method

        Returns:
            Grid-sized BEV cost map
        """
        H, W = image_cost_map.shape
        grid_size = self.bev_transform.grid_size

        # Create coordinate grids for image space
        y_coords, x_coords = np.meshgrid(
            np.arange(H),
            np.arange(W),
            indexing='ij'
        )

        # Flatten coordinates
        image_points = np.stack([
            x_coords.flatten(),
            y_coords.flatten()
        ], axis=-1).astype(np.float32)

        # Transform to BEV
        bev_points, valid_mask = self.bev_transform.image_to_bev(image_points)

        # Convert to grid coordinates
        grid_coords = self.bev_transform.world_to_grid(bev_points[valid_mask])

        # Round to integer indices
        grid_x = np.clip(grid_coords[:, 0].astype(int), 0, grid_size - 1)
        grid_y = np.clip(grid_coords[:, 1].astype(int), 0, grid_size - 1)

        # Get cost values
        costs = image_cost_map.flatten()[valid_mask]

        # Accumulate in BEV grid (average multiple projections per cell)
        bev_cost = np.zeros((grid_size, grid_size), dtype=np.float32)
        count_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for i in range(len(grid_x)):
            bev_cost[grid_y[i], grid_x[i]] += costs[i]
            count_grid[grid_y[i], grid_x[i]] += 1

        # Average where multiple points project to same cell
        valid = count_grid > 0
        bev_cost[valid] /= count_grid[valid]

        # Fill empty cells with max cost (unknown = dangerous)
        bev_cost[~valid] = 0.8

        return bev_cost

    def create_cost_map(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate complete terrain cost map in BEV.

        Args:
            frame: Input RGB image (H, W, 3)
            depth_map: Optional depth map for geometry costs

        Returns:
            BEV cost map (grid_size, grid_size)
        """
        # 1. Run terrain segmentation
        if self.segmentor is not None:
            with torch.no_grad():
                # Convert to tensor
                img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.float() / 255.0

                # Run segmentation
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                    self.segmentor = self.segmentor.cuda()

                seg_output = self.segmentor(img_tensor)
                seg_mask = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            # Dummy segmentation for testing
            seg_mask = np.zeros(frame.shape[:2], dtype=np.int32)

        # 2. Assign base terrain costs
        image_cost_map = self.assign_terrain_costs(seg_mask)

        # 3. Project to BEV
        bev_cost = self.project_to_bev(image_cost_map)

        # 4. Add geometry costs if depth map provided
        if depth_map is not None:
            bev_cost = self.add_geometry_costs(bev_cost, depth_map)

        # 5. Apply temporal averaging
        if self.prev_cost_map is not None and self.temporal_alpha > 0:
            bev_cost = (
                self.temporal_alpha * self.prev_cost_map +
                (1 - self.temporal_alpha) * bev_cost
            )

        # Store for next frame
        self.prev_cost_map = bev_cost.copy()

        # Update internal grid
        self.cost_grid.grid = bev_cost

        return bev_cost

    def get_cost_map(self) -> np.ndarray:
        """Get current cost map.

        Returns:
            BEV cost map (grid_size, grid_size)
        """
        return self.cost_grid.get_grid()

    def reset(self) -> None:
        """Reset cost map and temporal averaging."""
        self.cost_grid.reset()
        self.prev_cost_map = None

    def compute_slope_cost(
        self,
        depth_map: np.ndarray,
        max_slope_deg: float = 30.0
    ) -> np.ndarray:
        """Compute slope-based cost modifier from depth map.

        Args:
            depth_map: HxW depth map in meters
            max_slope_deg: Maximum slope in degrees for scaling

        Returns:
            HxW slope cost map [0.0-0.3] (SPEC: +0.1 per 10° up to 30°)
        """
        # Compute gradients (depth change in x and y directions)
        grad_y, grad_x = np.gradient(depth_map)

        # Compute slope magnitude (in meters per pixel)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Convert to degrees (approximate)
        # Assuming resolution gives pixel size in meters
        pixel_size = self.bev_transform.resolution
        slope_rad = np.arctan(slope_magnitude / pixel_size)
        slope_deg = np.degrees(slope_rad)

        # Clip to max slope
        slope_deg = np.clip(slope_deg, 0, max_slope_deg)

        # Map to cost: 0.1 per 10 degrees, max 0.3 at 30 degrees
        slope_cost = (slope_deg / 10.0) * 0.1
        slope_cost = np.clip(slope_cost, 0, 0.3)

        return slope_cost

    def compute_roughness_cost(
        self,
        depth_map: np.ndarray,
        window_size: int = 5,
        max_roughness: float = 0.5
    ) -> np.ndarray:
        """Compute roughness-based cost modifier from depth variance.

        Args:
            depth_map: HxW depth map in meters
            window_size: Window size for local variance computation
            max_roughness: Maximum roughness value for scaling

        Returns:
            HxW roughness cost map [0.0-0.1] (SPEC: up to +0.1)
        """
        # Compute local variance using convolution
        kernel = np.ones((window_size, window_size), dtype=np.float32)
        kernel /= kernel.sum()

        # Mean of depth
        depth_mean = cv2.filter2D(depth_map, -1, kernel)

        # Mean of depth squared
        depth_sq_mean = cv2.filter2D(depth_map**2, -1, kernel)

        # Variance = E[X²] - E[X]²
        variance = depth_sq_mean - depth_mean**2
        variance = np.maximum(variance, 0)  # Ensure non-negative

        # Standard deviation as roughness measure
        roughness = np.sqrt(variance)

        # Normalize to [0, 1] range
        roughness = np.clip(roughness / max_roughness, 0, 1)

        # Map to cost: up to 0.1
        roughness_cost = roughness * 0.1

        return roughness_cost

    def add_geometry_costs(
        self,
        base_cost_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Add geometry-based cost modifiers to base cost map.

        Args:
            base_cost_map: Base terrain cost map in BEV
            depth_map: Optional depth map in image space

        Returns:
            Cost map with geometry modifiers added [0.0-1.4]
        """
        if depth_map is None:
            return base_cost_map

        # Compute geometry costs in image space
        slope_cost = self.compute_slope_cost(depth_map)
        roughness_cost = self.compute_roughness_cost(depth_map)

        # Total geometry cost in image space
        geometry_cost = slope_cost + roughness_cost

        # Project geometry cost to BEV
        bev_geometry_cost = self.project_to_bev(geometry_cost)

        # Add to base cost (SPEC: Cost_terrain + Cost_geometry)
        total_cost = base_cost_map + bev_geometry_cost

        return total_cost
