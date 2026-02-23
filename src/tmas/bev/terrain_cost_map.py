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

        # 4. Apply temporal averaging
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
