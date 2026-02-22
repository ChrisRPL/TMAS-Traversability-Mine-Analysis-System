"""Mine placement and burial simulation for synthetic dataset generation.

This module handles realistic mine placement with:
- Random positioning within terrain bounds
- Burial depth simulation (0-15cm)
- Realistic orientation and weathering
- Collision detection to prevent overlap
"""

import bpy
import bmesh
import math
import random
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class MinePlacement:
    """Handle mine placement and burial simulation."""

    def __init__(
        self,
        terrain: bpy.types.Object,
        mine_models_dir: str = "data/synthetic/mine_models"
    ):
        """Initialize mine placement system.

        Args:
            terrain: Terrain object to place mines on
            mine_models_dir: Directory containing mine .blend files
        """
        self.terrain = terrain
        self.mine_models_dir = Path(mine_models_dir)
        self.placed_mines: List[Dict] = []
        self.min_distance = 0.5  # Minimum distance between mines in meters

    def load_mine_model(self, mine_path: str) -> bpy.types.Object:
        """Load mine model from .blend file.

        Args:
            mine_path: Path to .blend file

        Returns:
            Loaded mine object
        """
        # Import from .blend file
        with bpy.data.libraries.load(mine_path, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        # Add to scene
        mine_obj = None
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                mine_obj = obj
                break

        if mine_obj is None:
            raise ValueError(f"No object found in {mine_path}")

        return mine_obj

    def get_terrain_height(self, x: float, y: float) -> float:
        """Get terrain height at given (x, y) position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Z height at position
        """
        # Use ray casting to find terrain surface
        scene = bpy.context.scene
        origin = (x, y, 100.0)  # Start above terrain
        direction = (0, 0, -1)

        # Cast ray downward
        result, location, normal, index, obj, matrix = scene.ray_cast(
            bpy.context.view_layer.depsgraph,
            origin,
            direction
        )

        if result and obj == self.terrain:
            return location[2]
        else:
            return 0.0  # Default to ground level

    def check_collision(
        self,
        position: Tuple[float, float],
        radius: float = 0.3
    ) -> bool:
        """Check if position collides with existing mines.

        Args:
            position: (x, y) position to check
            radius: Collision radius

        Returns:
            True if collision detected
        """
        x, y = position

        for mine_info in self.placed_mines:
            mx, my = mine_info["position"][:2]
            distance = math.sqrt((x - mx) ** 2 + (y - my) ** 2)

            if distance < self.min_distance + radius:
                return True

        return False

    def get_random_position(
        self,
        bounds: Tuple[float, float, float, float],
        max_attempts: int = 100
    ) -> Optional[Tuple[float, float, float]]:
        """Get random valid position within terrain bounds.

        Args:
            bounds: (min_x, max_x, min_y, max_y) bounds
            max_attempts: Maximum placement attempts

        Returns:
            (x, y, z) position or None if failed
        """
        min_x, max_x, min_y, max_y = bounds

        for _ in range(max_attempts):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            # Check collision
            if not self.check_collision((x, y)):
                z = self.get_terrain_height(x, y)
                return (x, y, z)

        return None

    def bury_mine(
        self,
        mine: bpy.types.Object,
        burial_depth: float,
        terrain_surface_z: float
    ):
        """Bury mine to specified depth.

        Args:
            mine: Mine object
            burial_depth: Depth in meters (0 = surface, 0.15 = fully buried)
            terrain_surface_z: Z coordinate of terrain surface
        """
        # Move mine down by burial depth
        mine.location[2] = terrain_surface_z - burial_depth

        # Add dirt/soil material overlay if partially buried
        if 0 < burial_depth < 0.15:
            self._add_dirt_overlay(mine, burial_depth / 0.15)

    def _add_dirt_overlay(self, mine: bpy.types.Object, coverage: float):
        """Add dirt material overlay to simulate partial burial.

        Args:
            mine: Mine object
            coverage: Coverage amount (0-1)
        """
        # Create dirt material
        dirt_mat = bpy.data.materials.new(name="Dirt_Overlay")
        dirt_mat.use_nodes = True
        nodes = dirt_mat.node_tree.nodes
        links = dirt_mat.node_tree.links

        nodes.clear()

        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")

        # Dirt color
        bsdf_node.inputs["Base Color"].default_value = (0.28, 0.24, 0.18, 1.0)
        bsdf_node.inputs["Roughness"].default_value = 0.9

        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

        # Apply as second material slot (for weathering effect)
        mine.data.materials.append(dirt_mat)

    def randomize_orientation(
        self,
        mine: bpy.types.Object,
        terrain_normal: Optional[Tuple[float, float, float]] = None
    ):
        """Randomize mine orientation.

        Args:
            mine: Mine object
            terrain_normal: Terrain surface normal (optional)
        """
        # Random rotation around Z axis (yaw)
        mine.rotation_euler[2] = random.uniform(0, 2 * math.pi)

        # Slight tilt for realism
        if terrain_normal is None:
            mine.rotation_euler[0] = random.uniform(-0.1, 0.1)
            mine.rotation_euler[1] = random.uniform(-0.1, 0.1)
        else:
            # Align with terrain normal
            # (Simplified - full implementation would use vector math)
            pass

    def add_weathering(self, mine: bpy.types.Object, weathering_level: float = 0.5):
        """Add weathering effects to mine.

        Args:
            mine: Mine object
            weathering_level: Weathering intensity (0-1)
        """
        # Get mine material
        if len(mine.data.materials) == 0:
            return

        mat = mine.data.materials[0]
        if not mat.use_nodes:
            return

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find Principled BSDF
        bsdf = None
        for node in nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break

        if bsdf is None:
            return

        # Increase roughness for weathering
        base_roughness = bsdf.inputs["Roughness"].default_value
        bsdf.inputs["Roughness"].default_value = min(
            1.0,
            base_roughness + weathering_level * 0.3
        )

        # Add slight color variation (rust, dirt)
        if "Base Color" in bsdf.inputs:
            color = bsdf.inputs["Base Color"].default_value
            # Darken slightly
            factor = 1.0 - (weathering_level * 0.2)
            bsdf.inputs["Base Color"].default_value = (
                color[0] * factor,
                color[1] * factor,
                color[2] * factor,
                1.0
            )

    def place_mine(
        self,
        mine_model_path: str,
        position: Tuple[float, float, float],
        burial_depth: float = 0.0,
        weathering: float = 0.5,
        mine_class: str = "ap_blast"
    ) -> Dict:
        """Place a single mine in the scene.

        Args:
            mine_model_path: Path to mine .blend file
            position: (x, y, z) position
            burial_depth: Burial depth in meters
            weathering: Weathering level (0-1)
            mine_class: Mine classification

        Returns:
            Dictionary with mine placement info
        """
        # Load mine model
        mine = self.load_mine_model(mine_model_path)

        # Set position
        mine.location = position

        # Randomize orientation
        self.randomize_orientation(mine)

        # Apply burial
        if burial_depth > 0:
            self.bury_mine(mine, burial_depth, position[2])

        # Add weathering
        self.add_weathering(mine, weathering)

        # Store placement info
        mine_info = {
            "object": mine,
            "position": position,
            "burial_depth": burial_depth,
            "weathering": weathering,
            "class": mine_class,
            "model_path": mine_model_path
        }

        self.placed_mines.append(mine_info)

        return mine_info

    def place_random_mines(
        self,
        num_mines: int = 5,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        burial_depth_range: Tuple[float, float] = (0.0, 0.15),
        mine_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Place multiple random mines in scene.

        Args:
            num_mines: Number of mines to place
            bounds: (min_x, max_x, min_y, max_y), auto from terrain if None
            burial_depth_range: (min, max) burial depth
            mine_types: List of mine type paths, random if None

        Returns:
            List of placement info dictionaries
        """
        # Auto-detect bounds from terrain
        if bounds is None:
            bbox = self.terrain.bound_box
            min_x = min(v[0] for v in bbox) * 0.8
            max_x = max(v[0] for v in bbox) * 0.8
            min_y = min(v[1] for v in bbox) * 0.8
            max_y = max(v[1] for v in bbox) * 0.8
            bounds = (min_x, max_x, min_y, max_y)

        # Default mine types if not specified
        if mine_types is None:
            mine_types = [
                "ap_blast/pmn2.blend",
                "ap_blast/m14.blend",
                "at_blast/tm62m.blend",
                "uxo/mortar_81mm.blend",
                "ied/simple_ied.blend"
            ]

        placed = []

        for i in range(num_mines):
            # Get random position
            position = self.get_random_position(bounds)
            if position is None:
                print(f"Warning: Could not place mine {i+1}/{num_mines}")
                continue

            # Random mine type
            mine_type = random.choice(mine_types)
            mine_path = str(self.mine_models_dir / mine_type)

            # Random burial depth
            burial_depth = random.uniform(*burial_depth_range)

            # Random weathering
            weathering = random.uniform(0.2, 0.8)

            # Extract class from path
            mine_class = mine_type.split("/")[0]

            # Place mine
            try:
                info = self.place_mine(
                    mine_model_path=mine_path,
                    position=position,
                    burial_depth=burial_depth,
                    weathering=weathering,
                    mine_class=mine_class
                )
                placed.append(info)
            except Exception as e:
                print(f"Error placing mine {i+1}: {e}")

        return placed

    def get_annotations(self) -> List[Dict]:
        """Get COCO format annotations for placed mines.

        Returns:
            List of annotation dictionaries
        """
        annotations = []

        for i, mine_info in enumerate(self.placed_mines):
            mine_obj = mine_info["object"]

            # Get 2D bounding box from camera view
            # (Simplified - full implementation needs camera projection)
            bbox = self._get_2d_bbox(mine_obj)

            annotation = {
                "id": i,
                "category": mine_info["class"],
                "bbox": bbox,  # [x, y, width, height]
                "area": bbox[2] * bbox[3],
                "burial_depth": mine_info["burial_depth"],
                "weathering": mine_info["weathering"],
                "position_3d": mine_info["position"]
            }

            annotations.append(annotation)

        return annotations

    def _get_2d_bbox(self, obj: bpy.types.Object) -> List[float]:
        """Get 2D bounding box in camera view (placeholder).

        Args:
            obj: Object to get bbox for

        Returns:
            [x, y, width, height] in pixels
        """
        # Placeholder - full implementation requires camera projection
        return [0, 0, 100, 100]


def main():
    """Test mine placement."""
    print("Testing mine placement system...")

    # Assuming terrain exists in scene
    terrain = bpy.data.objects.get("Terrain")
    if terrain is None:
        print("Error: No terrain found. Generate terrain first.")
        return

    placer = MinePlacement(terrain)
    mines = placer.place_random_mines(num_mines=5)

    print(f"Placed {len(mines)} mines")
    for i, mine in enumerate(mines):
        print(f"  Mine {i+1}: {mine['class']} at {mine['position']}, "
              f"buried {mine['burial_depth']:.2f}m")


if __name__ == "__main__":
    main()
