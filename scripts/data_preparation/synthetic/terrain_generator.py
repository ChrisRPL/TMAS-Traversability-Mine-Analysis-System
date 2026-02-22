"""Procedural terrain generation for synthetic mine detection dataset.

This module generates realistic terrain using Blender's shader nodes and
displacement modifiers to create various terrain types: desert, grassland,
forest floor, rocky, and mixed environments.
"""

import bpy
import math
import random
from typing import Tuple, Optional


class TerrainGenerator:
    """Generate procedural terrain with realistic materials."""

    def __init__(self, size: float = 20.0, resolution: int = 512):
        """Initialize terrain generator.

        Args:
            size: Terrain size in meters (default 20x20m)
            resolution: Subdivision resolution for detail
        """
        self.size = size
        self.resolution = resolution
        self.terrain_obj = None

    def clear_scene(self):
        """Remove all objects from scene."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def create_base_terrain(self) -> bpy.types.Object:
        """Create base terrain mesh with subdivision.

        Returns:
            Terrain mesh object
        """
        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=self.size, location=(0, 0, 0))
        terrain = bpy.context.object
        terrain.name = "Terrain"

        # Add subdivision for detail
        bpy.ops.object.modifier_add(type="SUBSURF")
        terrain.modifiers["Subdivision"].levels = 6
        terrain.modifiers["Subdivision"].render_levels = 7

        self.terrain_obj = terrain
        return terrain

    def add_displacement(
        self,
        terrain_type: str = "desert",
        strength: float = 0.5,
        scale: float = 5.0,
        detail: float = 2.0
    ):
        """Add displacement modifier with noise texture.

        Args:
            terrain_type: Type of terrain (desert, grassland, rocky, forest)
            strength: Displacement strength in meters
            scale: Noise scale
            detail: Noise detail level
        """
        if self.terrain_obj is None:
            raise ValueError("Create base terrain first")

        # Create displacement texture
        tex = bpy.data.textures.new("TerrainDisplacement", type="CLOUDS")
        tex.noise_scale = scale
        tex.noise_depth = int(detail)

        # Adjust noise based on terrain type
        if terrain_type == "rocky":
            tex.noise_scale = 3.0
            tex.noise_depth = 4
            strength = 1.0
        elif terrain_type == "desert":
            tex.noise_scale = 8.0
            tex.noise_depth = 2
            strength = 0.3
        elif terrain_type == "grassland":
            tex.noise_scale = 6.0
            tex.noise_depth = 2
            strength = 0.2
        elif terrain_type == "forest":
            tex.noise_scale = 4.0
            tex.noise_depth = 3
            strength = 0.4

        # Add displacement modifier
        displace = self.terrain_obj.modifiers.new("Displace", type="DISPLACE")
        displace.texture = tex
        displace.strength = strength
        displace.mid_level = 0.5

    def create_pbr_material(
        self,
        terrain_type: str = "desert",
        base_color: Optional[Tuple[float, float, float]] = None,
        roughness: float = 0.8
    ) -> bpy.types.Material:
        """Create PBR material for terrain.

        Args:
            terrain_type: Type of terrain
            base_color: RGB color (0-1), auto-selected if None
            roughness: Surface roughness

        Returns:
            Created material
        """
        # Auto-select color based on terrain type
        if base_color is None:
            colors = {
                "desert": (0.76, 0.70, 0.50),      # Sandy tan
                "grassland": (0.34, 0.54, 0.17),   # Green grass
                "rocky": (0.40, 0.40, 0.38),       # Gray rock
                "forest": (0.28, 0.24, 0.20),      # Dark brown soil
                "mixed": (0.50, 0.45, 0.35)        # Mixed brown
            }
            base_color = colors.get(terrain_type, (0.5, 0.5, 0.5))

        # Create material
        mat = bpy.data.materials.new(name=f"Terrain_{terrain_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (400, 0)

        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.location = (0, 0)
        bsdf_node.inputs["Base Color"].default_value = (*base_color, 1.0)
        bsdf_node.inputs["Roughness"].default_value = roughness

        # Add texture coordinate and mapping
        texcoord_node = nodes.new(type="ShaderNodeTexCoord")
        texcoord_node.location = (-800, 0)

        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.location = (-600, 0)
        mapping_node.inputs["Scale"].default_value = (5.0, 5.0, 5.0)

        # Add noise texture for color variation
        noise_node = nodes.new(type="ShaderNodeTexNoise")
        noise_node.location = (-400, 100)
        noise_node.inputs["Scale"].default_value = 15.0
        noise_node.inputs["Detail"].default_value = 3.0

        # Add color ramp for variation
        colorramp_node = nodes.new(type="ShaderNodeValToRGB")
        colorramp_node.location = (-200, 100)

        # Adjust color ramp for terrain type
        colorramp = colorramp_node.color_ramp
        if terrain_type == "desert":
            colorramp.elements[0].color = (0.70, 0.64, 0.45, 1.0)
            colorramp.elements[1].color = (0.82, 0.76, 0.55, 1.0)
        elif terrain_type == "grassland":
            colorramp.elements[0].color = (0.28, 0.48, 0.12, 1.0)
            colorramp.elements[1].color = (0.40, 0.60, 0.22, 1.0)
        elif terrain_type == "rocky":
            colorramp.elements[0].color = (0.35, 0.35, 0.33, 1.0)
            colorramp.elements[1].color = (0.45, 0.45, 0.43, 1.0)
        elif terrain_type == "forest":
            colorramp.elements[0].color = (0.23, 0.19, 0.15, 1.0)
            colorramp.elements[1].color = (0.33, 0.29, 0.25, 1.0)

        # Connect nodes
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], noise_node.inputs["Vector"])
        links.new(noise_node.outputs["Fac"], colorramp_node.inputs["Fac"])
        links.new(colorramp_node.outputs["Color"], bsdf_node.inputs["Base Color"])
        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

        # Add bump mapping for surface detail
        bump_node = nodes.new(type="ShaderNodeBump")
        bump_node.location = (-200, -200)
        bump_node.inputs["Strength"].default_value = 0.3

        noise2_node = nodes.new(type="ShaderNodeTexNoise")
        noise2_node.location = (-400, -200)
        noise2_node.inputs["Scale"].default_value = 50.0
        noise2_node.inputs["Detail"].default_value = 4.0

        links.new(mapping_node.outputs["Vector"], noise2_node.inputs["Vector"])
        links.new(noise2_node.outputs["Fac"], bump_node.inputs["Height"])
        links.new(bump_node.outputs["Normal"], bsdf_node.inputs["Normal"])

        return mat

    def apply_material(self, material: bpy.types.Material):
        """Apply material to terrain.

        Args:
            material: Material to apply
        """
        if self.terrain_obj is None:
            raise ValueError("Create base terrain first")

        # Clear existing materials
        self.terrain_obj.data.materials.clear()

        # Apply new material
        self.terrain_obj.data.materials.append(material)

    def add_scatter_objects(
        self,
        terrain_type: str = "grassland",
        density: float = 100,
        size_range: Tuple[float, float] = (0.05, 0.15)
    ):
        """Add scattered objects (rocks, grass, debris) using particle system.

        Args:
            terrain_type: Type of scatter objects to add
            density: Number of scattered objects
            size_range: Min/max size of objects
        """
        if self.terrain_obj is None:
            raise ValueError("Create base terrain first")

        # Create scatter object based on terrain type
        if terrain_type in ["grassland", "forest"]:
            # Grass blades
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.01,
                radius2=0.001,
                depth=0.1,
                location=(100, 100, 0)  # Off-screen
            )
        elif terrain_type in ["rocky", "desert"]:
            # Small rocks
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=1,
                radius=0.05,
                location=(100, 100, 0)
            )
        else:
            return

        scatter_obj = bpy.context.object
        scatter_obj.name = f"Scatter_{terrain_type}"

        # Add particle system to terrain
        particle_mod = self.terrain_obj.modifiers.new("Scatter", type="PARTICLE_SYSTEM")
        particle_settings = self.terrain_obj.particle_systems[0].settings

        particle_settings.count = int(density)
        particle_settings.emit_from = "FACE"
        particle_settings.use_rotation_instance = True
        particle_settings.particle_size = random.uniform(*size_range)
        particle_settings.size_random = 0.5
        particle_settings.render_type = "OBJECT"
        particle_settings.instance_object = scatter_obj

    def generate(
        self,
        terrain_type: str = "desert",
        add_scatter: bool = True
    ) -> bpy.types.Object:
        """Generate complete terrain.

        Args:
            terrain_type: Type of terrain (desert, grassland, rocky, forest, mixed)
            add_scatter: Add scattered objects

        Returns:
            Terrain object
        """
        # Create base
        self.create_base_terrain()

        # Add displacement
        self.add_displacement(terrain_type=terrain_type)

        # Create and apply material
        material = self.create_pbr_material(terrain_type=terrain_type)
        self.apply_material(material)

        # Add scatter objects
        if add_scatter and terrain_type != "desert":
            self.add_scatter_objects(terrain_type=terrain_type)

        return self.terrain_obj


def create_random_terrain() -> bpy.types.Object:
    """Create random terrain for data augmentation.

    Returns:
        Terrain object
    """
    terrain_types = ["desert", "grassland", "rocky", "forest"]
    terrain_type = random.choice(terrain_types)

    generator = TerrainGenerator(size=random.uniform(15.0, 25.0))
    terrain = generator.generate(terrain_type=terrain_type, add_scatter=True)

    return terrain


def main():
    """Test terrain generation."""
    print("Generating test terrain...")

    generator = TerrainGenerator(size=20.0, resolution=512)
    terrain = generator.generate(terrain_type="desert", add_scatter=False)

    print(f"Terrain generated: {terrain.name}")
    print("Scene ready for rendering")


if __name__ == "__main__":
    main()
