"""Thermal camera simulation for synthetic dataset generation.

This module simulates thermal imaging (LWIR 8-14μm) by:
- Assigning temperature-based emission to materials
- Simulating heat transfer and thermal properties
- Rendering with emission shader (no traditional lighting)
- Applying thermal noise and sensor characteristics
"""

import bpy
import random
from typing import Dict, Tuple, Optional


class ThermalCamera:
    """Simulate thermal camera rendering."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 512),
        sensor_noise: float = 0.02
    ):
        """Initialize thermal camera.

        Args:
            resolution: (width, height) in pixels
            sensor_noise: Thermal noise level (0-1)
        """
        self.resolution = resolution
        self.sensor_noise = sensor_noise

        # Typical thermal camera parameters
        self.temp_range = (253.15, 353.15)  # -20°C to 80°C in Kelvin
        self.wavelength_range = (8e-6, 14e-6)  # LWIR in meters

    def setup_render_settings(self):
        """Configure Blender render settings for thermal output."""
        scene = bpy.context.scene

        # Resolution
        scene.render.resolution_x = self.resolution[0]
        scene.render.resolution_y = self.resolution[1]
        scene.render.resolution_percentage = 100

        # Use Cycles for emission rendering
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True

        # Film settings (for emission)
        scene.render.film_transparent = False
        scene.view_settings.view_transform = "Raw"  # Linear output

        # Disable all lighting
        scene.world.use_nodes = True
        nodes = scene.world.node_tree.nodes
        nodes.clear()

        # Create black background (no ambient light)
        bg_node = nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0, 0, 0, 1)
        bg_node.inputs["Strength"].default_value = 0

        output_node = nodes.new(type="ShaderNodeOutputWorld")
        scene.world.node_tree.links.new(
            bg_node.outputs["Background"],
            output_node.inputs["Surface"]
        )

    def setup_camera(
        self,
        location: Tuple[float, float, float] = (0, -10, 5),
        look_at: Tuple[float, float, float] = (0, 0, 0),
        fov: float = 45.0
    ) -> bpy.types.Object:
        """Setup thermal camera.

        Args:
            location: Camera position
            look_at: Point to look at
            fov: Field of view in degrees

        Returns:
            Camera object
        """
        # Create camera
        bpy.ops.object.camera_add(location=location)
        camera = bpy.context.object
        camera.name = "ThermalCamera"

        # Set camera properties
        camera.data.lens_unit = "FOV"
        camera.data.angle = math.radians(fov)

        # Point at target
        direction = (
            look_at[0] - location[0],
            look_at[1] - location[1],
            look_at[2] - location[2]
        )
        rot_quat = direction_to_rotation(direction)
        camera.rotation_euler = rot_quat.to_euler()

        # Set as active camera
        bpy.context.scene.camera = camera

        return camera


class ThermalMaterial:
    """Create thermal emission materials based on temperature."""

    # Thermal properties of common materials
    MATERIAL_TEMPS = {
        "metal": (288.15, 295.15),      # 15-22°C (cool)
        "plastic": (295.15, 305.15),    # 22-32°C (warmer)
        "soil": (293.15, 303.15),       # 20-30°C
        "vegetation": (291.15, 298.15), # 18-25°C
        "water": (285.15, 295.15),      # 12-22°C
        "air": (288.15, 298.15)         # 15-25°C
    }

    # Emissivity values (0-1)
    EMISSIVITY = {
        "metal": 0.3,       # Low emissivity (reflective)
        "plastic": 0.95,    # High emissivity
        "soil": 0.92,
        "vegetation": 0.96,
        "water": 0.98,
        "air": 1.0
    }

    @staticmethod
    def kelvin_to_rgb(temp_k: float, emissivity: float = 1.0) -> Tuple[float, float, float]:
        """Convert temperature in Kelvin to RGB emission color.

        Args:
            temp_k: Temperature in Kelvin
            emissivity: Material emissivity (0-1)

        Returns:
            RGB color (0-1) for emission shader
        """
        # Normalize temperature to 0-1 range
        # Using typical outdoor temperature range
        min_temp = 253.15  # -20°C
        max_temp = 353.15  # 80°C

        normalized = (temp_k - min_temp) / (max_temp - min_temp)
        normalized = max(0.0, min(1.0, normalized))  # Clamp

        # Apply emissivity
        intensity = normalized * emissivity

        # Thermal cameras show hotter = brighter (grayscale in LWIR)
        return (intensity, intensity, intensity)

    @staticmethod
    def create_thermal_material(
        name: str,
        material_type: str = "soil",
        temperature: Optional[float] = None,
        add_noise: bool = True
    ) -> bpy.types.Material:
        """Create thermal emission material.

        Args:
            name: Material name
            material_type: Type (metal, plastic, soil, vegetation, water)
            temperature: Temperature in Kelvin (random if None)
            add_noise: Add thermal noise variation

        Returns:
            Thermal material
        """
        # Get temperature range for material type
        temp_range = ThermalMaterial.MATERIAL_TEMPS.get(
            material_type,
            (293.15, 303.15)
        )

        # Random temperature if not specified
        if temperature is None:
            temperature = random.uniform(*temp_range)

        # Get emissivity
        emissivity = ThermalMaterial.EMISSIVITY.get(material_type, 0.9)

        # Create material
        mat = bpy.data.materials.new(name=f"Thermal_{name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create emission shader
        emission_node = nodes.new(type="ShaderNodeEmission")
        emission_node.location = (0, 0)

        # Calculate emission color from temperature
        emission_color = ThermalMaterial.kelvin_to_rgb(temperature, emissivity)
        emission_strength = 1.0

        if add_noise:
            # Add noise texture for temperature variation
            noise_node = nodes.new(type="ShaderNodeTexNoise")
            noise_node.location = (-400, 0)
            noise_node.inputs["Scale"].default_value = 100.0
            noise_node.inputs["Detail"].default_value = 2.0

            # Color ramp for variation
            colorramp_node = nodes.new(type="ShaderNodeValToRGB")
            colorramp_node.location = (-200, 0)

            # Small temperature variation (±2°C)
            var = 2.0 / (353.15 - 253.15)  # Normalized variation
            base = emission_color[0]

            colorramp = colorramp_node.color_ramp
            colorramp.elements[0].color = (
                max(0, base - var),
                max(0, base - var),
                max(0, base - var),
                1.0
            )
            colorramp.elements[1].color = (
                min(1, base + var),
                min(1, base + var),
                min(1, base + var),
                1.0
            )

            links.new(noise_node.outputs["Fac"], colorramp_node.inputs["Fac"])
            links.new(colorramp_node.outputs["Color"], emission_node.inputs["Color"])
        else:
            emission_node.inputs["Color"].default_value = (*emission_color, 1.0)

        emission_node.inputs["Strength"].default_value = emission_strength

        # Output
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (200, 0)

        links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

        return mat

    @staticmethod
    def apply_thermal_to_object(
        obj: bpy.types.Object,
        material_type: str = "soil",
        temperature: Optional[float] = None
    ):
        """Apply thermal material to object.

        Args:
            obj: Object to apply material to
            material_type: Material thermal type
            temperature: Temperature in Kelvin
        """
        thermal_mat = ThermalMaterial.create_thermal_material(
            name=obj.name,
            material_type=material_type,
            temperature=temperature
        )

        # Replace all materials with thermal version
        obj.data.materials.clear()
        obj.data.materials.append(thermal_mat)


class ThermalScene:
    """Setup complete thermal scene."""

    def __init__(self):
        """Initialize thermal scene."""
        self.camera_system = ThermalCamera()
        self.objects_thermal_map: Dict[str, str] = {}

    def convert_scene_to_thermal(self):
        """Convert all materials in scene to thermal emission."""
        # Setup render settings
        self.camera_system.setup_render_settings()

        # Process all mesh objects
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue

            # Determine material type from object name or default
            material_type = self._guess_material_type(obj)

            # Apply thermal material
            ThermalMaterial.apply_thermal_to_object(
                obj,
                material_type=material_type
            )

            self.objects_thermal_map[obj.name] = material_type

    def _guess_material_type(self, obj: bpy.types.Object) -> str:
        """Guess material type from object name.

        Args:
            obj: Object to analyze

        Returns:
            Material type string
        """
        name_lower = obj.name.lower()

        if "terrain" in name_lower or "ground" in name_lower:
            return "soil"
        elif "mine" in name_lower or "metal" in name_lower:
            return "metal"
        elif "ied" in name_lower or "plastic" in name_lower:
            return "plastic"
        elif "grass" in name_lower or "vegetation" in name_lower:
            return "vegetation"
        elif "water" in name_lower:
            return "water"
        else:
            return "soil"  # Default

    def setup_thermal_camera(
        self,
        location: Tuple[float, float, float] = (0, -10, 5),
        look_at: Tuple[float, float, float] = (0, 0, 0)
    ) -> bpy.types.Object:
        """Setup thermal camera in scene.

        Args:
            location: Camera position
            look_at: Point to look at

        Returns:
            Camera object
        """
        return self.camera_system.setup_camera(location, look_at)

    def render_thermal(self, output_path: str):
        """Render thermal image.

        Args:
            output_path: Path to save thermal image
        """
        scene = bpy.context.scene
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)


def direction_to_rotation(direction: Tuple[float, float, float]):
    """Convert direction vector to rotation quaternion.

    Args:
        direction: Direction vector (x, y, z)

    Returns:
        Rotation quaternion
    """
    import mathutils
    dir_vec = mathutils.Vector(direction).normalized()
    default = mathutils.Vector((0, 0, -1))
    return default.rotation_difference(dir_vec)


import math


def setup_thermal_rendering():
    """Setup scene for thermal rendering."""
    thermal_scene = ThermalScene()

    # Convert all materials to thermal
    thermal_scene.convert_scene_to_thermal()

    # Setup camera
    thermal_scene.setup_thermal_camera(
        location=(0, -15, 8),
        look_at=(0, 0, 0)
    )

    print("Thermal rendering setup complete")


def main():
    """Test thermal simulation."""
    print("Setting up thermal simulation...")
    setup_thermal_rendering()
    print("Scene ready for thermal rendering")


if __name__ == "__main__":
    main()
