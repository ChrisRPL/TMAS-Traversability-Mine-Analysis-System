"""Lighting and weather randomization for synthetic dataset generation.

This module provides realistic lighting conditions and weather effects:
- Time of day (dawn, day, dusk, night)
- Weather conditions (clear, overcast, rain, fog)
- Sun position and color temperature
- HDRI environment maps
"""

import bpy
import math
import random
from typing import Tuple, Optional


class LightingSystem:
    """Manage scene lighting and environment."""

    def __init__(self):
        """Initialize lighting system."""
        self.sun = None
        self.world = bpy.context.scene.world

    def clear_lights(self):
        """Remove all lights from scene."""
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj, do_unlink=True)

    def create_sun_light(
        self,
        energy: float = 1.0,
        angle: float = 0.00915,  # Sun angular diameter
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> bpy.types.Object:
        """Create sun light.

        Args:
            energy: Light intensity
            angle: Sun angular size in radians
            color: RGB color (0-1)

        Returns:
            Sun light object
        """
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
        sun = bpy.context.object
        sun.name = "Sun"

        sun.data.energy = energy
        sun.data.angle = angle
        sun.data.color = color

        self.sun = sun
        return sun

    def set_sun_position(
        self,
        azimuth: float,
        elevation: float,
        distance: float = 100.0
    ):
        """Set sun position using spherical coordinates.

        Args:
            azimuth: Horizontal angle in degrees (0-360)
            elevation: Vertical angle in degrees (0-90)
            distance: Distance from origin
        """
        if self.sun is None:
            raise ValueError("Create sun light first")

        # Convert to radians
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)

        # Calculate position
        x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = distance * math.sin(elevation_rad)

        self.sun.location = (x, y, z)

        # Point at origin
        direction = (-x, -y, -z)
        rot_quat = direction_to_rotation(direction)
        self.sun.rotation_euler = rot_quat.to_euler()

    def setup_environment(
        self,
        sky_color: Tuple[float, float, float] = (0.5, 0.7, 1.0),
        sky_strength: float = 1.0
    ):
        """Setup basic sky environment.

        Args:
            sky_color: RGB sky color (0-1)
            sky_strength: Environment intensity
        """
        if self.world is None:
            self.world = bpy.data.worlds.new("World")
            bpy.context.scene.world = self.world

        # Enable nodes
        self.world.use_nodes = True
        nodes = self.world.node_tree.nodes
        links = self.world.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Create background node
        bg_node = nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (*sky_color, 1.0)
        bg_node.inputs["Strength"].default_value = sky_strength

        # Create output node
        output_node = nodes.new(type="ShaderNodeOutputWorld")

        # Connect
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    def setup_hdri_environment(self, hdri_path: str, rotation: float = 0.0):
        """Setup HDRI environment map.

        Args:
            hdri_path: Path to HDRI .exr or .hdr file
            rotation: Environment rotation in degrees
        """
        if self.world is None:
            self.world = bpy.data.worlds.new("World")
            bpy.context.scene.world = self.world

        self.world.use_nodes = True
        nodes = self.world.node_tree.nodes
        links = self.world.node_tree.links

        nodes.clear()

        # Environment texture
        env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
        env_tex_node.image = bpy.data.images.load(hdri_path)

        # Mapping for rotation
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = (0, 0, math.radians(rotation))

        # Texture coordinate
        texcoord_node = nodes.new(type="ShaderNodeTexCoord")

        # Background
        bg_node = nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Strength"].default_value = 1.0

        # Output
        output_node = nodes.new(type="ShaderNodeOutputWorld")

        # Connect
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_tex_node.inputs["Vector"])
        links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


class WeatherSystem:
    """Simulate weather conditions."""

    def __init__(self):
        """Initialize weather system."""
        pass

    def add_fog(
        self,
        density: float = 0.1,
        color: Tuple[float, float, float] = (0.7, 0.7, 0.8)
    ):
        """Add volumetric fog to scene.

        Args:
            density: Fog density (0-1)
            color: Fog color RGB (0-1)
        """
        world = bpy.context.scene.world
        if world is None:
            return

        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Find or create background node
        bg_node = None
        for node in nodes:
            if node.type == "BACKGROUND":
                bg_node = node
                break

        if bg_node is None:
            return

        # Add volume scatter
        volume_scatter = nodes.new(type="ShaderNodeVolumeScatter")
        volume_scatter.inputs["Color"].default_value = (*color, 1.0)
        volume_scatter.inputs["Density"].default_value = density

        # Connect to volume output
        output_node = None
        for node in nodes:
            if node.type == "OUTPUT_WORLD":
                output_node = node
                break

        if output_node:
            links.new(volume_scatter.outputs["Volume"], output_node.inputs["Volume"])

    def add_rain(self, intensity: float = 0.5):
        """Add rain effect using particle system.

        Args:
            intensity: Rain intensity (0-1)
        """
        # Create rain particle emitter
        bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 20))
        emitter = bpy.context.object
        emitter.name = "Rain_Emitter"

        # Add particle system
        particle_mod = emitter.modifiers.new("Rain", type="PARTICLE_SYSTEM")
        particle_settings = emitter.particle_systems[0].settings

        particle_settings.count = int(1000 * intensity)
        particle_settings.frame_start = 1
        particle_settings.frame_end = 1
        particle_settings.lifetime = 50
        particle_settings.emit_from = "FACE"
        particle_settings.normal_factor = -1.0
        particle_settings.factor_random = 0.1

        # Particle appearance (thin cylinders)
        particle_settings.render_type = "OBJECT"

        # Create raindrop object
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.002,
            depth=0.05,
            location=(100, 100, 0)  # Off-screen
        )
        raindrop = bpy.context.object
        raindrop.name = "Raindrop"

        particle_settings.instance_object = raindrop

        # Add velocity
        particle_settings.normal_factor = -10.0


class TimeOfDay:
    """Setup lighting for different times of day."""

    @staticmethod
    def dawn(lighting: LightingSystem):
        """Setup dawn lighting (sunrise).

        Args:
            lighting: LightingSystem instance
        """
        # Low sun, warm color
        sun = lighting.create_sun_light(
            energy=0.5,
            color=(1.0, 0.85, 0.7)  # Warm orange
        )
        lighting.set_sun_position(azimuth=90, elevation=5)

        # Cool sky
        lighting.setup_environment(
            sky_color=(0.7, 0.8, 1.0),
            sky_strength=0.3
        )

    @staticmethod
    def day(lighting: LightingSystem):
        """Setup daytime lighting (noon).

        Args:
            lighting: LightingSystem instance
        """
        # High sun, neutral color
        sun = lighting.create_sun_light(
            energy=1.5,
            color=(1.0, 1.0, 0.95)  # Slightly warm white
        )
        lighting.set_sun_position(
            azimuth=random.uniform(0, 360),
            elevation=random.uniform(45, 75)
        )

        # Bright sky
        lighting.setup_environment(
            sky_color=(0.5, 0.7, 1.0),
            sky_strength=1.0
        )

    @staticmethod
    def dusk(lighting: LightingSystem):
        """Setup dusk lighting (sunset).

        Args:
            lighting: LightingSystem instance
        """
        # Low sun, warm color
        sun = lighting.create_sun_light(
            energy=0.6,
            color=(1.0, 0.7, 0.5)  # Deep orange
        )
        lighting.set_sun_position(azimuth=270, elevation=10)

        # Warm sky
        lighting.setup_environment(
            sky_color=(0.9, 0.7, 0.6),
            sky_strength=0.4
        )

    @staticmethod
    def night(lighting: LightingSystem):
        """Setup night lighting (moonlight).

        Args:
            lighting: LightingSystem instance
        """
        # Moon (weak sun)
        sun = lighting.create_sun_light(
            energy=0.1,
            color=(0.7, 0.8, 1.0)  # Cool blue
        )
        lighting.set_sun_position(
            azimuth=random.uniform(0, 360),
            elevation=random.uniform(30, 60)
        )

        # Dark sky
        lighting.setup_environment(
            sky_color=(0.1, 0.1, 0.15),
            sky_strength=0.05
        )


def direction_to_rotation(direction: Tuple[float, float, float]):
    """Convert direction vector to rotation quaternion.

    Args:
        direction: Direction vector (x, y, z)

    Returns:
        Rotation quaternion
    """
    import mathutils
    # Normalize direction
    dir_vec = mathutils.Vector(direction).normalized()
    # Default forward is -Z
    default = mathutils.Vector((0, 0, -1))
    # Calculate rotation
    return default.rotation_difference(dir_vec)


def setup_random_lighting(weather_type: str = "clear", time: str = "day"):
    """Setup random lighting based on weather and time.

    Args:
        weather_type: clear, overcast, rain, fog
        time: dawn, day, dusk, night
    """
    lighting = LightingSystem()
    weather = WeatherSystem()

    # Clear existing lights
    lighting.clear_lights()

    # Setup time of day
    if time == "dawn":
        TimeOfDay.dawn(lighting)
    elif time == "day":
        TimeOfDay.day(lighting)
    elif time == "dusk":
        TimeOfDay.dusk(lighting)
    elif time == "night":
        TimeOfDay.night(lighting)

    # Apply weather
    if weather_type == "overcast":
        # Reduce sun intensity, increase sky light
        if lighting.sun:
            lighting.sun.data.energy *= 0.5
        lighting.setup_environment(sky_color=(0.6, 0.6, 0.65), sky_strength=0.8)

    elif weather_type == "fog":
        weather.add_fog(density=0.05, color=(0.7, 0.7, 0.8))

    elif weather_type == "rain":
        weather.add_fog(density=0.02, color=(0.65, 0.65, 0.7))
        weather.add_rain(intensity=0.5)
        if lighting.sun:
            lighting.sun.data.energy *= 0.3


def main():
    """Test lighting setup."""
    print("Setting up random lighting...")

    times = ["dawn", "day", "dusk", "night"]
    weathers = ["clear", "overcast", "fog", "rain"]

    time = random.choice(times)
    weather = random.choice(weathers)

    print(f"Time: {time}, Weather: {weather}")
    setup_random_lighting(weather_type=weather, time=time)

    print("Lighting setup complete")


if __name__ == "__main__":
    main()
