"""Domain randomization for synthetic data generation.

This module implements domain randomization techniques to improve model
generalization by varying scene parameters across a wide range of values.
Randomizes textures, colors, lighting, mine appearances, and environmental
conditions to reduce the synthetic-to-real domain gap.
"""

import bpy
import random
from typing import Tuple, Optional, List
import colorsys


class DomainRandomizer:
    """Apply domain randomization to Blender scenes."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize domain randomizer.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def randomize_colors(
        self,
        material: bpy.types.Material,
        hue_range: float = 0.1,
        saturation_range: float = 0.2,
        value_range: float = 0.2
    ):
        """Randomize material colors in HSV space.

        Args:
            material: Material to randomize
            hue_range: Maximum hue shift (-0.5 to 0.5)
            saturation_range: Maximum saturation shift
            value_range: Maximum value/brightness shift
        """
        if not material.use_nodes:
            return

        nodes = material.node_tree.nodes

        # Find Principled BSDF or emission nodes
        for node in nodes:
            if node.type == "BSDF_PRINCIPLED":
                base_color = node.inputs["Base Color"].default_value

                # Convert RGB to HSV
                h, s, v = colorsys.rgb_to_hsv(
                    base_color[0], base_color[1], base_color[2]
                )

                # Apply random shifts
                h = (h + random.uniform(-hue_range, hue_range)) % 1.0
                s = max(0, min(1, s + random.uniform(-saturation_range, saturation_range)))
                v = max(0, min(1, v + random.uniform(-value_range, value_range)))

                # Convert back to RGB
                r, g, b = colorsys.hsv_to_rgb(h, s, v)

                node.inputs["Base Color"].default_value = (r, g, b, 1.0)

            elif node.type == "EMISSION":
                # Randomize emission color
                base_color = node.inputs["Color"].default_value

                h, s, v = colorsys.rgb_to_hsv(
                    base_color[0], base_color[1], base_color[2]
                )

                h = (h + random.uniform(-hue_range, hue_range)) % 1.0
                v = max(0, min(1, v + random.uniform(-value_range, value_range)))

                r, g, b = colorsys.hsv_to_rgb(h, s, v)

                node.inputs["Color"].default_value = (r, g, b, 1.0)

    def randomize_roughness(
        self,
        material: bpy.types.Material,
        variation: float = 0.2
    ):
        """Randomize material roughness.

        Args:
            material: Material to randomize
            variation: Maximum roughness variation
        """
        if not material.use_nodes:
            return

        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                base_roughness = node.inputs["Roughness"].default_value
                new_roughness = base_roughness + random.uniform(-variation, variation)
                node.inputs["Roughness"].default_value = max(0, min(1, new_roughness))

    def randomize_metallic(
        self,
        material: bpy.types.Material,
        variation: float = 0.1
    ):
        """Randomize material metallic property.

        Args:
            material: Material to randomize
            variation: Maximum metallic variation
        """
        if not material.use_nodes:
            return

        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                base_metallic = node.inputs["Metallic"].default_value
                new_metallic = base_metallic + random.uniform(-variation, variation)
                node.inputs["Metallic"].default_value = max(0, min(1, new_metallic))

    def randomize_object_scale(
        self,
        obj: bpy.types.Object,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """Randomize object scale.

        Args:
            obj: Object to scale
            scale_range: (min_scale, max_scale) multiplier
        """
        scale_factor = random.uniform(*scale_range)
        obj.scale = (scale_factor, scale_factor, scale_factor)

    def randomize_object_rotation(
        self,
        obj: bpy.types.Object,
        max_tilt: float = 0.2
    ):
        """Randomize object rotation.

        Args:
            obj: Object to rotate
            max_tilt: Maximum tilt in radians (X and Y axes)
        """
        # Random Z rotation (yaw)
        obj.rotation_euler[2] = random.uniform(0, 6.283185)  # 0 to 2π

        # Random tilt
        obj.rotation_euler[0] += random.uniform(-max_tilt, max_tilt)
        obj.rotation_euler[1] += random.uniform(-max_tilt, max_tilt)

    def randomize_camera_position(
        self,
        camera: bpy.types.Object,
        position_variation: Tuple[float, float, float] = (2.0, 2.0, 1.0)
    ):
        """Randomize camera position.

        Args:
            camera: Camera object
            position_variation: (x, y, z) maximum variation
        """
        camera.location[0] += random.uniform(-position_variation[0], position_variation[0])
        camera.location[1] += random.uniform(-position_variation[1], position_variation[1])
        camera.location[2] += random.uniform(-position_variation[2], position_variation[2])

    def randomize_camera_fov(
        self,
        camera: bpy.types.Object,
        fov_range: Tuple[float, float] = (35, 55)
    ):
        """Randomize camera field of view.

        Args:
            camera: Camera object
            fov_range: (min_fov, max_fov) in degrees
        """
        import math
        fov_deg = random.uniform(*fov_range)
        camera.data.angle = math.radians(fov_deg)

    def randomize_sun_strength(
        self,
        sun: bpy.types.Object,
        strength_range: Tuple[float, float] = (0.5, 2.0)
    ):
        """Randomize sun light strength.

        Args:
            sun: Sun light object
            strength_range: (min, max) energy multiplier
        """
        if sun.type == "LIGHT" and sun.data.type == "SUN":
            sun.data.energy = random.uniform(*strength_range)

    def randomize_sun_color(
        self,
        sun: bpy.types.Object,
        temperature_range: Tuple[float, float] = (4000, 7000)
    ):
        """Randomize sun color temperature.

        Args:
            sun: Sun light object
            temperature_range: (min, max) temperature in Kelvin
        """
        if sun.type != "LIGHT" or sun.data.type != "SUN":
            return

        temp = random.uniform(*temperature_range)

        # Simple temperature to RGB conversion
        if temp <= 6600:
            r = 1.0
            g = max(0, min(1, (temp - 2000) / 4600))
            b = max(0, min(1, (temp - 2000) / 6600))
        else:
            r = max(0, min(1, 1.0 - (temp - 6600) / 3400))
            g = 1.0
            b = 1.0

        sun.data.color = (r, g, b)

    def add_random_noise_to_image(
        self,
        scene: bpy.types.Scene,
        noise_amount: float = 0.02
    ):
        """Add noise to rendered image via compositor.

        Args:
            scene: Scene object
            noise_amount: Noise strength (0-1)
        """
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        # Find render layers and composite nodes
        render_layers = None
        composite = None

        for node in nodes:
            if node.type == "R_LAYERS":
                render_layers = node
            elif node.type == "COMPOSITE":
                composite = node

        if render_layers is None or composite is None:
            return

        # Add RGB to BW node
        rgb2bw = nodes.new(type="CompositorNodeRGBToBW")
        rgb2bw.location = (200, 0)

        # Add noise texture
        noise_tex = nodes.new(type="CompositorNodeTexture")
        noise_tex.location = (200, -200)

        # Add mix node
        mix = nodes.new(type="CompositorNodeMixRGB")
        mix.blend_type = "ADD"
        mix.inputs["Fac"].default_value = noise_amount
        mix.location = (400, 0)

        # Connect nodes
        links.new(render_layers.outputs["Image"], mix.inputs[1])
        links.new(noise_tex.outputs["Color"], mix.inputs[2])
        links.new(mix.outputs["Image"], composite.inputs["Image"])

    def randomize_terrain_displacement(
        self,
        terrain: bpy.types.Object,
        strength_variation: float = 0.2
    ):
        """Randomize terrain displacement strength.

        Args:
            terrain: Terrain object
            strength_variation: Maximum variation in displacement
        """
        for modifier in terrain.modifiers:
            if modifier.type == "DISPLACE":
                base_strength = modifier.strength
                modifier.strength = base_strength * random.uniform(
                    1.0 - strength_variation,
                    1.0 + strength_variation
                )

    def randomize_all_materials(
        self,
        objects: Optional[List[bpy.types.Object]] = None
    ):
        """Apply randomization to all materials in scene.

        Args:
            objects: List of objects to randomize (all if None)
        """
        if objects is None:
            objects = bpy.data.objects

        for obj in objects:
            if obj.type != "MESH":
                continue

            for material in obj.data.materials:
                if material is None:
                    continue

                # Randomize color
                self.randomize_colors(
                    material,
                    hue_range=random.uniform(0.05, 0.15),
                    saturation_range=random.uniform(0.1, 0.3),
                    value_range=random.uniform(0.1, 0.3)
                )

                # Randomize roughness
                self.randomize_roughness(
                    material,
                    variation=random.uniform(0.1, 0.3)
                )

                # Randomize metallic (less variation)
                self.randomize_metallic(
                    material,
                    variation=random.uniform(0.05, 0.15)
                )

    def apply_full_randomization(
        self,
        scene: bpy.types.Scene,
        randomize_camera: bool = True,
        randomize_lighting: bool = True,
        randomize_materials: bool = True,
        randomize_objects: bool = True
    ):
        """Apply full domain randomization to scene.

        Args:
            scene: Blender scene
            randomize_camera: Randomize camera parameters
            randomize_lighting: Randomize lighting
            randomize_materials: Randomize materials
            randomize_objects: Randomize object transforms
        """
        print("Applying domain randomization...")

        # Randomize materials
        if randomize_materials:
            self.randomize_all_materials()

        # Randomize camera
        if randomize_camera and scene.camera:
            self.randomize_camera_position(
                scene.camera,
                position_variation=(1.5, 1.5, 0.8)
            )
            self.randomize_camera_fov(
                scene.camera,
                fov_range=(38, 52)
            )

        # Randomize lighting
        if randomize_lighting:
            for obj in bpy.data.objects:
                if obj.type == "LIGHT":
                    if obj.data.type == "SUN":
                        self.randomize_sun_strength(
                            obj,
                            strength_range=(0.6, 1.8)
                        )
                        self.randomize_sun_color(
                            obj,
                            temperature_range=(4500, 6500)
                        )

        # Randomize object transforms
        if randomize_objects:
            for obj in bpy.data.objects:
                if obj.type == "MESH" and "mine" in obj.name.lower():
                    # Small scale variation for mines
                    self.randomize_object_scale(
                        obj,
                        scale_range=(0.9, 1.1)
                    )
                    # Randomize rotation
                    self.randomize_object_rotation(
                        obj,
                        max_tilt=0.15
                    )

        print("Domain randomization complete")


def main():
    """Test domain randomization on current scene."""
    print("Testing domain randomization...")

    randomizer = DomainRandomizer(seed=42)
    scene = bpy.context.scene

    randomizer.apply_full_randomization(
        scene,
        randomize_camera=True,
        randomize_lighting=True,
        randomize_materials=True,
        randomize_objects=True
    )

    print("Randomization applied successfully")


if __name__ == "__main__":
    main()
