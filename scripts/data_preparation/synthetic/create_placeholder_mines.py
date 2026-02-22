"""Create placeholder 3D mine models in Blender.

This script generates simple geometric mine models based on real dimensions
for use in synthetic data generation while proper models are being acquired.
"""

import bpy
import math
from pathlib import Path


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_ap_blast_mine(name: str = "AP_Blast", diameter: float = 0.10, height: float = 0.04):
    """Create anti-personnel blast mine model.

    Args:
        name: Mine name
        diameter: Diameter in meters (default 10cm)
        height: Height in meters (default 4cm)
    """
    clear_scene()

    # Create cylindrical body
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diameter / 2,
        depth=height,
        location=(0, 0, 0)
    )
    mine = bpy.context.object
    mine.name = name

    # Add slight bevel for realism
    bpy.ops.object.modifier_add(type="BEVEL")
    mine.modifiers["Bevel"].width = 0.002
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # Create pressure plate on top
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diameter / 3,
        depth=0.005,
        location=(0, 0, height / 2 + 0.0025)
    )
    pressure_plate = bpy.context.object
    pressure_plate.name = f"{name}_PressurePlate"

    # Join parts
    bpy.ops.object.select_all(action="DESELECT")
    mine.select_set(True)
    pressure_plate.select_set(True)
    bpy.context.view_layer.objects.active = mine
    bpy.ops.object.join()

    # Add material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.25, 0.2, 1.0)  # Dark brown/green
    bsdf.inputs["Roughness"].default_value = 0.6
    mine.data.materials.append(mat)

    return mine


def create_at_blast_mine(name: str = "AT_Blast", diameter: float = 0.30, height: float = 0.12):
    """Create anti-tank blast mine model.

    Args:
        name: Mine name
        diameter: Diameter in meters (default 30cm)
        height: Height in meters (default 12cm)
    """
    clear_scene()

    # Create main cylindrical body
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diameter / 2,
        depth=height,
        location=(0, 0, 0)
    )
    mine = bpy.context.object
    mine.name = name

    # Add slight taper
    bpy.ops.object.modifier_add(type="SIMPLE_DEFORM")
    mine.modifiers["SimpleDeform"].deform_method = "TAPER"
    mine.modifiers["SimpleDeform"].factor = -0.1
    bpy.ops.object.modifier_apply(modifier="SimpleDeform")

    # Add bevel
    bpy.ops.object.modifier_add(type="BEVEL")
    mine.modifiers["Bevel"].width = 0.005
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # Add material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.25, 0.3, 0.2, 1.0)  # Olive green
    bsdf.inputs["Metallic"].default_value = 0.2
    bsdf.inputs["Roughness"].default_value = 0.7
    mine.data.materials.append(mat)

    return mine


def create_mortar_uxo(name: str = "UXO_Mortar", diameter: float = 0.08, length: float = 0.30):
    """Create mortar UXO model.

    Args:
        name: UXO name
        diameter: Diameter in meters (default 8cm)
        length: Length in meters (default 30cm)
    """
    clear_scene()

    # Create main body (cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diameter / 2,
        depth=length * 0.7,
        location=(0, 0, 0),
        rotation=(math.pi / 2, 0, 0)
    )
    body = bpy.context.object
    body.name = f"{name}_Body"

    # Create nose cone
    bpy.ops.mesh.primitive_cone_add(
        radius1=diameter / 2,
        radius2=0.01,
        depth=length * 0.2,
        location=(length * 0.35 + length * 0.1, 0, 0),
        rotation=(math.pi / 2, 0, 0)
    )
    nose = bpy.context.object
    nose.name = f"{name}_Nose"

    # Create tail fins
    for i in range(4):
        angle = i * math.pi / 2
        bpy.ops.mesh.primitive_cube_add(
            size=0.01,
            location=(
                -length * 0.35 - 0.02,
                (diameter / 2 + 0.015) * math.cos(angle),
                (diameter / 2 + 0.015) * math.sin(angle)
            ),
            rotation=(0, 0, angle)
        )
        fin = bpy.context.object
        fin.scale = (length * 0.1, 0.001, diameter / 2)
        fin.name = f"{name}_Fin{i}"

    # Join all parts
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    mortar = bpy.context.object
    mortar.name = name

    # Add material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.4, 0.4, 0.3, 1.0)  # Dull metal
    bsdf.inputs["Metallic"].default_value = 0.8
    bsdf.inputs["Roughness"].default_value = 0.5
    mortar.data.materials.append(mat)

    return mortar


def create_simple_ied(name: str = "IED_Simple", size: float = 0.15):
    """Create simple IED model (container-based).

    Args:
        name: IED name
        size: Container size in meters (default 15cm)
    """
    clear_scene()

    # Create container (box or cylinder)
    bpy.ops.mesh.primitive_cube_add(
        size=size,
        location=(0, 0, 0)
    )
    container = bpy.context.object
    container.name = f"{name}_Container"

    # Add bevel for realism
    bpy.ops.object.modifier_add(type="BEVEL")
    container.modifiers["Bevel"].width = 0.005
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # Add protruding wires
    for i in range(2):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.002,
            depth=size * 0.5,
            location=(size / 2 - 0.02 - i * 0.04, 0, size / 2),
            rotation=(math.pi / 2, 0, 0)
        )
        wire = bpy.context.object
        wire.name = f"{name}_Wire{i}"

    # Join parts
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = container
    bpy.ops.object.join()
    ied = bpy.context.object
    ied.name = name

    # Add material (plastic container)
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.5, 0.4, 0.3, 1.0)  # Tan plastic
    bsdf.inputs["Roughness"].default_value = 0.4
    ied.data.materials.append(mat)

    return ied


def save_model(output_path: str):
    """Save current scene as .blend file.

    Args:
        output_path: Path to save .blend file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Saved: {output_path}")


def main():
    """Generate all placeholder mine models."""
    output_dir = Path("data/synthetic/mine_models")

    print("Creating placeholder mine models...")

    # AP Blast mines
    mine = create_ap_blast_mine("PMN2", diameter=0.10, height=0.04)
    save_model(str(output_dir / "ap_blast" / "pmn2.blend"))

    mine = create_ap_blast_mine("M14", diameter=0.056, height=0.04)
    save_model(str(output_dir / "ap_blast" / "m14.blend"))

    mine = create_ap_blast_mine("Type72", diameter=0.08, height=0.035)
    save_model(str(output_dir / "ap_blast" / "type72.blend"))

    # AT Blast mines
    mine = create_at_blast_mine("TM62M", diameter=0.30, height=0.12)
    save_model(str(output_dir / "at_blast" / "tm62m.blend"))

    mine = create_at_blast_mine("M15", diameter=0.33, height=0.13)
    save_model(str(output_dir / "at_blast" / "m15.blend"))

    # UXO
    mine = create_mortar_uxo("Mortar_81mm", diameter=0.081, length=0.35)
    save_model(str(output_dir / "uxo" / "mortar_81mm.blend"))

    mine = create_mortar_uxo("Mortar_120mm", diameter=0.12, length=0.50)
    save_model(str(output_dir / "uxo" / "mortar_120mm.blend"))

    # IED
    mine = create_simple_ied("IED_Container", size=0.15)
    save_model(str(output_dir / "ied" / "simple_ied.blend"))

    mine = create_simple_ied("IED_Large", size=0.25)
    save_model(str(output_dir / "ied" / "large_ied.blend"))

    print("\nPlaceholder mine models created successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
