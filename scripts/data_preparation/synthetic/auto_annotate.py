"""Automatic COCO annotation generation from Blender scenes.

This module extracts mine positions, bounding boxes, and metadata from
Blender rendered scenes to generate COCO format annotations for training
object detection models.
"""

import bpy
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class COCOAnnotator:
    """Generate COCO format annotations from Blender scene."""

    # Mine class mapping
    MINE_CLASSES = {
        "ap_blast": 1,
        "ap_fragmentation": 2,
        "at_blast": 3,
        "at_anti_handling": 4,
        "submunition": 5,
        "ied": 6,
        "uxo_mortar": 7,
        "uxo_artillery": 8,
    }

    def __init__(self, scene_name: str = "Scene"):
        """Initialize COCO annotator.

        Args:
            scene_name: Blender scene name
        """
        self.scene = bpy.data.scenes.get(scene_name)
        if self.scene is None:
            raise ValueError(f"Scene '{scene_name}' not found")

        self.camera = self.scene.camera
        if self.camera is None:
            raise ValueError("No active camera in scene")

        self.annotations = {
            "info": self._create_info(),
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self._create_categories()
        }

        self.annotation_id = 1
        self.image_id = 1

    def _create_info(self) -> Dict:
        """Create COCO info section.

        Returns:
            Info dictionary
        """
        return {
            "description": "TMAS Synthetic Mine Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "TMAS Project",
            "date_created": datetime.now().isoformat()
        }

    def _create_categories(self) -> List[Dict]:
        """Create COCO categories for mine types.

        Returns:
            List of category dictionaries
        """
        categories = []

        for name, cat_id in self.MINE_CLASSES.items():
            # Format name for display
            display_name = name.replace("_", " ").title()

            categories.append({
                "id": cat_id,
                "name": name,
                "supercategory": "explosive_threat",
                "display_name": display_name
            })

        return categories

    def get_2d_bbox_from_object(
        self,
        obj: bpy.types.Object,
        margin: float = 0.05
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project 3D object to 2D bounding box in camera view.

        Args:
            obj: Blender object
            margin: Margin to add around bbox (fraction of width/height)

        Returns:
            (x, y, width, height) in pixels, or None if not visible
        """
        scene = self.scene
        camera = self.camera

        # Get render resolution
        render = scene.render
        res_x = render.resolution_x
        res_y = render.resolution_y

        # Get object bounding box corners in world space
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner)
                       for corner in obj.bound_box]

        # Project each corner to 2D
        coords_2d = []

        for corner in bbox_corners:
            # Convert to camera space
            co_camera = bpy_extras.object_utils.world_to_camera_view(
                scene, camera, corner
            )

            # Check if point is in front of camera
            if co_camera.z < 0:
                return None  # Behind camera

            # Convert normalized coords to pixel coords
            x = co_camera.x * res_x
            y = (1 - co_camera.y) * res_y  # Flip Y axis

            coords_2d.append((x, y))

        if not coords_2d:
            return None

        # Get bounding box
        xs = [x for x, y in coords_2d]
        ys = [y for x, y in coords_2d]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Add margin
        width = x_max - x_min
        height = y_max - y_min

        x_min -= width * margin
        y_min -= height * margin
        width *= (1 + 2 * margin)
        height *= (1 + 2 * margin)

        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(res_x, x_min + width)
        y_max = min(res_y, y_min + height)

        width = x_max - x_min
        height = y_max - y_min

        # Check if bbox is valid
        if width <= 0 or height <= 0:
            return None

        # Check if bbox is too small (likely occluded or far away)
        if width < 5 or height < 5:
            return None

        return (x_min, y_min, width, height)

    def extract_mine_info(self, obj: bpy.types.Object) -> Optional[Dict]:
        """Extract mine information from object.

        Args:
            obj: Mine object

        Returns:
            Dictionary with mine metadata or None
        """
        # Get mine class from object name
        obj_name_lower = obj.name.lower()
        mine_class = None

        for class_name in self.MINE_CLASSES.keys():
            if class_name in obj_name_lower:
                mine_class = class_name
                break

        if mine_class is None:
            # Default classification based on keywords
            if "mine" in obj_name_lower:
                if "ap" in obj_name_lower or "personnel" in obj_name_lower:
                    mine_class = "ap_blast"
                elif "at" in obj_name_lower or "tank" in obj_name_lower:
                    mine_class = "at_blast"
                else:
                    mine_class = "at_blast"  # Default
            elif "ied" in obj_name_lower:
                mine_class = "ied"
            elif "mortar" in obj_name_lower:
                mine_class = "uxo_mortar"
            elif "artillery" in obj_name_lower or "shell" in obj_name_lower:
                mine_class = "uxo_artillery"
            else:
                return None  # Unknown object

        # Get custom properties if they exist
        burial_depth = obj.get("burial_depth", 0.0)
        weathering = obj.get("weathering", 0.5)

        return {
            "class": mine_class,
            "category_id": self.MINE_CLASSES[mine_class],
            "burial_depth": burial_depth,
            "weathering": weathering,
            "position_3d": list(obj.location)
        }

    def annotate_current_scene(
        self,
        image_filename: str,
        thermal_filename: Optional[str] = None
    ) -> int:
        """Annotate current Blender scene.

        Args:
            image_filename: RGB image filename
            thermal_filename: Optional thermal image filename

        Returns:
            Number of annotations created
        """
        render = self.scene.render
        res_x = render.resolution_x
        res_y = render.resolution_y

        # Add image entry
        image_entry = {
            "id": self.image_id,
            "file_name": image_filename,
            "width": res_x,
            "height": res_y,
            "date_captured": datetime.now().isoformat()
        }

        if thermal_filename:
            image_entry["thermal_file_name"] = thermal_filename

        self.annotations["images"].append(image_entry)

        # Find all mine objects in scene
        annotation_count = 0

        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue

            # Extract mine info
            mine_info = self.extract_mine_info(obj)
            if mine_info is None:
                continue

            # Get 2D bounding box
            bbox = self.get_2d_bbox_from_object(obj)
            if bbox is None:
                continue  # Not visible in camera

            x, y, w, h = bbox

            # Create annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": mine_info["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "burial_depth": mine_info["burial_depth"],
                "weathering": mine_info["weathering"],
                "position_3d": mine_info["position_3d"]
            }

            self.annotations["annotations"].append(annotation)
            self.annotation_id += 1
            annotation_count += 1

        self.image_id += 1

        return annotation_count

    def save_annotations(self, output_path: str):
        """Save COCO annotations to JSON file.

        Args:
            output_path: Path to save annotations.json
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        print(f"Saved {len(self.annotations['annotations'])} annotations to {output_path}")

    def get_statistics(self) -> Dict:
        """Get annotation statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_images": len(self.annotations["images"]),
            "total_annotations": len(self.annotations["annotations"]),
            "annotations_per_image": 0.0,
            "class_distribution": {}
        }

        if stats["total_images"] > 0:
            stats["annotations_per_image"] = (
                stats["total_annotations"] / stats["total_images"]
            )

        # Count per class
        for ann in self.annotations["annotations"]:
            cat_id = ann["category_id"]
            category_name = None

            for cat in self.annotations["categories"]:
                if cat["id"] == cat_id:
                    category_name = cat["name"]
                    break

            if category_name:
                stats["class_distribution"][category_name] = \
                    stats["class_distribution"].get(category_name, 0) + 1

        return stats


# Import required Blender modules
try:
    import mathutils
    import bpy_extras
except ImportError:
    # Not running in Blender
    pass


def main():
    """Test annotation generation on current scene."""
    print("Generating COCO annotations for current scene...")

    annotator = COCOAnnotator()

    # Annotate current scene
    num_annotations = annotator.annotate_current_scene(
        image_filename="test_rgb.png",
        thermal_filename="test_thermal.png"
    )

    print(f"Created {num_annotations} annotations")

    # Save annotations
    annotator.save_annotations("annotations_test.json")

    # Print statistics
    stats = annotator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Images: {stats['total_images']}")
    print(f"  Annotations: {stats['total_annotations']}")
    print(f"  Avg per image: {stats['annotations_per_image']:.2f}")
    print(f"\nClass distribution:")
    for class_name, count in stats["class_distribution"].items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
