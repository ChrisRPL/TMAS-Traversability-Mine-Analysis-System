"""Visualization tools for COCO annotations on synthetic dataset.

This module provides utilities to visualize and verify the quality of
automatically generated COCO annotations on rendered synthetic images.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


class AnnotationVisualizer:
    """Visualize COCO annotations on images."""

    # Color map for different mine classes
    CLASS_COLORS = {
        "ap_blast": (1.0, 0.0, 0.0),           # Red
        "ap_fragmentation": (1.0, 0.5, 0.0),   # Orange
        "at_blast": (0.8, 0.0, 0.8),           # Purple
        "at_anti_handling": (0.6, 0.0, 1.0),   # Violet
        "submunition": (0.0, 1.0, 1.0),        # Cyan
        "ied": (1.0, 1.0, 0.0),                # Yellow
        "uxo_mortar": (0.0, 1.0, 0.0),         # Green
        "uxo_artillery": (0.0, 0.5, 1.0),      # Blue
    }

    def __init__(self, data_dir: str):
        """Initialize visualizer.

        Args:
            data_dir: Root directory of synthetic dataset
        """
        self.data_dir = Path(data_dir)

        # Load annotations
        annotation_file = self.data_dir / "annotations.json"
        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        # Build index
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

    def visualize_image(
        self,
        image_id: int,
        show_labels: bool = True,
        show_burial: bool = True,
        show_thermal: bool = False,
        save_path: Optional[str] = None
    ):
        """Visualize annotations for a single image.

        Args:
            image_id: Image ID to visualize
            show_labels: Show class labels
            show_burial: Show burial depth info
            show_thermal: Show thermal image alongside RGB
            save_path: Path to save figure (displays if None)
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")

        img_info = self.images[image_id]
        annotations = self.image_annotations.get(image_id, [])

        # Load RGB image
        rgb_path = self.data_dir / "rgb" / img_info["file_name"]
        rgb_image = Image.open(rgb_path)

        # Load thermal if requested
        thermal_image = None
        if show_thermal and "thermal_file_name" in img_info:
            thermal_path = self.data_dir / "thermal" / img_info["thermal_file_name"]
            if thermal_path.exists():
                thermal_image = Image.open(thermal_path)

        # Create figure
        if thermal_image is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            ax2 = None

        # Display RGB image
        ax1.imshow(rgb_image)
        ax1.set_title(f"RGB - {img_info['file_name']}", fontsize=14)
        ax1.axis("off")

        # Draw annotations on RGB
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            category = self.categories[cat_id]
            class_name = category["name"]

            # Get color for this class
            color = self.CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))

            # Draw bounding box
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor="none"
            )
            ax1.add_patch(rect)

            # Create label text
            label_parts = []
            if show_labels:
                display_name = category.get("display_name", class_name)
                label_parts.append(display_name)

            if show_burial and "burial_depth" in ann:
                depth = ann["burial_depth"]
                if depth > 0:
                    label_parts.append(f"Buried: {depth*100:.1f}cm")
                else:
                    label_parts.append("Surface")

            label_text = "\n".join(label_parts)

            # Draw label background
            if label_text:
                ax1.text(
                    x, y - 5,
                    label_text,
                    fontsize=9,
                    color="white",
                    backgroundcolor=(*color, 0.7),
                    verticalalignment="bottom"
                )

        # Display thermal image
        if ax2 is not None and thermal_image is not None:
            ax2.imshow(thermal_image, cmap="hot")
            ax2.set_title("Thermal", fontsize=14)
            ax2.axis("off")

            # Draw annotations on thermal too
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                # Scale to thermal resolution (640x512 vs 1280x720)
                scale_x = 640 / 1280
                scale_y = 512 / 720
                x_t = x * scale_x
                y_t = y * scale_y
                w_t = w * scale_x
                h_t = h * scale_y

                cat_id = ann["category_id"]
                class_name = self.categories[cat_id]["name"]
                color = self.CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))

                rect = patches.Rectangle(
                    (x_t, y_t), w_t, h_t,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none"
                )
                ax2.add_patch(rect)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_random_samples(
        self,
        num_samples: int = 5,
        output_dir: Optional[str] = None,
        show_thermal: bool = True
    ):
        """Visualize random sample images.

        Args:
            num_samples: Number of random samples to visualize
            output_dir: Directory to save visualizations (displays if None)
            show_thermal: Show thermal images
        """
        # Get random image IDs
        image_ids = list(self.images.keys())
        sample_ids = random.sample(image_ids, min(num_samples, len(image_ids)))

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for i, img_id in enumerate(sample_ids):
            save_path = None
            if output_dir:
                save_path = str(output_path / f"sample_{i:03d}.png")

            self.visualize_image(
                img_id,
                show_labels=True,
                show_burial=True,
                show_thermal=show_thermal,
                save_path=save_path
            )

    def create_class_grid(
        self,
        samples_per_class: int = 3,
        output_path: Optional[str] = None
    ):
        """Create grid showing examples of each class.

        Args:
            samples_per_class: Number of samples per class
            output_path: Path to save figure
        """
        # Group annotations by class
        class_samples = {cat_id: [] for cat_id in self.categories.keys()}

        for img_id, annotations in self.image_annotations.items():
            for ann in annotations:
                cat_id = ann["category_id"]
                class_samples[cat_id].append((img_id, ann))

        # Create grid
        num_classes = len(self.categories)
        fig, axes = plt.subplots(
            num_classes, samples_per_class,
            figsize=(samples_per_class * 3, num_classes * 3)
        )

        if num_classes == 1:
            axes = axes.reshape(1, -1)

        for row, (cat_id, category) in enumerate(self.categories.items()):
            samples = class_samples[cat_id]
            random.shuffle(samples)

            for col in range(samples_per_class):
                ax = axes[row, col]

                if col < len(samples):
                    img_id, ann = samples[col]
                    img_info = self.images[img_id]

                    # Load and crop image to bbox
                    rgb_path = self.data_dir / "rgb" / img_info["file_name"]
                    rgb_image = Image.open(rgb_path)

                    x, y, w, h = ann["bbox"]
                    # Add margin
                    margin = 0.2
                    x1 = max(0, int(x - w * margin))
                    y1 = max(0, int(y - h * margin))
                    x2 = min(rgb_image.width, int(x + w * (1 + margin)))
                    y2 = min(rgb_image.height, int(y + h * (1 + margin)))

                    cropped = rgb_image.crop((x1, y1, x2, y2))
                    ax.imshow(cropped)

                    burial = ann.get("burial_depth", 0.0)
                    title = f"{category['display_name']}\nBurial: {burial*100:.0f}cm"
                    ax.set_title(title, fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No sample", ha="center", va="center")

                ax.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved class grid to {output_path}")
        else:
            plt.show()

        plt.close()

    def print_statistics(self):
        """Print annotation statistics."""
        print("=" * 60)
        print("Annotation Statistics")
        print("=" * 60)

        total_images = len(self.images)
        total_annotations = len(self.coco_data["annotations"])

        print(f"\nTotal images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"Avg annotations per image: {total_annotations / total_images:.2f}")

        # Class distribution
        print(f"\nClass distribution:")
        class_counts = {}
        burial_depths = {cat_id: [] for cat_id in self.categories.keys()}

        for ann in self.coco_data["annotations"]:
            cat_id = ann["category_id"]
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

            if "burial_depth" in ann:
                burial_depths[cat_id].append(ann["burial_depth"])

        for cat_id, category in sorted(self.categories.items()):
            count = class_counts.get(cat_id, 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0

            depths = burial_depths[cat_id]
            if depths:
                avg_burial = np.mean(depths)
                print(f"  {category['display_name']:20s}: {count:5d} ({percentage:5.1f}%) "
                      f"- Avg burial: {avg_burial*100:.1f}cm")
            else:
                print(f"  {category['display_name']:20s}: {count:5d} ({percentage:5.1f}%)")

        print("=" * 60)


def main():
    """Main entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize synthetic dataset annotations")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to synthetic dataset")
    parser.add_argument("--random-samples", type=int, default=5,
                       help="Number of random samples to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--class-grid", action="store_true",
                       help="Create class example grid")
    parser.add_argument("--no-thermal", action="store_true",
                       help="Don't show thermal images")

    args = parser.parse_args()

    visualizer = AnnotationVisualizer(args.data_dir)

    # Print statistics
    visualizer.print_statistics()

    # Visualize random samples
    if args.random_samples > 0:
        print(f"\nVisualizing {args.random_samples} random samples...")
        visualizer.visualize_random_samples(
            num_samples=args.random_samples,
            output_dir=args.output_dir,
            show_thermal=not args.no_thermal
        )

    # Create class grid
    if args.class_grid:
        print("\nCreating class example grid...")
        grid_path = None
        if args.output_dir:
            grid_path = str(Path(args.output_dir) / "class_grid.png")

        visualizer.create_class_grid(
            samples_per_class=3,
            output_path=grid_path
        )


if __name__ == "__main__":
    main()
