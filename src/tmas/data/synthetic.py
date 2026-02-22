"""PyTorch DataLoader for synthetic mine detection dataset.

This module provides a DataLoader for the synthetically generated
mine detection dataset with RGB, thermal, and COCO annotations.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import json
from PIL import Image
import numpy as np


class SyntheticMineDataset(Dataset):
    """Synthetic mine detection dataset loader."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        load_thermal: bool = True,
        target_transform: Optional[Callable] = None
    ):
        """Initialize synthetic dataset.

        Args:
            data_dir: Root directory of synthetic dataset
            split: Dataset split (train/val/test)
            transform: Image transformations
            load_thermal: Whether to load thermal images
            target_transform: Target transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_thermal = load_thermal
        self.target_transform = target_transform

        # Load annotations
        self.annotation_file = self.data_dir / "annotations.json"
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotations not found: {self.annotation_file}")

        with open(self.annotation_file) as f:
            self.coco_data = json.load(f)

        # Build index
        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Group annotations by image ID
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        # Filter by split if split manifest exists
        split_file = self.data_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                split_ids = set(json.load(f))
            self.images = [img for img in self.images if img["id"] in split_ids]

        print(f"Loaded {len(self.images)} images for {split} split")

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of images
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item.

        Args:
            idx: Image index

        Returns:
            Dictionary with image, thermal (optional), boxes, labels, metadata
        """
        img_info = self.images[idx]
        img_id = img_info["id"]

        # Load RGB image
        rgb_path = self.data_dir / "rgb" / img_info["file_name"]
        rgb_image = Image.open(rgb_path).convert("RGB")

        # Load thermal image if requested
        thermal_image = None
        if self.load_thermal and "thermal_file_name" in img_info:
            thermal_path = self.data_dir / "thermal" / img_info["thermal_file_name"]
            if thermal_path.exists():
                thermal_image = Image.open(thermal_path).convert("L")

        # Get annotations for this image
        annotations = self.image_to_annotations.get(img_id, [])

        # Extract boxes and labels
        boxes = []
        labels = []
        burial_depths = []
        weathering = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

            # Additional metadata
            burial_depths.append(ann.get("burial_depth", 0.0))
            weathering.append(ann.get("weathering", 0.5))

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor([ann["area"] for ann in annotations], dtype=torch.float32),
            "iscrowd": torch.zeros((len(annotations),), dtype=torch.int64),
            "burial_depth": torch.as_tensor(burial_depths, dtype=torch.float32),
            "weathering": torch.as_tensor(weathering, dtype=torch.float32)
        }

        # Apply transforms
        if self.transform is not None:
            rgb_image, target = self.transform(rgb_image, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Build output dict
        output = {
            "image": rgb_image,
            "target": target,
            "metadata": {
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"]
            }
        }

        if thermal_image is not None:
            output["thermal"] = thermal_image

        return output

    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID.

        Args:
            category_id: Category ID

        Returns:
            Category name
        """
        return self.categories.get(category_id, {}).get("name", "unknown")

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_images": len(self.images),
            "total_annotations": len(self.annotations),
            "avg_annotations_per_image": 0.0,
            "class_distribution": {},
            "burial_depth_stats": {
                "min": float("inf"),
                "max": float("-inf"),
                "mean": 0.0
            }
        }

        if stats["total_images"] > 0:
            stats["avg_annotations_per_image"] = (
                stats["total_annotations"] / stats["total_images"]
            )

        # Class distribution
        burial_depths = []
        for ann in self.annotations:
            cat_id = ann["category_id"]
            cat_name = self.get_category_name(cat_id)
            stats["class_distribution"][cat_name] = \
                stats["class_distribution"].get(cat_name, 0) + 1

            # Burial depth stats
            if "burial_depth" in ann:
                burial_depths.append(ann["burial_depth"])

        # Burial depth statistics
        if burial_depths:
            stats["burial_depth_stats"]["min"] = min(burial_depths)
            stats["burial_depth_stats"]["max"] = max(burial_depths)
            stats["burial_depth_stats"]["mean"] = np.mean(burial_depths)

        return stats


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching.

    Args:
        batch: List of dataset items

    Returns:
        Batched dictionary
    """
    images = []
    thermals = []
    targets = []
    metadata = []

    for item in batch:
        images.append(item["image"])
        targets.append(item["target"])
        metadata.append(item["metadata"])

        if "thermal" in item:
            thermals.append(item["thermal"])

    # Stack images
    images = torch.stack(images, dim=0)

    output = {
        "images": images,
        "targets": targets,  # Keep as list (variable number of boxes)
        "metadata": metadata
    }

    if thermals:
        thermals = torch.stack(thermals, dim=0)
        output["thermals"] = thermals

    return output


def create_split_files(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/val/test split files.

    Args:
        data_dir: Dataset directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_ids, val_ids, test_ids)
    """
    data_path = Path(data_dir)
    annotation_file = data_path / "annotations.json"

    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Get all image IDs
    image_ids = [img["id"] for img in coco_data["images"]]

    # Shuffle with fixed seed
    np.random.seed(seed)
    np.random.shuffle(image_ids)

    # Calculate split sizes
    total = len(image_ids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Split
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:train_size + val_size]
    test_ids = image_ids[train_size + val_size:]

    # Save split files
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_file = data_path / f"{split_name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_ids, f)

        print(f"Created {split_name} split: {len(split_ids)} images")

    return train_ids, val_ids, test_ids


def main():
    """Test synthetic dataset loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test synthetic dataset loader")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to synthetic dataset")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create train/val/test splits")

    args = parser.parse_args()

    # Create splits if requested
    if args.create_splits:
        print("Creating dataset splits...")
        create_split_files(args.data_dir)

    # Load dataset
    print("\nLoading train dataset...")
    dataset = SyntheticMineDataset(
        data_dir=args.data_dir,
        split="train",
        load_thermal=True
    )

    # Print statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Avg annotations per image: {stats['avg_annotations_per_image']:.2f}")
    print(f"\nClass distribution:")
    for class_name, count in stats["class_distribution"].items():
        print(f"  {class_name}: {count}")

    print(f"\nBurial depth statistics:")
    print(f"  Min: {stats['burial_depth_stats']['min']:.3f}m")
    print(f"  Max: {stats['burial_depth_stats']['max']:.3f}m")
    print(f"  Mean: {stats['burial_depth_stats']['mean']:.3f}m")

    # Test data loading
    print(f"\nTesting data loading...")
    item = dataset[0]
    print(f"  Image shape: {item['image'].size}")
    if "thermal" in item:
        print(f"  Thermal shape: {item['thermal'].size}")
    print(f"  Boxes: {item['target']['boxes'].shape}")
    print(f"  Labels: {item['target']['labels'].shape}")

    print("\nDataset test successful!")


if __name__ == "__main__":
    main()
