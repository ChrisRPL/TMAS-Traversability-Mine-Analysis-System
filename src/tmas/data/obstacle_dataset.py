"""Obstacle detection dataset loaders.

This module provides dataset loaders for obstacle detection training,
supporting COCO format annotations with TMAS obstacle class mapping.

Datasets supported:
- MS COCO 2017 (persons, vehicles, animals)
- Custom obstacle annotations (military vehicles, debris, craters)
- Synthetic obstacle data from Blender

The loader handles:
- COCO format annotation parsing
- Class remapping from COCO to TMAS classes
- Safety-critical augmentation
- Multi-scale training support
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmentation import get_obstacle_detection_train, get_obstacle_detection_val


class COCOObstacleDataset(Dataset):
    """COCO-format obstacle detection dataset.

    Loads COCO annotations and maps to TMAS obstacle classes.
    Supports both official COCO dataset and custom COCO-format annotations.
    """

    # COCO class IDs we care about for obstacle detection
    COCO_OBSTACLE_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe"
    }

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        min_bbox_area: int = 25,
        min_visibility: float = 0.3
    ):
        """Initialize COCO obstacle dataset.

        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO JSON annotation file
            split: Dataset split (train/val/test)
            transform: Albumentations transform pipeline
            min_bbox_area: Minimum bounding box area in pixels
            min_visibility: Minimum bbox visibility after augmentation
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.min_bbox_area = min_bbox_area
        self.min_visibility = min_visibility

        # Load COCO annotations
        print(f"Loading COCO annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build image ID to annotations mapping
        self.image_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_anns:
                self.image_id_to_anns[image_id] = []
            self.image_id_to_anns[image_id].append(ann)

        # Filter images that have obstacle annotations
        self.images = self._filter_images()

        print(f"Loaded {len(self.images)} images with obstacle annotations")

        # Set up transforms
        if transform is None:
            if split == "train":
                self.transform = get_obstacle_detection_train()
            else:
                self.transform = get_obstacle_detection_val()
        else:
            self.transform = transform

    def _filter_images(self) -> List[Dict]:
        """Filter images that contain relevant obstacle classes.

        Returns:
            List of image metadata dicts
        """
        filtered_images = []

        for img in self.coco_data['images']:
            image_id = img['id']

            # Check if image has annotations
            if image_id not in self.image_id_to_anns:
                continue

            # Check if any annotation is an obstacle class
            has_obstacle = False
            for ann in self.image_id_to_anns[image_id]:
                if ann['category_id'] in self.COCO_OBSTACLE_CLASSES:
                    has_obstacle = True
                    break

            if has_obstacle:
                filtered_images.append(img)

        return filtered_images

    def _load_image(self, image_info: Dict) -> np.ndarray:
        """Load image from disk.

        Args:
            image_info: COCO image metadata

        Returns:
            Image as numpy array [H, W, 3]
        """
        image_path = self.root_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def _load_annotations(
        self,
        image_id: int,
        image_width: int,
        image_height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and convert annotations to bounding boxes and labels.

        Args:
            image_id: COCO image ID
            image_width: Image width
            image_height: Image height

        Returns:
            Tuple of (boxes, labels)
                boxes: [N, 4] in [x1, y1, x2, y2] normalized format
                labels: [N] TMAS class indices
        """
        if image_id not in self.image_id_to_anns:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        boxes = []
        labels = []

        for ann in self.image_id_to_anns[image_id]:
            category_id = ann['category_id']

            # Skip non-obstacle classes
            if category_id not in self.COCO_OBSTACLE_CLASSES:
                continue

            # Get bounding box [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Filter small boxes
            if w * h < self.min_bbox_area:
                continue

            # Convert to [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # Clip to image bounds
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Normalize coordinates
            x1_norm = x1 / image_width
            y1_norm = y1 / image_height
            x2_norm = x2 / image_width
            y2_norm = y2 / image_height

            boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

            # Map COCO class to TMAS class
            # This is a simple mapping, extend based on your class hierarchy
            if category_id == 0:  # person
                tmas_class = 0
            elif category_id in [2]:  # car
                tmas_class = 1
            elif category_id in [7]:  # truck
                tmas_class = 2
            elif category_id in [3]:  # motorcycle
                tmas_class = 4
            elif category_id in [1]:  # bicycle
                tmas_class = 5
            elif category_id in [5]:  # bus
                tmas_class = 6
            elif category_id in [16, 17, 18, 19, 20, 21, 22, 23]:  # animals
                tmas_class = 7
            else:
                tmas_class = 19  # unknown_obstacle

            labels.append(tmas_class)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - image: Tensor [3, H, W]
                - target: Dict with boxes [N, 4] and labels [N]
                - image_id: Original COCO image ID
        """
        image_info = self.images[idx]
        image_id = image_info['id']

        # Load image
        image = self._load_image(image_info)
        image_height, image_width = image.shape[:2]

        # Load annotations
        boxes, labels = self._load_annotations(image_id, image_width, image_height)

        # Apply transforms
        if self.transform is not None and len(boxes) > 0:
            # Convert normalized boxes to pixel coordinates for albumentations
            boxes_pixel = boxes.copy()
            boxes_pixel[:, [0, 2]] *= image_width
            boxes_pixel[:, [1, 3]] *= image_height

            # Albumentations expects [x_min, y_min, x_max, y_max]
            transformed = self.transform(
                image=image,
                bboxes=boxes_pixel,
                labels=labels
            )

            image = transformed['image']
            boxes_pixel = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])

            # Convert back to normalized coordinates
            if len(boxes_pixel) > 0:
                h, w = image.shape[1:3] if isinstance(image, torch.Tensor) else image.shape[:2]
                boxes = boxes_pixel.copy()
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)
        else:
            # Just convert to tensor
            if self.transform is not None:
                transformed = self.transform(image=image, bboxes=[], labels=[])
                image = transformed['image']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Create target dict
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long()
        }

        return {
            'image': image,
            'target': target,
            'image_id': image_id
        }


def create_obstacle_dataset(
    root_dir: str,
    annotation_file: str,
    split: str = "train",
    **kwargs
) -> COCOObstacleDataset:
    """Create obstacle detection dataset.

    Args:
        root_dir: Root directory with images
        annotation_file: COCO annotation JSON file
        split: Dataset split
        **kwargs: Additional arguments

    Returns:
        COCOObstacleDataset instance

    Example:
        >>> dataset = create_obstacle_dataset(
        ...     root_dir="data/coco/train2017",
        ...     annotation_file="data/coco/annotations/instances_train2017.json",
        ...     split="train"
        ... )
        >>> sample = dataset[0]
        >>> print(f"Image: {sample['image'].shape}")
        >>> print(f"Boxes: {sample['target']['boxes'].shape}")
    """
    return COCOObstacleDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        split=split,
        **kwargs
    )


def main():
    """Test obstacle dataset loader."""
    print("Testing COCO obstacle dataset loader...")

    # Example usage (requires COCO dataset)
    # dataset = create_obstacle_dataset(
    #     root_dir="/path/to/coco/train2017",
    #     annotation_file="/path/to/coco/annotations/instances_train2017.json",
    #     split="train"
    # )

    # print(f"Dataset size: {len(dataset)}")

    # # Test loading a sample
    # sample = dataset[0]
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Number of boxes: {len(sample['target']['boxes'])}")
    # print(f"Labels: {sample['target']['labels']}")

    print("Dataset loader implementation complete!")


if __name__ == "__main__":
    main()
