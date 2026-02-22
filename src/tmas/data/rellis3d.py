"""PyTorch DataLoader for RELLIS-3D terrain segmentation dataset.

RELLIS-3D is an off-road terrain dataset with 13,556 frames containing
RGB images and semantic segmentation masks for 14 terrain classes.

Dataset: https://github.com/unmannedlab/RELLIS-3D
Paper: https://arxiv.org/abs/2011.12954
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import json
import numpy as np
from PIL import Image


class RELLIS3DDataset(Dataset):
    """RELLIS-3D terrain segmentation dataset loader."""

    # 14 terrain classes based on RELLIS-3D annotations
    CLASSES = [
        "void",           # 0 - Unknown/unlabeled
        "grass",          # 1 - Short grass
        "tree",           # 2 - Trees
        "pole",           # 3 - Poles, posts
        "water",          # 4 - Water bodies
        "sky",            # 5 - Sky
        "vehicle",        # 6 - Vehicles
        "object",         # 7 - Generic objects
        "asphalt",        # 8 - Paved roads
        "building",       # 9 - Buildings
        "log",            # 10 - Fallen logs
        "person",         # 11 - People
        "fence",          # 12 - Fences
        "bush",           # 13 - Bushes/shrubs
        "concrete",       # 14 - Concrete surfaces
        "barrier",        # 15 - Barriers
        "puddle",         # 16 - Puddles
        "mud",            # 17 - Mud
        "rubble"          # 18 - Rubble/debris
    ]

    # Map to traversability cost (0.0 = easy, 1.0 = impossible)
    TRAVERSABILITY_COST = {
        0: 1.0,   # void - unknown
        1: 0.15,  # grass
        2: 1.0,   # tree - obstacle
        3: 1.0,   # pole - obstacle
        4: 1.0,   # water - impassable
        5: 0.0,   # sky - ignore
        6: 1.0,   # vehicle - obstacle
        7: 0.8,   # object - likely obstacle
        8: 0.0,   # asphalt - easy
        9: 1.0,   # building - obstacle
        10: 0.9,  # log - difficult
        11: 1.0,  # person - must stop
        12: 1.0,  # fence - obstacle
        13: 0.5,  # bush - difficult
        14: 0.1,  # concrete - easy
        15: 1.0,  # barrier - obstacle
        16: 0.6,  # puddle - difficult
        17: 0.4,  # mud - difficult
        18: 0.8   # rubble - difficult
    }

    # Color map for visualization (RGB)
    COLOR_MAP = [
        (0, 0, 0),        # void - black
        (108, 64, 20),    # grass - brown
        (0, 102, 0),      # tree - dark green
        (0, 255, 0),      # pole - bright green
        (0, 153, 153),    # water - cyan
        (0, 128, 255),    # sky - blue
        (0, 0, 255),      # vehicle - blue
        (255, 255, 0),    # object - yellow
        (255, 0, 127),    # asphalt - pink
        (64, 64, 64),     # building - gray
        (255, 128, 0),    # log - orange
        (255, 0, 0),      # person - red
        (153, 76, 0),     # fence - brown
        (102, 102, 0),    # bush - olive
        (102, 0, 0),      # concrete - maroon
        (0, 255, 128),    # barrier - light green
        (204, 153, 255),  # puddle - lavender
        (102, 0, 204),    # mud - purple
        (255, 153, 204)   # rubble - light pink
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_paths: bool = False,
        ignore_index: int = 255
    ):
        """Initialize RELLIS-3D dataset.

        Args:
            data_dir: Root directory of RELLIS-3D dataset
            split: Dataset split (train/val/test)
            transform: Image and mask transformations
            target_transform: Additional mask transformations
            return_paths: Return file paths in output
            ignore_index: Index for ignored pixels in loss
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.ignore_index = ignore_index

        # RELLIS-3D directory structure
        self.images_dir = self.data_dir / "Rellis-3D" / "images"
        self.labels_dir = self.data_dir / "Rellis-3D" / "annotations"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Load split file
        split_file = self.data_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.file_list = json.load(f)
        else:
            # Auto-generate file list from all sequences
            self.file_list = self._scan_sequences()

        print(f"Loaded {len(self.file_list)} images for {split} split")

    def _scan_sequences(self) -> List[Dict[str, str]]:
        """Scan sequences directory and build file list.

        Returns:
            List of file info dictionaries
        """
        file_list = []

        # RELLIS-3D has sequences 00000-00004
        for seq_dir in sorted(self.images_dir.glob("*")):
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            # Find all images in sequence
            for img_path in sorted(seq_dir.glob("*.jpg")):
                # Corresponding annotation path
                label_path = self.labels_dir / seq_name / img_path.name.replace(".jpg", ".png")

                if label_path.exists():
                    file_list.append({
                        "image": str(img_path.relative_to(self.data_dir)),
                        "label": str(label_path.relative_to(self.data_dir)),
                        "sequence": seq_name
                    })

        return file_list

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of images
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item.

        Args:
            idx: Image index

        Returns:
            Dictionary with image, mask, and optional metadata
        """
        file_info = self.file_list[idx]

        # Load RGB image
        img_path = self.data_dir / file_info["image"]
        image = Image.open(img_path).convert("RGB")

        # Load segmentation mask
        label_path = self.data_dir / file_info["label"]
        mask = Image.open(label_path)

        # Convert mask to numpy array
        mask = np.array(mask, dtype=np.int64)

        # Apply transforms
        if self.transform is not None:
            # Albumentations expects numpy arrays
            image_np = np.array(image)
            transformed = self.transform(image=image_np, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # Build output
        output = {
            "image": image,
            "mask": mask,
            "sequence": file_info["sequence"]
        }

        if self.return_paths:
            output["image_path"] = str(img_path)
            output["mask_path"] = str(label_path)

        return output

    @staticmethod
    def get_class_name(class_id: int) -> str:
        """Get class name from ID.

        Args:
            class_id: Class ID

        Returns:
            Class name
        """
        if 0 <= class_id < len(RELLIS3DDataset.CLASSES):
            return RELLIS3DDataset.CLASSES[class_id]
        return "unknown"

    @staticmethod
    def get_traversability_cost(class_id: int) -> float:
        """Get traversability cost for class.

        Args:
            class_id: Class ID

        Returns:
            Cost value (0.0 = easy, 1.0 = impossible)
        """
        return RELLIS3DDataset.TRAVERSABILITY_COST.get(class_id, 1.0)

    @staticmethod
    def get_color(class_id: int) -> Tuple[int, int, int]:
        """Get visualization color for class.

        Args:
            class_id: Class ID

        Returns:
            RGB color tuple
        """
        if 0 <= class_id < len(RELLIS3DDataset.COLOR_MAP):
            return RELLIS3DDataset.COLOR_MAP[class_id]
        return (0, 0, 0)

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_images": len(self.file_list),
            "split": self.split,
            "num_classes": len(self.CLASSES),
            "sequences": {}
        }

        # Count images per sequence
        for file_info in self.file_list:
            seq = file_info["sequence"]
            stats["sequences"][seq] = stats["sequences"].get(seq, 0) + 1

        return stats

    @staticmethod
    def decode_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
        """Decode segmentation mask to RGB visualization.

        Args:
            mask: Segmentation mask [H, W] with class IDs

        Returns:
            RGB image [H, W, 3]
        """
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(len(RELLIS3DDataset.COLOR_MAP)):
            color = RELLIS3DDataset.COLOR_MAP[class_id]
            rgb[mask == class_id] = color

        return rgb


def create_rellis3d_splits(
    data_dir: str,
    train_sequences: List[str] = ["00000", "00001", "00002"],
    val_sequences: List[str] = ["00003"],
    test_sequences: List[str] = ["00004"]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/val/test splits by sequence.

    Args:
        data_dir: Root directory of RELLIS-3D
        train_sequences: Sequences for training
        val_sequences: Sequences for validation
        test_sequences: Sequences for testing

    Returns:
        (train_files, val_files, test_files)
    """
    data_path = Path(data_dir)
    images_dir = data_path / "Rellis-3D" / "images"
    labels_dir = data_path / "Rellis-3D" / "annotations"

    def get_sequence_files(sequences):
        files = []
        for seq in sequences:
            seq_img_dir = images_dir / seq
            if not seq_img_dir.exists():
                continue

            for img_path in sorted(seq_img_dir.glob("*.jpg")):
                label_path = labels_dir / seq / img_path.name.replace(".jpg", ".png")
                if label_path.exists():
                    files.append({
                        "image": str(img_path.relative_to(data_path)),
                        "label": str(label_path.relative_to(data_path)),
                        "sequence": seq
                    })
        return files

    train_files = get_sequence_files(train_sequences)
    val_files = get_sequence_files(val_sequences)
    test_files = get_sequence_files(test_sequences)

    # Save split files
    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_file = data_path / f"{split_name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_files, f, indent=2)

        print(f"Created {split_name} split: {len(split_files)} images")

    return train_files, val_files, test_files


def main():
    """Test RELLIS-3D dataset loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RELLIS-3D dataset loader")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to RELLIS-3D dataset")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create train/val/test splits")

    args = parser.parse_args()

    # Create splits if requested
    if args.create_splits:
        print("Creating dataset splits by sequence...")
        create_rellis3d_splits(args.data_dir)

    # Load dataset
    print("\nLoading train dataset...")
    dataset = RELLIS3DDataset(
        data_dir=args.data_dir,
        split="train"
    )

    # Print statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Split: {stats['split']}")
    print(f"  Num classes: {stats['num_classes']}")
    print(f"\nSequences:")
    for seq, count in stats["sequences"].items():
        print(f"  {seq}: {count} images")

    # Print class information
    print(f"\nTerrain Classes:")
    for i, class_name in enumerate(RELLIS3DDataset.CLASSES):
        cost = RELLIS3DDataset.get_traversability_cost(i)
        print(f"  {i:2d}: {class_name:15s} - Cost: {cost:.2f}")

    # Test data loading
    print(f"\nTesting data loading...")
    item = dataset[0]
    print(f"  Image shape: {item['image'].shape}")
    print(f"  Mask shape: {item['mask'].shape}")
    print(f"  Sequence: {item['sequence']}")

    print("\nDataset test successful!")


if __name__ == "__main__":
    main()
