"""PyTorch DataLoader for thermal imaging datasets.

This module provides data loaders for thermal (LWIR) imaging datasets,
including FLIR Thermal Dataset and custom thermal data. Supports both
standalone thermal images and RGB-Thermal paired datasets.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import json
import numpy as np
from PIL import Image


class ThermalDataset(Dataset):
    """General thermal imaging dataset loader."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        normalize_method: str = "minmax",
        temperature_range: Tuple[float, float] = (253.15, 353.15),
        return_paths: bool = False
    ):
        """Initialize thermal dataset.

        Args:
            data_dir: Root directory of thermal dataset
            split: Dataset split (train/val/test)
            transform: Image transformations
            normalize_method: Normalization method (minmax/zscore/percentile)
            temperature_range: Temperature range in Kelvin (min, max)
            return_paths: Return file paths in output
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize_method = normalize_method
        self.temperature_range = temperature_range
        self.return_paths = return_paths

        # Load file list
        split_file = self.data_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.file_list = json.load(f)
        else:
            # Scan for thermal images
            self.file_list = self._scan_directory()

        print(f"Loaded {len(self.file_list)} thermal images for {split} split")

    def _scan_directory(self) -> List[str]:
        """Scan directory for thermal images.

        Returns:
            List of image paths
        """
        extensions = [".tiff", ".tif", ".png", ".jpg"]
        image_files = []

        for ext in extensions:
            image_files.extend(self.data_dir.glob(f"**/*{ext}"))

        # Convert to relative paths
        return [str(p.relative_to(self.data_dir)) for p in sorted(image_files)]

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of images
        """
        return len(self.file_list)

    def _normalize_thermal(self, thermal: np.ndarray) -> np.ndarray:
        """Normalize thermal image.

        Args:
            thermal: Raw thermal image

        Returns:
            Normalized thermal image [0, 1]
        """
        if self.normalize_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = thermal.min()
            max_val = thermal.max()

            if max_val > min_val:
                thermal = (thermal - min_val) / (max_val - min_val)
            else:
                thermal = np.zeros_like(thermal)

        elif self.normalize_method == "zscore":
            # Z-score normalization
            mean = thermal.mean()
            std = thermal.std()

            if std > 0:
                thermal = (thermal - mean) / std
                # Clip to reasonable range and scale to [0, 1]
                thermal = np.clip(thermal, -3, 3)
                thermal = (thermal + 3) / 6
            else:
                thermal = np.zeros_like(thermal)

        elif self.normalize_method == "percentile":
            # Percentile-based normalization (robust to outliers)
            p_low = np.percentile(thermal, 2)
            p_high = np.percentile(thermal, 98)

            if p_high > p_low:
                thermal = (thermal - p_low) / (p_high - p_low)
                thermal = np.clip(thermal, 0, 1)
            else:
                thermal = np.zeros_like(thermal)

        elif self.normalize_method == "temperature":
            # Normalize based on expected temperature range
            min_temp, max_temp = self.temperature_range
            thermal = (thermal - min_temp) / (max_temp - min_temp)
            thermal = np.clip(thermal, 0, 1)

        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

        return thermal

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item.

        Args:
            idx: Image index

        Returns:
            Dictionary with thermal image and metadata
        """
        file_path = self.file_list[idx]
        img_path = self.data_dir / file_path

        # Load thermal image
        thermal = Image.open(img_path)

        # Convert to grayscale if RGB
        if thermal.mode == "RGB":
            thermal = thermal.convert("L")

        # Convert to numpy array
        thermal = np.array(thermal, dtype=np.float32)

        # Normalize
        thermal = self._normalize_thermal(thermal)

        # Apply transforms
        if self.transform is not None:
            # Albumentations expects 3-channel images
            thermal_3ch = np.stack([thermal, thermal, thermal], axis=-1)
            transformed = self.transform(image=thermal_3ch)
            thermal = transformed["image"][0:1]  # Keep only first channel
        else:
            # Convert to tensor
            thermal = torch.from_numpy(thermal).unsqueeze(0).float()

        # Build output
        output = {
            "thermal": thermal,
            "filename": file_path
        }

        if self.return_paths:
            output["path"] = str(img_path)

        return output


class RGBThermalDataset(Dataset):
    """RGB-Thermal paired dataset loader."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        thermal_normalize: str = "minmax",
        return_paths: bool = False
    ):
        """Initialize RGB-Thermal dataset.

        Args:
            data_dir: Root directory with rgb/ and thermal/ subdirs
            split: Dataset split (train/val/test)
            transform: Synchronized transformations for RGB and thermal
            thermal_normalize: Thermal normalization method
            return_paths: Return file paths
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.thermal_normalize = thermal_normalize
        self.return_paths = return_paths

        self.rgb_dir = self.data_dir / "rgb"
        self.thermal_dir = self.data_dir / "thermal"

        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.thermal_dir.exists():
            raise FileNotFoundError(f"Thermal directory not found: {self.thermal_dir}")

        # Load pairs
        split_file = self.data_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.file_list = json.load(f)
        else:
            self.file_list = self._find_pairs()

        print(f"Loaded {len(self.file_list)} RGB-Thermal pairs for {split} split")

    def _find_pairs(self) -> List[Dict[str, str]]:
        """Find RGB-Thermal image pairs.

        Returns:
            List of pair dictionaries
        """
        pairs = []

        # Find all RGB images
        for rgb_path in sorted(self.rgb_dir.glob("*.png")):
            # Look for corresponding thermal image
            thermal_path = self.thermal_dir / rgb_path.name

            # Try alternative naming conventions
            if not thermal_path.exists():
                # Try with _thermal suffix
                thermal_name = rgb_path.stem + "_thermal" + rgb_path.suffix
                thermal_path = self.thermal_dir / thermal_name

            if thermal_path.exists():
                pairs.append({
                    "rgb": str(rgb_path.relative_to(self.data_dir)),
                    "thermal": str(thermal_path.relative_to(self.data_dir))
                })

        return pairs

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of image pairs
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item.

        Args:
            idx: Image index

        Returns:
            Dictionary with RGB, thermal, and metadata
        """
        pair = self.file_list[idx]

        # Load RGB
        rgb_path = self.data_dir / pair["rgb"]
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb, dtype=np.float32) / 255.0

        # Load thermal
        thermal_path = self.data_dir / pair["thermal"]
        thermal = Image.open(thermal_path)

        if thermal.mode == "RGB":
            thermal = thermal.convert("L")

        thermal = np.array(thermal, dtype=np.float32)

        # Normalize thermal
        thermal_dataset = ThermalDataset(
            data_dir=str(self.data_dir),
            normalize_method=self.thermal_normalize
        )
        thermal = thermal_dataset._normalize_thermal(thermal)

        # Apply synchronized transforms
        if self.transform is not None:
            # Combine RGB and thermal for synchronized augmentation
            # Thermal as 4th channel
            combined = np.concatenate([rgb, thermal[..., np.newaxis]], axis=-1)
            transformed = self.transform(image=combined)
            combined = transformed["image"]

            # Split back
            rgb = combined[:3]
            thermal = combined[3:4]
        else:
            # Convert to tensors
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            thermal = torch.from_numpy(thermal).unsqueeze(0).float()

        output = {
            "rgb": rgb,
            "thermal": thermal,
            "filename": pair["rgb"]
        }

        if self.return_paths:
            output["rgb_path"] = str(rgb_path)
            output["thermal_path"] = str(thermal_path)

        return output


class FLIRThermalDataset(Dataset):
    """FLIR Thermal Dataset loader.

    Free thermal dataset: https://www.flir.com/oem/adas/adas-dataset-form/
    Contains 10,228 thermal images with annotations for people and vehicles.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        load_annotations: bool = False
    ):
        """Initialize FLIR thermal dataset.

        Args:
            data_dir: Root directory of FLIR dataset
            split: Dataset split (train/val/video)
            transform: Image transformations
            load_annotations: Load COCO annotations for detection
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_annotations = load_annotations

        # FLIR dataset structure: train/val/video
        self.images_dir = self.data_dir / split / "thermal_8_bit"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Get image list
        self.image_files = sorted(self.images_dir.glob("*.jpeg"))

        # Load annotations if requested
        self.annotations = None
        if load_annotations:
            ann_file = self.data_dir / split / "thermal_annotations.json"
            if ann_file.exists():
                with open(ann_file) as f:
                    self.annotations = json.load(f)

        print(f"Loaded {len(self.image_files)} FLIR thermal images for {split} split")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item."""
        img_path = self.image_files[idx]

        # Load thermal image (8-bit grayscale)
        thermal = Image.open(img_path).convert("L")
        thermal = np.array(thermal, dtype=np.float32) / 255.0

        # Apply transforms
        if self.transform is not None:
            thermal_3ch = np.stack([thermal, thermal, thermal], axis=-1)
            transformed = self.transform(image=thermal_3ch)
            thermal = transformed["image"][0:1]
        else:
            thermal = torch.from_numpy(thermal).unsqueeze(0).float()

        output = {
            "thermal": thermal,
            "filename": img_path.name
        }

        # Add annotations if available
        if self.annotations is not None:
            # Find annotations for this image
            # (Implementation depends on COCO format structure)
            pass

        return output


def main():
    """Test thermal dataset loaders."""
    import argparse

    parser = argparse.ArgumentParser(description="Test thermal dataset loaders")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to thermal dataset")
    parser.add_argument("--dataset-type", type=str, default="thermal",
                       choices=["thermal", "rgb-thermal", "flir"],
                       help="Dataset type")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")

    args = parser.parse_args()

    if args.dataset_type == "thermal":
        dataset = ThermalDataset(
            data_dir=args.data_dir,
            split=args.split,
            normalize_method="minmax"
        )
    elif args.dataset_type == "rgb-thermal":
        dataset = RGBThermalDataset(
            data_dir=args.data_dir,
            split=args.split
        )
    elif args.dataset_type == "flir":
        dataset = FLIRThermalDataset(
            data_dir=args.data_dir,
            split=args.split
        )

    print(f"\nDataset loaded: {len(dataset)} images")

    # Test loading
    print("\nTesting data loading...")
    item = dataset[0]

    if "thermal" in item:
        print(f"  Thermal shape: {item['thermal'].shape}")
    if "rgb" in item:
        print(f"  RGB shape: {item['rgb'].shape}")

    print("\nDataset test successful!")


if __name__ == "__main__":
    main()
