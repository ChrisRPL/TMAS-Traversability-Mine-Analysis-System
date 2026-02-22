"""Unit tests for TMAS data loaders.

Tests for RELLIS-3D, thermal, synthetic, and augmentation pipelines.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
from PIL import Image


@pytest.fixture
def temp_rellis3d_dataset(tmp_path):
    """Create minimal RELLIS-3D dataset structure for testing."""
    # Create directory structure
    data_dir = tmp_path / "rellis3d"
    images_dir = data_dir / "Rellis-3D" / "images" / "00000"
    labels_dir = data_dir / "Rellis-3D" / "annotations" / "00000"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create dummy images and labels
    for i in range(5):
        # RGB image
        img = Image.new("RGB", (640, 480), color=(i * 50, 100, 150))
        img.save(images_dir / f"frame_{i:06d}.jpg")

        # Segmentation mask
        mask = np.random.randint(0, 19, (480, 640), dtype=np.uint8)
        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(labels_dir / f"frame_{i:06d}.png")

    # Create split file
    split_data = [
        {
            "image": f"Rellis-3D/images/00000/frame_{i:06d}.jpg",
            "label": f"Rellis-3D/annotations/00000/frame_{i:06d}.png",
            "sequence": "00000"
        }
        for i in range(5)
    ]

    with open(data_dir / "train.json", 'w') as f:
        json.dump(split_data, f)

    return str(data_dir)


@pytest.fixture
def temp_thermal_dataset(tmp_path):
    """Create minimal thermal dataset for testing."""
    data_dir = tmp_path / "thermal"
    data_dir.mkdir()

    # Create dummy thermal images
    for i in range(3):
        thermal = np.random.randint(0, 255, (512, 640), dtype=np.uint8)
        thermal_img = Image.fromarray(thermal, mode="L")
        thermal_img.save(data_dir / f"thermal_{i:03d}.png")

    # Create split file
    with open(data_dir / "train.json", 'w') as f:
        json.dump([f"thermal_{i:03d}.png" for i in range(3)], f)

    return str(data_dir)


@pytest.fixture
def temp_synthetic_dataset(tmp_path):
    """Create minimal synthetic dataset for testing."""
    data_dir = tmp_path / "synthetic"
    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"

    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir(parents=True)

    # Create dummy images
    for i in range(3):
        # RGB
        rgb = Image.new("RGB", (1280, 720), color=(100, 150, 200))
        rgb.save(rgb_dir / f"scene_{i:06d}_rgb.png")

        # Thermal
        thermal = Image.new("L", (640, 512), color=128)
        thermal.save(thermal_dir / f"scene_{i:06d}_thermal.png")

    # Create COCO annotations
    annotations = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": i + 1,
                "file_name": f"scene_{i:06d}_rgb.png",
                "thermal_file_name": f"scene_{i:06d}_thermal.png",
                "width": 1280,
                "height": 720
            }
            for i in range(3)
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [100, 100, 50, 40],
                "area": 2000,
                "iscrowd": 0,
                "burial_depth": 0.05,
                "weathering": 0.5
            }
        ],
        "categories": [
            {"id": i + 1, "name": f"class_{i}", "supercategory": "explosive_threat"}
            for i in range(8)
        ]
    }

    with open(data_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f)

    return str(data_dir)


class TestRELLIS3DDataset:
    """Tests for RELLIS-3D dataset loader."""

    def test_dataset_initialization(self, temp_rellis3d_dataset):
        """Test dataset can be initialized."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        dataset = RELLIS3DDataset(
            data_dir=temp_rellis3d_dataset,
            split="train"
        )

        assert len(dataset) == 5

    def test_dataset_getitem(self, temp_rellis3d_dataset):
        """Test getting items from dataset."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        dataset = RELLIS3DDataset(
            data_dir=temp_rellis3d_dataset,
            split="train"
        )

        item = dataset[0]

        assert "image" in item
        assert "mask" in item
        assert "sequence" in item
        assert item["image"].shape[0] == 3  # RGB channels
        assert item["mask"].ndim == 2  # 2D mask

    def test_class_names(self):
        """Test class name retrieval."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        assert RELLIS3DDataset.get_class_name(0) == "void"
        assert RELLIS3DDataset.get_class_name(1) == "grass"
        assert RELLIS3DDataset.get_class_name(8) == "asphalt"

    def test_traversability_cost(self):
        """Test traversability cost mapping."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        # Asphalt should be easy (low cost)
        assert RELLIS3DDataset.get_traversability_cost(8) == 0.0

        # Tree should be obstacle (high cost)
        assert RELLIS3DDataset.get_traversability_cost(2) == 1.0

        # Grass should be moderate
        assert 0.0 < RELLIS3DDataset.get_traversability_cost(1) < 1.0

    def test_color_map(self):
        """Test color map for visualization."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        color = RELLIS3DDataset.get_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_decode_mask(self):
        """Test mask to RGB decoding."""
        from src.tmas.data.rellis3d import RELLIS3DDataset

        mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        rgb = RELLIS3DDataset.decode_mask_to_rgb(mask)

        assert rgb.shape == (2, 3, 3)
        assert rgb.dtype == np.uint8


class TestThermalDataset:
    """Tests for thermal dataset loader."""

    def test_dataset_initialization(self, temp_thermal_dataset):
        """Test thermal dataset initialization."""
        from src.tmas.data.thermal import ThermalDataset

        dataset = ThermalDataset(
            data_dir=temp_thermal_dataset,
            split="train"
        )

        assert len(dataset) == 3

    def test_dataset_getitem(self, temp_thermal_dataset):
        """Test getting thermal images."""
        from src.tmas.data.thermal import ThermalDataset

        dataset = ThermalDataset(
            data_dir=temp_thermal_dataset,
            split="train",
            normalize_method="minmax"
        )

        item = dataset[0]

        assert "thermal" in item
        assert item["thermal"].shape[0] == 1  # Single channel
        assert item["thermal"].dtype == torch.float32

    def test_normalization_methods(self, temp_thermal_dataset):
        """Test different normalization methods."""
        from src.tmas.data.thermal import ThermalDataset

        methods = ["minmax", "zscore", "percentile"]

        for method in methods:
            dataset = ThermalDataset(
                data_dir=temp_thermal_dataset,
                split="train",
                normalize_method=method
            )

            item = dataset[0]
            thermal = item["thermal"]

            # Check values are in reasonable range
            assert thermal.min() >= 0.0
            assert thermal.max() <= 1.0


class TestSyntheticDataset:
    """Tests for synthetic mine dataset loader."""

    def test_dataset_initialization(self, temp_synthetic_dataset):
        """Test synthetic dataset initialization."""
        from src.tmas.data.synthetic import SyntheticMineDataset

        dataset = SyntheticMineDataset(
            data_dir=temp_synthetic_dataset,
            split="train",
            load_thermal=True
        )

        assert len(dataset) == 3

    def test_dataset_getitem(self, temp_synthetic_dataset):
        """Test getting items with annotations."""
        from src.tmas.data.synthetic import SyntheticMineDataset

        dataset = SyntheticMineDataset(
            data_dir=temp_synthetic_dataset,
            split="train",
            load_thermal=True
        )

        item = dataset[0]

        assert "image" in item
        assert "thermal" in item
        assert "target" in item
        assert "metadata" in item

        # Check target structure
        target = item["target"]
        assert "boxes" in target
        assert "labels" in target
        assert "burial_depth" in target

    def test_statistics(self, temp_synthetic_dataset):
        """Test dataset statistics."""
        from src.tmas.data.synthetic import SyntheticMineDataset

        dataset = SyntheticMineDataset(
            data_dir=temp_synthetic_dataset,
            split="train"
        )

        stats = dataset.get_statistics()

        assert "total_images" in stats
        assert "total_annotations" in stats
        assert "class_distribution" in stats


class TestAugmentation:
    """Tests for augmentation pipelines."""

    def test_terrain_augmentation(self):
        """Test terrain segmentation augmentation."""
        from src.tmas.data.augmentation import get_augmentation

        aug = get_augmentation("terrain_seg", "train")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)

        transformed = aug(image=image, mask=mask)

        assert "image" in transformed
        assert "mask" in transformed
        assert isinstance(transformed["image"], torch.Tensor)
        assert isinstance(transformed["mask"], torch.Tensor)

    def test_mine_detection_augmentation(self):
        """Test mine detection augmentation with bboxes."""
        from src.tmas.data.augmentation import get_augmentation

        aug = get_augmentation("mine_det", "train")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        boxes = np.array([
            [100, 100, 200, 150],
            [300, 400, 380, 460]
        ], dtype=np.float32)
        labels = np.array([1, 3], dtype=np.int64)

        transformed = aug(image=image, bboxes=boxes, labels=labels)

        assert "image" in transformed
        assert "bboxes" in transformed
        assert "labels" in transformed
        assert len(transformed["bboxes"]) <= len(boxes)  # May filter out boxes

    def test_validation_augmentation(self):
        """Test validation augmentation (minimal transforms)."""
        from src.tmas.data.augmentation import get_augmentation

        aug = get_augmentation("terrain_seg", "val")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)

        transformed = aug(image=image, mask=mask)

        # Should only resize and normalize, not flip/rotate
        assert transformed["image"].shape[-2:] == (720, 1280)


def test_data_loading_performance(temp_rellis3d_dataset):
    """Test data loading performance."""
    from src.tmas.data.rellis3d import RELLIS3DDataset
    from torch.utils.data import DataLoader
    import time

    dataset = RELLIS3DDataset(
        data_dir=temp_rellis3d_dataset,
        split="train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False
    )

    start_time = time.time()
    for batch in dataloader:
        pass
    elapsed = time.time() - start_time

    # Should load quickly (dummy data)
    assert elapsed < 2.0

    print(f"Loaded {len(dataset)} samples in {elapsed:.2f}s")
