"""Dataset registry and metadata management."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a single dataset."""

    name: str
    path: str
    dataset_type: str  # terrain_segmentation, mine_detection, obstacle_detection, etc.
    num_samples: int = 0
    num_classes: Optional[int] = None
    splits: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    status: str = "pending"  # pending, downloading, downloaded, verified
    date_acquired: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.dataset_type,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "splits": self.splits,
            "status": self.status,
            "date_acquired": self.date_acquired,
            "description": self.description,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            dataset_type=data.get("type", "unknown"),
            num_samples=data.get("num_samples", 0),
            num_classes=data.get("num_classes"),
            splits=data.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1}),
            status=data.get("status", "pending"),
            date_acquired=data.get("date_acquired"),
            description=data.get("description"),
            source_url=data.get("source_url"),
        )


class DatasetRegistry:
    """Central registry for all TMAS datasets."""

    def __init__(self, registry_path: str = "data/registry.json"):
        """Initialize dataset registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.datasets: Dict[str, DatasetMetadata] = {}

        # Create registry file if it doesn't exist
        if self.registry_path.exists():
            self.load()
        else:
            self._create_default_registry()

    def _create_default_registry(self):
        """Create default registry with known datasets."""
        logger.info(f"Creating default registry at {self.registry_path}")

        # Add known datasets
        self.datasets = {
            "rellis3d": DatasetMetadata(
                name="rellis3d",
                path="data/raw/rellis3d",
                dataset_type="terrain_segmentation",
                num_samples=13556,
                num_classes=14,
                status="pending",
                description="RELLIS-3D off-road terrain segmentation dataset",
                source_url="https://github.com/unmannedlab/RELLIS-3D",
            ),
            "tartandrive": DatasetMetadata(
                name="tartandrive",
                path="data/raw/tartandrive",
                dataset_type="depth_estimation",
                num_samples=200000,
                status="pending",
                description="TartanDrive dataset for depth estimation",
                source_url="https://github.com/castacks/tartan_drive",
            ),
            "gichd_mines": DatasetMetadata(
                name="gichd_mines",
                path="data/raw/gichd_mines",
                dataset_type="mine_detection",
                num_samples=10000,
                num_classes=8,
                status="pending",
                description="GICHD Mine Action Database",
                source_url="https://www.gichd.org/",
            ),
            "flir_thermal": DatasetMetadata(
                name="flir_thermal",
                path="data/raw/flir_thermal",
                dataset_type="thermal_imaging",
                num_samples=10000,
                status="pending",
                description="FLIR Thermal Dataset for object detection",
                source_url="https://www.flir.com/oem/adas/adas-dataset-form/",
            ),
            "coco": DatasetMetadata(
                name="coco",
                path="data/raw/coco",
                dataset_type="obstacle_detection",
                num_samples=118287,
                num_classes=80,
                status="pending",
                description="COCO 2017 for obstacle detection",
                source_url="https://cocodataset.org/",
            ),
            "synthetic_mines": DatasetMetadata(
                name="synthetic_mines",
                path="data/synthetic/rendered",
                dataset_type="mine_detection",
                num_samples=100000,
                num_classes=8,
                status="pending",
                description="Synthetic mine detection dataset (Blender)",
                source_url="generated",
            ),
        }

        self.save()

    def load(self):
        """Load registry from JSON file."""
        logger.info(f"Loading dataset registry from {self.registry_path}")

        with open(self.registry_path) as f:
            data = json.load(f)

        self.datasets = {
            name: DatasetMetadata.from_dict({**meta, "name": name})
            for name, meta in data.get("datasets", {}).items()
        }

        logger.info(f"Loaded {len(self.datasets)} datasets from registry")

    def save(self):
        """Save registry to JSON file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "datasets": {name: meta.to_dict() for name, meta in self.datasets.items()},
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry to {self.registry_path}")

    def register_dataset(self, metadata: DatasetMetadata):
        """Register a new dataset.

        Args:
            metadata: Dataset metadata
        """
        self.datasets[metadata.name] = metadata
        self.save()
        logger.info(f"Registered dataset: {metadata.name}")

    def update_dataset(self, name: str, **kwargs):
        """Update dataset metadata.

        Args:
            name: Dataset name
            **kwargs: Fields to update
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset not found: {name}")

        dataset = self.datasets[name]
        for key, value in kwargs.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)

        self.save()
        logger.info(f"Updated dataset: {name}")

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name.

        Args:
            name: Dataset name

        Returns:
            Dataset metadata or None if not found
        """
        return self.datasets.get(name)

    def list_datasets(self, dataset_type: Optional[str] = None) -> List[DatasetMetadata]:
        """List all datasets, optionally filtered by type.

        Args:
            dataset_type: Filter by dataset type

        Returns:
            List of dataset metadata
        """
        datasets = list(self.datasets.values())

        if dataset_type:
            datasets = [d for d in datasets if d.dataset_type == dataset_type]

        return datasets

    def get_statistics(self) -> Dict:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_datasets": len(self.datasets),
            "by_type": {},
            "by_status": {},
            "total_samples": 0,
        }

        for dataset in self.datasets.values():
            # Count by type
            dtype = dataset.dataset_type
            stats["by_type"][dtype] = stats["by_type"].get(dtype, 0) + 1

            # Count by status
            status = dataset.status
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Sum samples
            if dataset.status == "downloaded" or dataset.status == "verified":
                stats["total_samples"] += dataset.num_samples

        return stats

    def verify_dataset(self, name: str) -> bool:
        """Verify dataset exists and is accessible.

        Args:
            name: Dataset name

        Returns:
            True if dataset is verified
        """
        dataset = self.get_dataset(name)
        if not dataset:
            logger.warning(f"Dataset not found in registry: {name}")
            return False

        dataset_path = Path(dataset.path)
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return False

        logger.info(f"Dataset verified: {name} at {dataset_path}")
        self.update_dataset(name, status="verified")
        return True
