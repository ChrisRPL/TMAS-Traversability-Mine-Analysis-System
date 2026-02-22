"""Configuration management for TMAS."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str = "efficientvit_l2"
    num_classes: int = 14
    latent_dim: int = 512
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset_name: str = "rellis3d"
    data_dir: str = "data/raw"
    batch_size: int = 16
    num_workers: int = 4
    image_size: tuple = (720, 1280)
    augmentation: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class TMASConfig:
    """Main TMAS configuration."""

    project_name: str = "tmas"
    experiment_name: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: Dict[str, Any] = field(
        default_factory=lambda: {"use_wandb": False, "use_tensorboard": True}
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TMASConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TMASConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_file) as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TMASConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TMASConfig instance
        """
        # Extract nested configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        # Create main config
        return cls(
            project_name=config_dict.get("project_name", "tmas"),
            experiment_name=config_dict.get("experiment_name"),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
            model=model_config,
            data=data_config,
            training=training_config,
            logging=config_dict.get(
                "logging", {"use_wandb": False, "use_tensorboard": True}
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        return {
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "model": {
                "backbone": self.model.backbone,
                "num_classes": self.model.num_classes,
                "latent_dim": self.model.latent_dim,
                "pretrained": self.model.pretrained,
                "checkpoint_path": self.model.checkpoint_path,
            },
            "data": {
                "dataset_name": self.data.dataset_name,
                "data_dir": self.data.data_dir,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "image_size": self.data.image_size,
                "augmentation": self.data.augmentation,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "optimizer": self.training.optimizer,
                "scheduler": self.training.scheduler,
                "warmup_epochs": self.training.warmup_epochs,
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "mixed_precision": self.training.mixed_precision,
            },
            "logging": self.logging,
        }

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_file = Path(yaml_path)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_file, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate model config
        if self.model.num_classes < 1:
            errors.append("model.num_classes must be >= 1")
        if self.model.latent_dim < 1:
            errors.append("model.latent_dim must be >= 1")

        # Validate data config
        if self.data.batch_size < 1:
            errors.append("data.batch_size must be >= 1")
        if self.data.num_workers < 0:
            errors.append("data.num_workers must be >= 0")

        # Validate training config
        if self.training.epochs < 1:
            errors.append("training.epochs must be >= 1")
        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be > 0")

        return errors
