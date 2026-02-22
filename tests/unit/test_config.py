"""Unit tests for configuration management."""

import pytest
import yaml
from pathlib import Path

from tmas.core import TMASConfig, ModelConfig, DataConfig, TrainingConfig


class TestTMASConfig:
    """Tests for TMASConfig class."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = TMASConfig()
        assert config.project_name == "tmas"
        assert config.seed == 42
        assert config.device == "cuda"

    def test_config_from_dict(self, sample_config_dict):
        """Test creating configuration from dictionary."""
        config = TMASConfig.from_dict(sample_config_dict)
        assert config.project_name == "test_project"
        assert config.experiment_name == "test_exp"
        assert config.seed == 123
        assert config.model.num_classes == 10

    def test_config_to_dict(self, default_config):
        """Test converting configuration to dictionary."""
        config_dict = default_config.to_dict()
        assert isinstance(config_dict, dict)
        assert "project_name" in config_dict
        assert "model" in config_dict
        assert "data" in config_dict
        assert "training" in config_dict

    def test_config_from_yaml(self, tmp_config_dir, sample_config_dict):
        """Test loading configuration from YAML file."""
        yaml_path = tmp_config_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = TMASConfig.from_yaml(str(yaml_path))
        assert config.project_name == "test_project"
        assert config.model.num_classes == 10

    def test_config_save_yaml(self, tmp_config_dir, default_config):
        """Test saving configuration to YAML file."""
        yaml_path = tmp_config_dir / "saved_config.yaml"
        default_config.save_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Load and verify
        with open(yaml_path) as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config["project_name"] == "tmas"

    def test_config_validation_valid(self, default_config):
        """Test configuration validation with valid config."""
        errors = default_config.validate()
        assert len(errors) == 0

    def test_config_validation_invalid_batch_size(self):
        """Test configuration validation with invalid batch size."""
        config = TMASConfig()
        config.data.batch_size = -1
        errors = config.validate()
        assert len(errors) > 0
        assert any("batch_size" in error for error in errors)

    def test_config_validation_invalid_learning_rate(self):
        """Test configuration validation with invalid learning rate."""
        config = TMASConfig()
        config.training.learning_rate = 0
        errors = config.validate()
        assert len(errors) > 0
        assert any("learning_rate" in error for error in errors)

    def test_config_from_nonexistent_yaml(self):
        """Test loading from non-existent YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            TMASConfig.from_yaml("nonexistent.yaml")


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.backbone == "efficientvit_l2"
        assert config.num_classes == 14
        assert config.latent_dim == 512
        assert config.pretrained is True

    def test_model_config_custom(self):
        """Test custom model configuration."""
        config = ModelConfig(
            backbone="resnet50", num_classes=10, latent_dim=256, pretrained=False
        )
        assert config.backbone == "resnet50"
        assert config.num_classes == 10
        assert config.latent_dim == 256
        assert config.pretrained is False


class TestDataConfig:
    """Tests for DataConfig class."""

    def test_default_data_config(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.dataset_name == "rellis3d"
        assert config.batch_size == 16
        assert config.num_workers == 4
        assert config.augmentation is True

    def test_data_config_custom(self):
        """Test custom data configuration."""
        config = DataConfig(
            dataset_name="custom", batch_size=32, num_workers=8, augmentation=False
        )
        assert config.dataset_name == "custom"
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.augmentation is False


class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.epochs == 50
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adamw"
        assert config.mixed_precision is True

    def test_training_config_custom(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            epochs=100, learning_rate=5e-5, optimizer="adam", mixed_precision=False
        )
        assert config.epochs == 100
        assert config.learning_rate == 5e-5
        assert config.optimizer == "adam"
        assert config.mixed_precision is False
