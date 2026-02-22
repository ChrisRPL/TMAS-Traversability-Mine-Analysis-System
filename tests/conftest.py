"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def tmp_config_dir():
    """Create temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_rgb_image():
    """Create a dummy RGB image tensor.

    Returns:
        torch.Tensor: RGB image (3, 720, 1280)
    """
    return torch.randn(3, 720, 1280)


@pytest.fixture
def sample_rgb_batch():
    """Create a batch of dummy RGB images.

    Returns:
        torch.Tensor: Batch of RGB images (4, 3, 720, 1280)
    """
    return torch.randn(4, 3, 720, 1280)


@pytest.fixture
def sample_thermal_image():
    """Create a dummy thermal image tensor.

    Returns:
        torch.Tensor: Thermal image (1, 640, 512)
    """
    return torch.randn(1, 640, 512)


@pytest.fixture
def sample_thermal_batch():
    """Create a batch of dummy thermal images.

    Returns:
        torch.Tensor: Batch of thermal images (4, 1, 640, 512)
    """
    return torch.randn(4, 1, 640, 512)


@pytest.fixture
def sample_segmentation_mask():
    """Create a dummy segmentation mask.

    Returns:
        torch.Tensor: Segmentation mask (720, 1280) with 14 classes
    """
    return torch.randint(0, 14, (720, 1280), dtype=torch.long)


@pytest.fixture
def sample_detection_annotations():
    """Create dummy detection annotations.

    Returns:
        dict: Detection annotations in COCO format
    """
    return {
        "boxes": torch.tensor(
            [
                [100, 100, 200, 200],
                [300, 300, 400, 400],
                [500, 200, 600, 300],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        "confidence": torch.tensor([0.95, 0.87, 0.92], dtype=torch.float32),
    }


@pytest.fixture
def default_config():
    """Create a default TMAS configuration.

    Returns:
        TMASConfig: Default configuration instance
    """
    from tmas.core import TMASConfig

    return TMASConfig()


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary.

    Returns:
        dict: Configuration dictionary
    """
    return {
        "project_name": "test_project",
        "experiment_name": "test_exp",
        "seed": 123,
        "device": "cpu",
        "model": {
            "backbone": "resnet18",
            "num_classes": 10,
            "latent_dim": 256,
        },
        "data": {
            "batch_size": 8,
            "num_workers": 2,
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
        },
    }
