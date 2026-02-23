"""Backbone architectures for TMAS models.

This module provides efficient backbone networks for feature extraction
in terrain segmentation, mine detection, and obstacle detection tasks.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

__all__ = ["get_backbone", "list_backbones"]


def get_backbone(
    name: str,
    pretrained: bool = True,
    features_only: bool = True,
    **kwargs
) -> nn.Module:
    """Get backbone model by name.

    Args:
        name: Backbone name (efficientvit_l2, resnet18, etc.)
        pretrained: Load pretrained weights
        features_only: Return multi-scale features
        **kwargs: Additional arguments

    Returns:
        Backbone model

    Example:
        >>> backbone = get_backbone('efficientvit_l2', pretrained=True)
        >>> features = backbone(x)
    """
    if name == "efficientvit_l2":
        from .efficientvit import EfficientViTBackbone
        return EfficientViTBackbone(
            model_name="efficientvit_l2",
            pretrained=pretrained,
            features_only=features_only,
            **kwargs
        )
    elif name == "resnet18":
        from .resnet import ResNetBackbone
        return ResNetBackbone(
            depth=18,
            pretrained=pretrained,
            features_only=features_only,
            **kwargs
        )
    elif name == "resnet50":
        from .resnet import ResNetBackbone
        return ResNetBackbone(
            depth=50,
            pretrained=pretrained,
            features_only=features_only,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")


def list_backbones() -> List[str]:
    """List available backbone models.

    Returns:
        List of backbone names
    """
    return [
        "efficientvit_l2",  # Primary for terrain segmentation and detection
        "resnet18",         # Thermal branch
        "resnet50"          # Alternative heavy backbone
    ]
