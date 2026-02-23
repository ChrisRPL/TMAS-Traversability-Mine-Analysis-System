"""EfficientViT backbone for TMAS terrain segmentation and detection.

EfficientViT is a fast and efficient vision transformer designed for
real-time applications. We use EfficientViT-L2 as the primary backbone
for terrain segmentation and will share it with mine detection.

Paper: EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
       https://arxiv.org/abs/2205.14756
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import timm


class EfficientViTBackbone(nn.Module):
    """EfficientViT backbone wrapper using timm library."""

    def __init__(
        self,
        model_name: str = "efficientvit_l2",
        pretrained: bool = True,
        features_only: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1
    ):
        """Initialize EfficientViT backbone.

        Args:
            model_name: Model variant (efficientvit_l2, efficientvit_l1, etc.)
            pretrained: Load ImageNet pretrained weights
            features_only: Return multi-scale features
            out_indices: Which stages to output (1-4)
            frozen_stages: Number of stages to freeze (-1 = none)
        """
        super().__init__()

        self.model_name = model_name
        self.features_only = features_only
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Create model using timm
        try:
            if features_only:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=out_indices
                )
            else:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained
                )
        except RuntimeError:
            # Fallback if model not available in timm
            print(f"Warning: {model_name} not found in timm, using efficientvit_b3")
            self.backbone = timm.create_model(
                "efficientvit_b3",
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )

        # Get feature dimensions
        if features_only:
            self.feature_info = self.backbone.feature_info
            self.num_features = [info['num_chs'] for info in self.feature_info]
        else:
            self.num_features = self.backbone.num_features

        # Freeze stages if requested
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze early stages for fine-tuning."""
        if self.frozen_stages >= 0:
            # Freeze backbone parameters up to frozen_stages
            # (Implementation depends on timm model structure)
            pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of feature maps at different scales if features_only=True,
            otherwise final features
        """
        if self.features_only:
            features = self.backbone(x)
            return features
        else:
            return self.backbone(x)

    def get_feature_dims(self) -> List[int]:
        """Get feature dimensions for each output stage.

        Returns:
            List of feature dimensions
        """
        if isinstance(self.num_features, list):
            return self.num_features
        else:
            return [self.num_features]

    def get_feature_strides(self) -> List[int]:
        """Get spatial stride for each feature map.

        Returns:
            List of strides (e.g., [4, 8, 16, 32])
        """
        if self.features_only and hasattr(self, 'feature_info'):
            return [info['reduction'] for info in self.feature_info]
        else:
            return [32]  # Default stride for final features


class EfficientViTFeaturePyramid(nn.Module):
    """Feature Pyramid Network on top of EfficientViT backbone.

    Combines multi-scale features for dense prediction tasks.
    """

    def __init__(
        self,
        backbone: EfficientViTBackbone,
        fpn_channels: int = 256
    ):
        """Initialize FPN.

        Args:
            backbone: EfficientViT backbone
            fpn_channels: Number of FPN output channels
        """
        super().__init__()

        self.backbone = backbone
        self.fpn_channels = fpn_channels

        # Get backbone feature dimensions
        in_channels_list = backbone.get_feature_dims()

        # Lateral convolutions (reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # FPN convolutions (smooth upsampled features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of FPN feature maps
        """
        # Get backbone features
        backbone_features = self.backbone(x)

        # Build FPN top-down
        fpn_features = []

        # Start from highest level (lowest resolution)
        prev_features = None
        for i in range(len(backbone_features) - 1, -1, -1):
            # Lateral connection
            lateral = self.lateral_convs[i](backbone_features[i])

            # Top-down pathway
            if prev_features is not None:
                # Upsample previous level to match current size
                h, w = lateral.shape[-2:]
                upsampled = torch.nn.functional.interpolate(
                    prev_features,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                lateral = lateral + upsampled

            # Smooth with 3x3 conv
            fpn_feat = self.fpn_convs[i](lateral)
            fpn_features.insert(0, fpn_feat)
            prev_features = fpn_feat

        return fpn_features


def create_efficientvit_backbone(
    variant: str = "l2",
    pretrained: bool = True,
    with_fpn: bool = False,
    fpn_channels: int = 256
) -> nn.Module:
    """Create EfficientViT backbone with optional FPN.

    Args:
        variant: Model variant (l1/l2/l3/b0-b3)
        pretrained: Load pretrained weights
        with_fpn: Add Feature Pyramid Network
        fpn_channels: FPN output channels

    Returns:
        Backbone model (with FPN if requested)

    Example:
        >>> backbone = create_efficientvit_backbone('l2', pretrained=True)
        >>> features = backbone(images)  # List of feature maps
    """
    model_name = f"efficientvit_{variant}"

    backbone = EfficientViTBackbone(
        model_name=model_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(1, 2, 3, 4)
    )

    if with_fpn:
        backbone = EfficientViTFeaturePyramid(
            backbone=backbone,
            fpn_channels=fpn_channels
        )

    return backbone


def main():
    """Test EfficientViT backbone."""
    print("Testing EfficientViT backbone...")

    # Create backbone
    backbone = create_efficientvit_backbone(
        variant="l2",
        pretrained=False,  # Set True to download weights
        with_fpn=False
    )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 720, 1280)

    with torch.no_grad():
        features = backbone(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Number of feature maps: {len(features)}")

    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape}")

    # Test with FPN
    print("\nTesting with FPN...")
    backbone_fpn = create_efficientvit_backbone(
        variant="l2",
        pretrained=False,
        with_fpn=True,
        fpn_channels=256
    )

    with torch.no_grad():
        fpn_features = backbone_fpn(x)

    print(f"FPN feature maps: {len(fpn_features)}")
    for i, feat in enumerate(fpn_features):
        print(f"  Level {i}: {feat.shape}")

    print("\nBackbone test successful!")


if __name__ == "__main__":
    main()
