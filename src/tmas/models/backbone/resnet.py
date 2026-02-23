"""ResNet backbone for thermal imaging in TMAS.

This module implements ResNet-18 optimized for single-channel thermal
images. ResNet-18 is chosen for its balance between accuracy and speed,
critical for real-time mine detection on edge devices (Jetson AGX Orin).

The backbone is pretrained on ImageNet and adapted for thermal input
by averaging RGB weights across the first convolutional layer.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import timm


class ResNetThermalBackbone(nn.Module):
    """ResNet backbone adapted for single-channel thermal imaging.

    Uses ResNet-18 pretrained on ImageNet with first conv layer adapted
    for thermal input (1 channel instead of 3). Provides multi-scale
    features for detection tasks.
    """

    def __init__(
        self,
        depth: int = 18,
        pretrained: bool = True,
        features_only: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1
    ):
        """Initialize ResNet thermal backbone.

        Args:
            depth: ResNet depth (18 or 50)
            pretrained: Load ImageNet pretrained weights
            features_only: Return multi-scale features
            out_indices: Which stages to output (1-4)
            frozen_stages: Number of stages to freeze (-1 = none)
        """
        super().__init__()

        self.depth = depth
        self.features_only = features_only
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Map depth to timm model name
        model_names = {
            18: "resnet18",
            50: "resnet50"
        }

        if depth not in model_names:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Use 18 or 50.")

        model_name = model_names[depth]

        # Create ResNet backbone using timm
        if features_only:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                in_chans=3  # Start with 3 channels for pretrained weights
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=3
            )

        # Get feature dimensions
        if features_only:
            self.feature_info = self.backbone.feature_info
            self.num_features = [info['num_chs'] for info in self.feature_info]
        else:
            self.num_features = self.backbone.num_features

        # Adapt first conv layer for single-channel thermal input
        self._adapt_first_conv()

        # Freeze stages if requested
        self._freeze_stages()

    def _adapt_first_conv(self):
        """Adapt first convolutional layer from RGB (3 channels) to thermal (1 channel).

        Strategy: Average pretrained RGB weights across channels to get
        single-channel weights. This preserves learned features better
        than random initialization.
        """
        # Find first conv layer
        first_conv = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                break

        if first_conv is None:
            raise RuntimeError("Could not find first Conv2d layer in ResNet")

        # Get pretrained weights [out_channels, 3, H, W]
        pretrained_weight = first_conv.weight.data

        # Average across RGB channels to get single-channel weights
        # [out_channels, 3, H, W] -> [out_channels, 1, H, W]
        single_channel_weight = pretrained_weight.mean(dim=1, keepdim=True)

        # Create new conv layer with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )

        # Copy averaged weights
        new_conv.weight.data = single_channel_weight

        # Copy bias if exists
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data

        # Replace first conv layer in backbone
        # Navigate through module hierarchy to replace
        parent = self.backbone
        parts = first_conv_name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    def _freeze_stages(self):
        """Freeze early stages for fine-tuning.

        Freezing early layers preserves low-level features learned from
        ImageNet while allowing later layers to adapt to thermal domain.
        """
        if self.frozen_stages < 0:
            return

        # Freeze backbone stages up to frozen_stages
        # ResNet stages: stem (conv1, bn1) + layer1, layer2, layer3, layer4
        if self.frozen_stages >= 0:
            # Freeze stem
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False

        # Freeze layer1, layer2, etc.
        for i in range(1, min(self.frozen_stages + 1, 5)):
            layer_name = f'layer{i}'
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input thermal images [B, 1, H, W]

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
            List of strides (e.g., [4, 8, 16, 32] for ResNet)
        """
        if self.features_only and hasattr(self, 'feature_info'):
            return [info['reduction'] for info in self.feature_info]
        else:
            return [32]  # Default stride for final features


def create_resnet_thermal_backbone(
    depth: int = 18,
    pretrained: bool = True,
    frozen_stages: int = -1
) -> ResNetThermalBackbone:
    """Create ResNet thermal backbone.

    Args:
        depth: ResNet depth (18 or 50)
        pretrained: Load pretrained ImageNet weights
        frozen_stages: Number of early stages to freeze

    Returns:
        ResNet thermal backbone

    Example:
        >>> backbone = create_resnet_thermal_backbone(depth=18, pretrained=True)
        >>> thermal = torch.randn(2, 1, 640, 512)  # [B, 1, H, W]
        >>> features = backbone(thermal)  # List of 4 feature maps
        >>> print([f.shape for f in features])
    """
    backbone = ResNetThermalBackbone(
        depth=depth,
        pretrained=pretrained,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        frozen_stages=frozen_stages
    )

    return backbone


def main():
    """Test ResNet thermal backbone."""
    print("Testing ResNet thermal backbone...")

    # Create backbone
    backbone = create_resnet_thermal_backbone(
        depth=18,
        pretrained=False,  # Set True to load ImageNet weights
        frozen_stages=-1
    )

    # Test forward pass with thermal input
    batch_size = 2
    thermal = torch.randn(batch_size, 1, 640, 512)  # FLIR Boson 640 resolution

    print(f"\nInput shape: {thermal.shape}")

    with torch.no_grad():
        features = backbone(thermal)

    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        stride = backbone.get_feature_strides()[i]
        print(f"  Stage {i+1}: {feat.shape} (stride={stride})")

    # Verify feature dimensions
    feature_dims = backbone.get_feature_dims()
    print(f"\nFeature dimensions: {feature_dims}")

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with frozen stages
    print("\nTesting with frozen stages...")
    backbone_frozen = create_resnet_thermal_backbone(
        depth=18,
        pretrained=False,
        frozen_stages=2  # Freeze stem + layer1 + layer2
    )

    trainable_frozen = sum(p.numel() for p in backbone_frozen.parameters() if p.requires_grad)
    print(f"Trainable parameters (frozen_stages=2): {trainable_frozen:,}")

    print("\nResNet thermal backbone test successful!")


if __name__ == "__main__":
    main()
