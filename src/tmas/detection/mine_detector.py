"""Complete mine detection model with RGB-Thermal fusion.

This module integrates all components for end-to-end mine detection:
- EfficientViT-L2 backbone for RGB (shared with terrain segmentation)
- ResNet-18 backbone for thermal
- Cross-attention fusion for multi-modal integration
- RT-DETR head for detection

Architecture:
    RGB Image → EfficientViT-L2 → RGB Features ┐
                                                 ├→ Cross-Attention Fusion → RT-DETR → Detections
    Thermal Image → ResNet-18 → Thermal Features ┘

The model is designed for safety-critical mine detection with:
- High recall priority (>99.5% target)
- Multi-modal fusion for robust detection in various conditions
- Real-time inference capability on Jetson AGX Orin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..models.backbone import get_backbone
from ..fusion.cross_attention import CrossAttentionFusion
from .rtdetr import RTDETRHead


class MineDetectionModel(nn.Module):
    """Complete end-to-end mine detection model.

    Combines RGB and thermal imaging with cross-attention fusion
    for robust mine detection across different environmental conditions.
    """

    def __init__(
        self,
        rgb_backbone: str = "efficientvit_l2",
        thermal_backbone: str = "resnet18",
        num_classes: int = 8,
        num_queries: int = 300,
        fusion_dim: int = 256,
        pretrained_rgb: bool = True,
        pretrained_thermal: bool = True,
        frozen_stages_rgb: int = -1,
        frozen_stages_thermal: int = -1
    ):
        """Initialize mine detection model.

        Args:
            rgb_backbone: RGB backbone name (efficientvit_l2)
            thermal_backbone: Thermal backbone name (resnet18)
            num_classes: Number of detection classes (8 for mines)
            num_queries: Number of object queries in RT-DETR
            fusion_dim: Fusion feature dimension
            pretrained_rgb: Use pretrained RGB backbone
            pretrained_thermal: Use pretrained thermal backbone
            frozen_stages_rgb: Number of frozen stages in RGB backbone
            frozen_stages_thermal: Number of frozen stages in thermal backbone
        """
        super().__init__()

        self.num_classes = num_classes
        self.fusion_dim = fusion_dim

        # RGB backbone (shared with terrain segmentation)
        self.rgb_backbone = get_backbone(
            rgb_backbone,
            pretrained=pretrained_rgb,
            features_only=True,
            frozen_stages=frozen_stages_rgb
        )

        # Thermal backbone
        self.thermal_backbone = get_backbone(
            thermal_backbone,
            pretrained=pretrained_thermal,
            features_only=True,
            frozen_stages=frozen_stages_thermal
        )

        # Get feature dimensions from backbones
        rgb_feature_dims = self.rgb_backbone.get_feature_dims()
        thermal_feature_dims = self.thermal_backbone.get_feature_dims()

        # Use highest resolution features for detection
        # EfficientViT-L2: [64, 128, 256, 512]
        # ResNet-18: [64, 128, 256, 512]
        self.rgb_feature_dim = rgb_feature_dims[-1]  # Use last stage
        self.thermal_feature_dim = thermal_feature_dims[-1]

        # Project RGB and thermal features to fusion dimension
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(self.rgb_feature_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )

        self.thermal_proj = nn.Sequential(
            nn.Conv2d(self.thermal_feature_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )

        # Cross-attention fusion module
        self.fusion = CrossAttentionFusion(
            rgb_dim=fusion_dim,
            thermal_dim=fusion_dim,
            fusion_dim=fusion_dim,
            num_heads=8,
            num_layers=2
        )

        # RT-DETR detection head
        self.detection_head = RTDETRHead(
            in_channels=fusion_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=fusion_dim
        )

    def extract_features(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from RGB and thermal images.

        Args:
            rgb: RGB images [B, 3, H, W]
            thermal: Thermal images [B, 1, H, W]

        Returns:
            Tuple of (rgb_features, thermal_features) [B, C, H', W']
        """
        # Extract multi-scale features
        rgb_features_list = self.rgb_backbone(rgb)
        thermal_features_list = self.thermal_backbone(thermal)

        # Use highest resolution features
        rgb_features = rgb_features_list[-1]  # [B, C_rgb, H', W']
        thermal_features = thermal_features_list[-1]  # [B, C_thermal, H'', W'']

        # Project to fusion dimension
        rgb_features = self.rgb_proj(rgb_features)  # [B, fusion_dim, H', W']
        thermal_features = self.thermal_proj(thermal_features)  # [B, fusion_dim, H'', W'']

        return rgb_features, thermal_features

    def fuse_features(
        self,
        rgb_features: torch.Tensor,
        thermal_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse RGB and thermal features via cross-attention.

        Args:
            rgb_features: RGB features [B, C, H, W]
            thermal_features: Thermal features [B, C, H', W']

        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = rgb_features.shape

        # Flatten spatial dimensions for cross-attention
        rgb_flat = rgb_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        thermal_flat = thermal_features.flatten(2).transpose(1, 2)  # [B, H'*W', C]

        # Apply cross-attention fusion
        fused_flat = self.fusion(rgb_flat, thermal_flat)  # [B, H*W, C]

        # Reshape back to spatial
        fused = fused_flat.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        return fused

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            rgb: RGB images [B, 3, H, W]
            thermal: Thermal images [B, 1, H, W]
            return_features: Return intermediate features

        Returns:
            Dictionary with:
                - pred_logits: Class predictions [B, num_queries, num_classes]
                - pred_boxes: Bounding boxes [B, num_queries, 4]
                - (optional) rgb_features, thermal_features, fused_features
        """
        # Extract features
        rgb_features, thermal_features = self.extract_features(rgb, thermal)

        # Fuse features
        fused_features = self.fuse_features(rgb_features, thermal_features)

        # Detect objects
        detections = self.detection_head(fused_features)

        # Optionally return intermediate features
        if return_features:
            detections["rgb_features"] = rgb_features
            detections["thermal_features"] = thermal_features
            detections["fused_features"] = fused_features

        return detections

    def predict(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        confidence_threshold: float = 0.3,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict mine detections.

        Args:
            rgb: RGB images [B, 3, H, W]
            thermal: Thermal images [B, 1, H, W]
            confidence_threshold: Minimum confidence (low for high recall)
            max_detections: Maximum detections per image

        Returns:
            List of detections per image with boxes, scores, labels
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(rgb, thermal)

        # Filter and format detections
        detections = self.detection_head.predict(
            outputs["fused_features"],
            confidence_threshold=confidence_threshold,
            max_detections=max_detections
        )

        return detections


def create_mine_detector(
    rgb_backbone: str = "efficientvit_l2",
    thermal_backbone: str = "resnet18",
    num_classes: int = 8,
    pretrained: bool = True,
    **kwargs
) -> MineDetectionModel:
    """Create mine detection model.

    Args:
        rgb_backbone: RGB backbone name
        thermal_backbone: Thermal backbone name
        num_classes: Number of classes (8 for mine detection)
        pretrained: Use pretrained backbones
        **kwargs: Additional arguments

    Returns:
        Mine detection model

    Example:
        >>> model = create_mine_detector(pretrained=True)
        >>> rgb = torch.randn(2, 3, 640, 512)
        >>> thermal = torch.randn(2, 1, 640, 512)
        >>> outputs = model(rgb, thermal)
        >>> print(outputs['pred_logits'].shape)  # [2, 300, 8]
    """
    model = MineDetectionModel(
        rgb_backbone=rgb_backbone,
        thermal_backbone=thermal_backbone,
        num_classes=num_classes,
        pretrained_rgb=pretrained,
        pretrained_thermal=pretrained,
        **kwargs
    )

    return model


def main():
    """Test mine detection model."""
    print("Testing mine detection model...")

    # Create model
    model = create_mine_detector(
        rgb_backbone="efficientvit_l2",
        thermal_backbone="resnet18",
        num_classes=8,
        pretrained=False  # Set True to load weights
    )

    # Test forward pass
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 640, 512)  # FLIR Boson resolution
    thermal = torch.randn(batch_size, 1, 640, 512)

    print(f"\nInput shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Thermal: {thermal.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(rgb, thermal, return_features=True)

    print(f"\nOutput shapes:")
    print(f"  Predicted logits: {outputs['pred_logits'].shape}")
    print(f"  Predicted boxes: {outputs['pred_boxes'].shape}")
    print(f"  RGB features: {outputs['rgb_features'].shape}")
    print(f"  Thermal features: {outputs['thermal_features'].shape}")
    print(f"  Fused features: {outputs['fused_features'].shape}")

    # Test prediction
    print("\nTesting prediction...")
    detections = model.predict(
        rgb, thermal,
        confidence_threshold=0.5,
        max_detections=50
    )

    print(f"Batch size: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {len(det['boxes'])} detections")
        if len(det['boxes']) > 0:
            print(f"    Classes: {det['labels'].unique().tolist()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Component breakdown
    rgb_params = sum(p.numel() for p in model.rgb_backbone.parameters())
    thermal_params = sum(p.numel() for p in model.thermal_backbone.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    head_params = sum(p.numel() for p in model.detection_head.parameters())

    print(f"\nParameter breakdown:")
    print(f"  RGB backbone: {rgb_params:,}")
    print(f"  Thermal backbone: {thermal_params:,}")
    print(f"  Fusion module: {fusion_params:,}")
    print(f"  Detection head: {head_params:,}")

    print("\nMine detection model test successful!")


if __name__ == "__main__":
    main()
