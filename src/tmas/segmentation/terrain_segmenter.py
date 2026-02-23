"""Terrain segmentation model for TMAS.

This module implements the terrain segmentation network using EfficientViT
backbone with a segmentation head for 19 RELLIS-3D terrain classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..models.backbone import get_backbone


class SegmentationHead(nn.Module):
    """Segmentation decoder head for multi-scale features."""

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """Initialize segmentation head.

        Args:
            in_channels: List of input channel dimensions from backbone
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for feature fusion
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Lateral convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels
        ])

        # Feature fusion module
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim * len(in_channels),
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Final classifier
        self.classifier = nn.Conv2d(
            hidden_dim,
            num_classes,
            kernel_size=1
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from backbone
            target_size: Target output size (H, W)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Process each feature level
        processed_features = []

        for i, feat in enumerate(features):
            # Apply lateral convolution
            lateral_feat = self.lateral_convs[i](feat)
            processed_features.append(lateral_feat)

        # Determine target size for upsampling
        if target_size is None:
            # Use size of highest resolution feature
            target_size = processed_features[0].shape[-2:]

        # Upsample all features to same size
        upsampled_features = []
        for feat in processed_features:
            if feat.shape[-2:] != target_size:
                upsampled = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                upsampled = feat
            upsampled_features.append(upsampled)

        # Concatenate and fuse
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)

        # Final classification
        logits = self.classifier(fused)

        return logits


class TerrainSegmenter(nn.Module):
    """Complete terrain segmentation model.

    Combines EfficientViT backbone with segmentation head for
    pixel-wise terrain classification.
    """

    def __init__(
        self,
        backbone: str = "efficientvit_l2",
        num_classes: int = 19,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        aux_loss: bool = False
    ):
        """Initialize terrain segmenter.

        Args:
            backbone: Backbone architecture name
            num_classes: Number of terrain classes
            pretrained: Use pretrained backbone
            hidden_dim: Hidden dimension in segmentation head
            dropout: Dropout rate
            aux_loss: Use auxiliary loss from intermediate features
        """
        super().__init__()

        self.num_classes = num_classes
        self.aux_loss = aux_loss

        # Create backbone
        self.backbone = get_backbone(
            backbone,
            pretrained=pretrained,
            features_only=True
        )

        # Get feature dimensions
        feature_dims = self.backbone.get_feature_dims()

        # Create segmentation head
        self.seg_head = SegmentationHead(
            in_channels=feature_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Auxiliary head for intermediate supervision (optional)
        if aux_loss:
            # Use middle-level features
            mid_idx = len(feature_dims) // 2
            self.aux_head = nn.Sequential(
                nn.Conv2d(feature_dims[mid_idx], hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(hidden_dim, num_classes, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            return_aux: Return auxiliary predictions

        Returns:
            Dictionary with 'out' (main predictions) and optionally 'aux'
        """
        input_size = x.shape[-2:]

        # Extract features
        features = self.backbone(x)

        # Main segmentation head
        logits = self.seg_head(features, target_size=input_size)

        output = {"out": logits}

        # Auxiliary head if enabled
        if self.aux_loss and return_aux and hasattr(self, 'aux_head'):
            mid_idx = len(features) // 2
            aux_logits = self.aux_head(features[mid_idx])

            # Upsample to input size
            aux_logits = F.interpolate(
                aux_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

            output["aux"] = aux_logits

        return output

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """Predict terrain classes.

        Args:
            x: Input images [B, 3, H, W]
            return_probs: Return probabilities instead of class indices

        Returns:
            Class predictions [B, H, W] or probabilities [B, C, H, W]
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x, return_aux=False)
            logits = output["out"]

            if return_probs:
                probs = F.softmax(logits, dim=1)
                return probs
            else:
                preds = torch.argmax(logits, dim=1)
                return preds

    def get_traversability_map(
        self,
        x: torch.Tensor,
        cost_mapping: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """Generate traversability cost map from predictions.

        Args:
            x: Input images [B, 3, H, W]
            cost_mapping: Dictionary mapping class ID to cost (0.0-1.0)

        Returns:
            Cost map [B, H, W] with values 0.0 (easy) to 1.0 (impossible)
        """
        # Get class predictions
        preds = self.predict(x, return_probs=False)

        # Default cost mapping from RELLIS-3D classes
        if cost_mapping is None:
            from ..data.rellis3d import RELLIS3DDataset
            cost_mapping = RELLIS3DDataset.TRAVERSABILITY_COST

        # Create cost map
        cost_map = torch.zeros_like(preds, dtype=torch.float32)

        for class_id, cost in cost_mapping.items():
            mask = (preds == class_id)
            cost_map[mask] = cost

        return cost_map


def create_terrain_segmenter(
    backbone: str = "efficientvit_l2",
    num_classes: int = 19,
    pretrained: bool = True,
    **kwargs
) -> TerrainSegmenter:
    """Create terrain segmentation model.

    Args:
        backbone: Backbone architecture
        num_classes: Number of classes (19 for RELLIS-3D)
        pretrained: Use pretrained backbone
        **kwargs: Additional arguments

    Returns:
        TerrainSegmenter model

    Example:
        >>> model = create_terrain_segmenter('efficientvit_l2', num_classes=19)
        >>> output = model(images)
        >>> predictions = output['out']  # [B, 19, H, W]
    """
    model = TerrainSegmenter(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )

    return model


def main():
    """Test terrain segmentation model."""
    print("Testing terrain segmentation model...")

    # Create model
    model = create_terrain_segmenter(
        backbone="efficientvit_l2",
        num_classes=19,
        pretrained=False,
        aux_loss=True
    )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 720, 1280)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x, return_aux=True)

    print(f"Main output shape: {output['out'].shape}")
    if 'aux' in output:
        print(f"Aux output shape: {output['aux'].shape}")

    # Test prediction
    preds = model.predict(x)
    print(f"Predictions shape: {preds.shape}")
    print(f"Unique classes: {torch.unique(preds).tolist()}")

    # Test cost map generation
    cost_map = model.get_traversability_map(x)
    print(f"Cost map shape: {cost_map.shape}")
    print(f"Cost range: [{cost_map.min():.2f}, {cost_map.max():.2f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel test successful!")


if __name__ == "__main__":
    main()
