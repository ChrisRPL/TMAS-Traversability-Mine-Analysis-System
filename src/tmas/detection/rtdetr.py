"""RT-DETR detection head for real-time mine detection.

RT-DETR (Real-Time Detection Transformer) is a DETR variant optimized
for real-time performance. Key advantages:

1. NMS-free detection (DETR eliminates non-maximum suppression)
2. End-to-end optimization
3. Better small object detection (critical for mines)
4. Efficient hybrid encoder design
5. Real-time inference on Jetson AGX Orin

Paper: DETRs Beat YOLOs on Real-time Object Detection
       https://arxiv.org/abs/2304.08069
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class MLP(nn.Module):
    """Multi-layer perceptron for DETR."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
        """
        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for RT-DETR."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """Initialize decoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: Target queries [B, N_queries, D]
            memory: Encoder memory [B, N_memory, D]
            tgt_mask: Target mask
            memory_mask: Memory mask

        Returns:
            Updated queries [B, N_queries, D]
        """
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class RTDETRHead(nn.Module):
    """RT-DETR detection head for mine detection.

    Implements DETR-style detection with learnable object queries
    that attend to image features via cross-attention.

    Mine classes (8 total):
    1. AT mine (confirmed) - Anti-tank mines
    2. AP mine (confirmed) - Anti-personnel mines
    3. IED roadside - Improvised explosive devices near roads
    4. IED buried - Buried IEDs
    5. UXO - Unexploded ordnance
    6. Wire/trigger - Visible tripwires or triggers
    7. Soil anomaly - Suspicious ground disturbances
    8. False positive - Background objects (for training)
    """

    # Mine class definitions
    CLASSES = [
        "at_mine",
        "ap_mine",
        "ied_roadside",
        "ied_buried",
        "uxo",
        "wire_trigger",
        "soil_anomaly",
        "false_positive"
    ]

    NUM_CLASSES = 8

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 8,
        num_queries: int = 300,
        hidden_dim: int = 256,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """Initialize RT-DETR head.

        Args:
            in_channels: Input feature channels from backbone
            num_classes: Number of detection classes (8 for mines)
            num_queries: Number of object queries (detections per image)
            hidden_dim: Hidden dimension for transformer
            num_decoder_layers: Number of decoder layers
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Input projection (backbone features -> hidden_dim)
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])

        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Positional encoding for spatial features
        self.pos_embed = None  # Will be created dynamically

    def _create_positional_encoding(
        self,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create 2D positional encoding.

        Args:
            h: Height
            w: Width
            device: Device

        Returns:
            Positional encoding [1, h*w, hidden_dim]
        """
        # Create sinusoidal positional encoding
        y_embed = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)

        # Normalize to [0, 1]
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Flatten
        y_embed = y_embed.flatten()
        x_embed = x_embed.flatten()

        # Create positional encoding
        dim_t = torch.arange(self.hidden_dim // 2, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t

        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)

        pos = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)

        return pos

    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features from backbone [B, C, H, W]

        Returns:
            Dictionary with:
                - pred_logits: Class predictions [B, num_queries, num_classes]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
                  (normalized [cx, cy, w, h] format)
        """
        B, C, H, W = features.shape

        # Project features to hidden dimension
        features = self.input_proj(features)  # [B, hidden_dim, H, W]

        # Flatten spatial dimensions
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # Create positional encoding
        pos_embed = self._create_positional_encoding(H, W, features.device)  # [1, H*W, hidden_dim]

        # Add positional encoding to features
        memory = features_flat + pos_embed  # [B, H*W, hidden_dim]

        # Initialize object queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]

        # Pass through decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # Predict classes and boxes
        pred_logits = self.class_embed(queries)  # [B, num_queries, num_classes]
        pred_boxes = self.bbox_embed(queries).sigmoid()  # [B, num_queries, 4]

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }

    def predict(
        self,
        features: torch.Tensor,
        confidence_threshold: float = 0.3,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict detections with confidence filtering.

        Args:
            features: Input features [B, C, H, W]
            confidence_threshold: Minimum confidence score
            max_detections: Maximum detections per image

        Returns:
            List of detections per image, each containing:
                - boxes: [N, 4] in [cx, cy, w, h] normalized format
                - scores: [N] confidence scores
                - labels: [N] class indices
        """
        outputs = self.forward(features)

        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]

        # Get confidence scores
        probs = F.softmax(pred_logits, dim=-1)  # [B, num_queries, num_classes]
        scores, labels = probs.max(dim=-1)  # [B, num_queries]

        # Filter by confidence and max detections
        batch_detections = []
        for i in range(pred_logits.shape[0]):
            # Filter low confidence
            mask = scores[i] >= confidence_threshold
            filtered_boxes = pred_boxes[i][mask]
            filtered_scores = scores[i][mask]
            filtered_labels = labels[i][mask]

            # Sort by confidence and keep top-k
            if len(filtered_scores) > max_detections:
                top_scores, top_indices = torch.topk(filtered_scores, max_detections)
                filtered_boxes = filtered_boxes[top_indices]
                filtered_scores = top_scores
                filtered_labels = filtered_labels[top_indices]

            batch_detections.append({
                "boxes": filtered_boxes,
                "scores": filtered_scores,
                "labels": filtered_labels
            })

        return batch_detections


def create_rtdetr_head(
    in_channels: int = 256,
    num_classes: int = 8,
    num_queries: int = 300,
    **kwargs
) -> RTDETRHead:
    """Create RT-DETR detection head.

    Args:
        in_channels: Input feature channels
        num_classes: Number of classes (8 for mine detection)
        num_queries: Number of object queries
        **kwargs: Additional arguments

    Returns:
        RT-DETR head

    Example:
        >>> head = create_rtdetr_head(in_channels=256, num_classes=8)
        >>> features = torch.randn(2, 256, 40, 40)
        >>> outputs = head(features)
        >>> print(outputs['pred_logits'].shape)  # [2, 300, 8]
        >>> print(outputs['pred_boxes'].shape)   # [2, 300, 4]
    """
    return RTDETRHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_queries=num_queries,
        **kwargs
    )


def main():
    """Test RT-DETR detection head."""
    print("Testing RT-DETR detection head...")

    # Create detection head
    head = create_rtdetr_head(
        in_channels=256,
        num_classes=8,
        num_queries=300,
        num_decoder_layers=6
    )

    # Test forward pass
    batch_size = 2
    features = torch.randn(batch_size, 256, 40, 40)

    print(f"\nInput features shape: {features.shape}")

    with torch.no_grad():
        outputs = head(features)

    print(f"Predicted logits shape: {outputs['pred_logits'].shape}")
    print(f"Predicted boxes shape: {outputs['pred_boxes'].shape}")

    # Test prediction with confidence filtering
    print("\nTesting prediction with confidence filtering...")
    detections = head.predict(features, confidence_threshold=0.5, max_detections=50)

    print(f"Batch size: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {len(det['boxes'])} detections")
        if len(det['boxes']) > 0:
            print(f"    Score range: [{det['scores'].min():.3f}, {det['scores'].max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print mine classes
    print(f"\nMine detection classes ({head.NUM_CLASSES}):")
    for i, cls in enumerate(head.CLASSES):
        print(f"  {i}: {cls}")

    print("\nRT-DETR head test successful!")


if __name__ == "__main__":
    main()
