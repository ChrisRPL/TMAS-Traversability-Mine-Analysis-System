"""Cross-attention fusion for RGB-Thermal multi-modal detection.

This module implements cross-attention mechanisms for fusing RGB and
thermal features. Cross-attention allows each modality to attend to
complementary information from the other modality, improving detection
of mines that may be more visible in one modality than the other.

RGB advantages:
- High spatial resolution
- Texture and color information
- Surface features (disturbed soil, wires)

Thermal advantages:
- Temperature differences (buried objects)
- Day/night operation
- Penetrates light vegetation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism.

    Implements scaled dot-product cross-attention where queries come
    from one modality and keys/values from another.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """Initialize multi-head cross-attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            qkv_bias: Use bias in QKV projections
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
        """
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, key, value projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Query features [B, N_q, D]
            key: Key features [B, N_k, D]
            value: Value features [B, N_v, D] (N_v == N_k)
            attn_mask: Optional attention mask [B, N_q, N_k]

        Returns:
            Tuple of (output features [B, N_q, D], attention weights [B, H, N_q, N_k])
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # [B, H, N_q, head_dim] @ [B, H, head_dim, N_k] -> [B, H, N_q, N_k]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        # [B, H, N_q, N_k] @ [B, H, N_k, head_dim] -> [B, H, N_q, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, D)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention and feed-forward network.

    Architecture:
        x = x + CrossAttention(LN(x), LN(context))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        """Initialize cross-attention block.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Use bias in QKV projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
        """
        super().__init__()

        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)

        self.cross_attn = MultiHeadCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (queries) [B, N, D]
            context: Context features (keys/values) [B, M, D]

        Returns:
            Output features [B, N, D]
        """
        # Cross-attention
        x_norm = self.norm1_q(x)
        context_norm = self.norm1_kv(context)

        attn_out, _ = self.cross_attn(x_norm, context_norm, context_norm)
        x = x + attn_out

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for RGB-Thermal features.

    Implements bidirectional cross-attention where:
    1. RGB features attend to thermal features
    2. Thermal features attend to RGB features
    3. Fused features are combined with learnable weights

    This allows each modality to selectively integrate complementary
    information from the other modality.
    """

    def __init__(
        self,
        rgb_dim: int,
        thermal_dim: int,
        fusion_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.1
    ):
        """Initialize cross-attention fusion.

        Args:
            rgb_dim: RGB feature dimension
            thermal_dim: Thermal feature dimension
            fusion_dim: Fusion feature dimension (output)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            mlp_ratio: MLP hidden dimension ratio
            drop: Dropout rate
        """
        super().__init__()

        self.rgb_dim = rgb_dim
        self.thermal_dim = thermal_dim
        self.fusion_dim = fusion_dim

        # Project RGB and thermal to common dimension
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.thermal_proj = nn.Linear(thermal_dim, fusion_dim)

        # RGB -> Thermal cross-attention layers
        self.rgb_to_thermal = nn.ModuleList([
            CrossAttentionBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop
            )
            for _ in range(num_layers)
        ])

        # Thermal -> RGB cross-attention layers
        self.thermal_to_rgb = nn.ModuleList([
            CrossAttentionBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop
            )
            for _ in range(num_layers)
        ])

        # Learnable fusion weights
        self.fusion_weight_rgb = nn.Parameter(torch.ones(1))
        self.fusion_weight_thermal = nn.Parameter(torch.ones(1))

        # Final fusion layer
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(
        self,
        rgb_features: torch.Tensor,
        thermal_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb_features: RGB features [B, N_rgb, D_rgb]
            thermal_features: Thermal features [B, N_thermal, D_thermal]

        Returns:
            Fused features [B, N_rgb, fusion_dim]

        Note:
            Output spatial resolution matches RGB features. If thermal
            features have different spatial size, they are used as
            context via cross-attention (no spatial alignment needed).
        """
        B = rgb_features.shape[0]

        # Project to common dimension
        rgb_feat = self.rgb_proj(rgb_features)  # [B, N_rgb, fusion_dim]
        thermal_feat = self.thermal_proj(thermal_features)  # [B, N_thermal, fusion_dim]

        # RGB features attend to thermal context
        rgb_enhanced = rgb_feat
        for layer in self.rgb_to_thermal:
            rgb_enhanced = layer(rgb_enhanced, thermal_feat)

        # Thermal features attend to RGB context
        thermal_enhanced = thermal_feat
        for layer in self.thermal_to_rgb:
            thermal_enhanced = layer(thermal_enhanced, rgb_feat)

        # If thermal has different spatial size, upsample to match RGB
        if thermal_enhanced.shape[1] != rgb_enhanced.shape[1]:
            # Reshape to spatial grid for interpolation
            # Assume features are from HxW spatial grid
            N_rgb = rgb_enhanced.shape[1]
            N_thermal = thermal_enhanced.shape[1]

            H_rgb = W_rgb = int(math.sqrt(N_rgb))
            H_thermal = W_thermal = int(math.sqrt(N_thermal))

            # Reshape and interpolate
            thermal_spatial = thermal_enhanced.transpose(1, 2).reshape(B, self.fusion_dim, H_thermal, W_thermal)
            thermal_upsampled = F.interpolate(
                thermal_spatial,
                size=(H_rgb, W_rgb),
                mode='bilinear',
                align_corners=False
            )
            thermal_enhanced = thermal_upsampled.reshape(B, self.fusion_dim, -1).transpose(1, 2)

        # Weighted fusion
        fused = (
            self.fusion_weight_rgb * rgb_enhanced +
            self.fusion_weight_thermal * thermal_enhanced
        )

        # Final normalization and projection
        fused = self.fusion_norm(fused)
        fused = self.fusion_proj(fused)

        return fused


def create_cross_attention_fusion(
    rgb_dim: int = 256,
    thermal_dim: int = 256,
    fusion_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 2
) -> CrossAttentionFusion:
    """Create cross-attention fusion module.

    Args:
        rgb_dim: RGB feature dimension
        thermal_dim: Thermal feature dimension
        fusion_dim: Output fusion dimension
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers

    Returns:
        CrossAttentionFusion module

    Example:
        >>> fusion = create_cross_attention_fusion(256, 256, 256)
        >>> rgb_feat = torch.randn(2, 1600, 256)  # [B, H*W, D]
        >>> thermal_feat = torch.randn(2, 400, 256)  # [B, H'*W', D]
        >>> fused = fusion(rgb_feat, thermal_feat)  # [B, 1600, 256]
    """
    return CrossAttentionFusion(
        rgb_dim=rgb_dim,
        thermal_dim=thermal_dim,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )


def main():
    """Test cross-attention fusion module."""
    print("Testing cross-attention fusion...")

    # Create fusion module
    fusion = create_cross_attention_fusion(
        rgb_dim=256,
        thermal_dim=128,
        fusion_dim=256,
        num_heads=8,
        num_layers=2
    )

    # Test with features from different spatial resolutions
    # RGB: 40x40 = 1600 tokens
    # Thermal: 20x20 = 400 tokens
    batch_size = 2
    rgb_features = torch.randn(batch_size, 1600, 256)
    thermal_features = torch.randn(batch_size, 400, 128)

    print(f"\nRGB features shape: {rgb_features.shape}")
    print(f"Thermal features shape: {thermal_features.shape}")

    # Forward pass
    with torch.no_grad():
        fused = fusion(rgb_features, thermal_features)

    print(f"Fused features shape: {fused.shape}")

    # Verify output spatial resolution matches RGB
    assert fused.shape[:2] == rgb_features.shape[:2], "Output should match RGB spatial resolution"

    # Check learnable weights
    print(f"\nFusion weights:")
    print(f"  RGB weight: {fusion.fusion_weight_rgb.item():.4f}")
    print(f"  Thermal weight: {fusion.fusion_weight_thermal.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nCross-attention fusion test successful!")


if __name__ == "__main__":
    main()
