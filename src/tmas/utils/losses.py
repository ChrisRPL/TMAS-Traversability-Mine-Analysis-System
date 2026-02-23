"""Loss functions for TMAS training.

This module provides loss functions optimized for safety-critical
terrain segmentation and mine detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation.

    Dice loss is effective for imbalanced datasets and focuses on
    spatial overlap between prediction and ground truth.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = "mean"
    ):
        """Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method (mean/sum/none)
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]

        Returns:
            Dice loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)

        # Get number of classes
        num_classes = logits.shape[1]

        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        # Apply valid mask
        if self.ignore_index >= 0:
            probs = probs * valid_mask.unsqueeze(1).float()
            targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1).float()

        # Compute Dice coefficient per class
        dims = (0, 2, 3)  # Batch, Height, Width

        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Dice loss is 1 - dice_score
        dice_loss = 1.0 - dice_score

        # Reduce
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.
    Useful for mine detection where mines are rare.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean"
    ):
        """Initialize Focal loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Focal loss.

        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]

        Returns:
            Focal loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Get probabilities
        probs = torch.exp(log_probs)

        # Create valid mask
        valid_mask = (targets != self.ignore_index)

        # Gather log probs for target classes
        targets_clamped = targets.clamp(0, logits.shape[1] - 1)
        log_probs_target = log_probs.gather(
            1,
            targets_clamped.unsqueeze(1)
        ).squeeze(1)

        probs_target = probs.gather(
            1,
            targets_clamped.unsqueeze(1)
        ).squeeze(1)

        # Compute focal term
        focal_weight = (1 - probs_target) ** self.gamma

        # Compute loss
        loss = -self.alpha * focal_weight * log_probs_target

        # Apply valid mask
        if self.ignore_index >= 0:
            loss = loss * valid_mask.float()

        # Reduce
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1.0)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedSegmentationLoss(nn.Module):
    """Combined loss for semantic segmentation.

    Combines CrossEntropy and Dice losses for better convergence
    and handling of class imbalance.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize combined loss.

        Args:
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            ignore_index: Index to ignore
            class_weights: Per-class weights for CE loss
        """
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean"
        )

        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]

        Returns:
            Combined loss value
        """
        ce_loss_val = self.ce_loss(logits, targets)
        dice_loss_val = self.dice_loss(logits, targets)

        total_loss = (
            self.ce_weight * ce_loss_val +
            self.dice_weight * dice_loss_val
        )

        return total_loss


class SegmentationLossWithAux(nn.Module):
    """Segmentation loss with auxiliary supervision.

    Adds auxiliary loss from intermediate features to improve
    gradient flow and feature learning.
    """

    def __init__(
        self,
        main_loss: nn.Module,
        aux_weight: float = 0.4
    ):
        """Initialize loss with auxiliary supervision.

        Args:
            main_loss: Main loss function
            aux_weight: Weight for auxiliary loss
        """
        super().__init__()

        self.main_loss = main_loss
        self.aux_weight = aux_weight

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with auxiliary supervision.

        Args:
            outputs: Dictionary with 'out' and optionally 'aux' keys
            targets: Ground truth labels [B, H, W]

        Returns:
            Total loss value
        """
        # Main loss
        main_loss_val = self.main_loss(outputs["out"], targets)

        # Auxiliary loss if available
        if "aux" in outputs:
            aux_loss_val = self.main_loss(outputs["aux"], targets)
            total_loss = main_loss_val + self.aux_weight * aux_loss_val
        else:
            total_loss = main_loss_val

        return total_loss


def create_segmentation_loss(
    loss_type: str = "combined",
    ignore_index: int = 255,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Create segmentation loss function.

    Args:
        loss_type: Type of loss (ce/dice/combined/focal)
        ignore_index: Index to ignore in loss
        class_weights: Per-class weights
        **kwargs: Additional arguments

    Returns:
        Loss function

    Example:
        >>> loss_fn = create_segmentation_loss('combined', ignore_index=255)
        >>> loss = loss_fn(logits, targets)
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean"
        )
    elif loss_type == "dice":
        return DiceLoss(ignore_index=ignore_index)
    elif loss_type == "combined":
        return CombinedSegmentationLoss(
            ce_weight=kwargs.get("ce_weight", 1.0),
            dice_weight=kwargs.get("dice_weight", 1.0),
            ignore_index=ignore_index,
            class_weights=class_weights
        )
    elif loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            ignore_index=ignore_index
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    """Test loss functions."""
    print("Testing loss functions...")

    batch_size = 2
    num_classes = 19
    height, width = 180, 320

    # Create dummy data
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test CrossEntropy loss
    ce_loss = create_segmentation_loss("ce", ignore_index=255)
    ce_val = ce_loss(logits, targets)
    print(f"CrossEntropy loss: {ce_val.item():.4f}")

    # Test Dice loss
    dice_loss = create_segmentation_loss("dice", ignore_index=255)
    dice_val = dice_loss(logits, targets)
    print(f"Dice loss: {dice_val.item():.4f}")

    # Test combined loss
    combined_loss = create_segmentation_loss("combined", ignore_index=255)
    combined_val = combined_loss(logits, targets)
    print(f"Combined loss: {combined_val.item():.4f}")

    # Test focal loss
    focal_loss = create_segmentation_loss("focal", ignore_index=255)
    focal_val = focal_loss(logits, targets)
    print(f"Focal loss: {focal_val.item():.4f}")

    # Test with auxiliary loss
    outputs = {
        "out": logits,
        "aux": torch.randn(batch_size, num_classes, height, width)
    }

    aux_loss = SegmentationLossWithAux(
        main_loss=combined_loss,
        aux_weight=0.4
    )
    aux_val = aux_loss(outputs, targets)
    print(f"Loss with auxiliary: {aux_val.item():.4f}")

    print("\nLoss tests successful!")


if __name__ == "__main__":
    main()
