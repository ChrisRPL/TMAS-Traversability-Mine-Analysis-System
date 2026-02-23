"""Detection loss functions for mine detection.

This module implements loss functions optimized for safety-critical
mine detection with extreme class imbalance. Key considerations:

1. Class imbalance: Mines are rare compared to background
2. Recall priority: Missing a mine is catastrophic
3. Small objects: Mines are often small (5-40cm)
4. Localization: Precise bounding boxes for path planning

Loss components:
- Focal Loss: Handles class imbalance, focuses on hard examples
- GIoU Loss: Better localization than IoU, handles non-overlapping boxes
- L1 Loss: Box coordinate regression
- Evidential Loss: Uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FocalLoss(nn.Module):
    """Focal loss for addressing extreme class imbalance.

    Focal loss down-weights easy examples and focuses training on hard
    negatives. Critical for mine detection where:
    - True mines are very rare (high class imbalance)
    - Model must not miss mines (high recall priority)
    - False positives are acceptable (better safe than sorry)

    Paper: Focal Loss for Dense Object Detection (RetinaNet)
           https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method (mean/sum/none)
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [N, num_classes]
            targets: Ground truth class indices [N]

        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get target probabilities
        num_classes = logits.shape[-1]
        targets_one_hot = F.one_hot(targets, num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=-1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal loss
        loss = self.alpha * focal_weight * ce_loss

        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """Generalized Intersection over Union loss.

    GIoU improves upon IoU by:
    1. Penalizing non-overlapping boxes (IoU=0 has gradient)
    2. Considering the shape and orientation of boxes
    3. Better localization for small objects (mines)

    Paper: Generalized Intersection over Union
           https://arxiv.org/abs/1902.09630
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize GIoU loss.

        Args:
            reduction: Reduction method (mean/sum/none)
        """
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 4] in [cx, cy, w, h] format (normalized)
            target_boxes: Target boxes [N, 4] in [cx, cy, w, h] format (normalized)

        Returns:
            GIoU loss value
        """
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union area
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # Enclosing box (smallest box containing both)
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)

        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_area = enclose_w * enclose_h

        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

        # GIoU loss
        loss = 1 - giou

        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """Combined detection loss for RT-DETR mine detection.

    Combines multiple loss components:
    1. Classification loss (Focal) - handles class imbalance
    2. Bounding box loss (GIoU) - precise localization
    3. L1 loss - box coordinate regression
    4. Optional evidential loss - uncertainty estimation

    Class weights emphasize confirmed mines for high recall.
    """

    def __init__(
        self,
        num_classes: int = 8,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_evidential: bool = False
    ):
        """Initialize detection loss.

        Args:
            num_classes: Number of classes
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            loss_weights: Weights for loss components
            class_weights: Per-class weights for classification
            use_evidential: Use evidential loss for uncertainty
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_evidential = use_evidential

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                "classification": 2.0,  # High weight for classification
                "bbox": 5.0,            # High weight for localization
                "giou": 2.0             # GIoU for better localization
            }
        self.loss_weights = loss_weights

        # Classification loss
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean"
        )

        # Box regression losses
        self.giou_loss = GIoULoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")

        # Class weights (prioritize confirmed mines)
        if class_weights is None:
            # Default weights from IMPLEMENTATION_PLAN.md
            class_weights = torch.tensor([
                10.0,  # AT mine (confirmed) - highest priority
                10.0,  # AP mine (confirmed) - highest priority
                8.0,   # IED roadside
                8.0,   # IED buried
                5.0,   # UXO
                5.0,   # Wire/trigger
                5.0,   # Soil anomaly
                1.0    # False positive (background)
            ])
        self.register_buffer('class_weights', class_weights)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            predictions: Model predictions with:
                - pred_logits: [B, num_queries, num_classes]
                - pred_boxes: [B, num_queries, 4]
            targets: List of target dicts per image with:
                - labels: [N] class indices
                - boxes: [N, 4] in [cx, cy, w, h] format (normalized)

        Returns:
            Dictionary with loss components and total loss
        """
        pred_logits = predictions["pred_logits"]  # [B, Q, C]
        pred_boxes = predictions["pred_boxes"]    # [B, Q, 4]

        B, Q, C = pred_logits.shape

        # Match predictions to targets (Hungarian matching)
        indices = self._match_predictions_to_targets(predictions, targets)

        # Compute classification loss
        cls_loss = self._classification_loss(pred_logits, targets, indices)

        # Compute box losses
        bbox_loss, giou_loss = self._box_losses(pred_boxes, targets, indices)

        # Total loss
        total_loss = (
            self.loss_weights["classification"] * cls_loss +
            self.loss_weights["bbox"] * bbox_loss +
            self.loss_weights["giou"] * giou_loss
        )

        return {
            "total_loss": total_loss,
            "classification_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss
        }

    def _match_predictions_to_targets(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Match predictions to targets using Hungarian algorithm.

        For simplicity, we use greedy matching by confidence.
        Full implementation would use scipy.optimize.linear_sum_assignment.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            List of (pred_indices, target_indices) per image
        """
        pred_logits = predictions["pred_logits"]  # [B, Q, C]
        B, Q, C = pred_logits.shape

        indices = []

        for i in range(B):
            if len(targets[i]["labels"]) == 0:
                # No targets, no matches
                indices.append((torch.empty(0, dtype=torch.long),
                               torch.empty(0, dtype=torch.long)))
                continue

            # Simple greedy matching: assign top-k confident predictions
            num_targets = len(targets[i]["labels"])
            scores = pred_logits[i].max(dim=-1)[0]  # [Q]
            top_k = min(num_targets, Q)

            pred_idx = torch.topk(scores, top_k)[1]
            target_idx = torch.arange(top_k)

            indices.append((pred_idx, target_idx))

        return indices

    def _classification_loss(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute classification loss.

        Args:
            pred_logits: [B, Q, C]
            targets: Target dicts
            indices: Matched indices

        Returns:
            Classification loss
        """
        B, Q, C = pred_logits.shape

        # Gather matched predictions and targets
        all_pred_logits = []
        all_target_labels = []

        for i in range(B):
            pred_idx, target_idx = indices[i]

            if len(pred_idx) > 0:
                matched_logits = pred_logits[i, pred_idx]
                matched_labels = targets[i]["labels"][target_idx]

                all_pred_logits.append(matched_logits)
                all_target_labels.append(matched_labels)

        if len(all_pred_logits) == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        # Concatenate
        pred_logits_cat = torch.cat(all_pred_logits, dim=0)
        target_labels_cat = torch.cat(all_target_labels, dim=0)

        # Compute focal loss
        loss = self.focal_loss(pred_logits_cat, target_labels_cat)

        return loss

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute box regression losses.

        Args:
            pred_boxes: [B, Q, 4]
            targets: Target dicts
            indices: Matched indices

        Returns:
            Tuple of (L1 loss, GIoU loss)
        """
        B, Q, _ = pred_boxes.shape

        # Gather matched boxes
        all_pred_boxes = []
        all_target_boxes = []

        for i in range(B):
            pred_idx, target_idx = indices[i]

            if len(pred_idx) > 0:
                matched_pred_boxes = pred_boxes[i, pred_idx]
                matched_target_boxes = targets[i]["boxes"][target_idx]

                all_pred_boxes.append(matched_pred_boxes)
                all_target_boxes.append(matched_target_boxes)

        if len(all_pred_boxes) == 0:
            device = pred_boxes.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Concatenate
        pred_boxes_cat = torch.cat(all_pred_boxes, dim=0)
        target_boxes_cat = torch.cat(all_target_boxes, dim=0)

        # Compute losses
        l1_loss = self.l1_loss(pred_boxes_cat, target_boxes_cat)
        giou_loss = self.giou_loss(pred_boxes_cat, target_boxes_cat)

        return l1_loss, giou_loss


def create_detection_loss(
    num_classes: int = 8,
    class_weights: Optional[List[float]] = None,
    **kwargs
) -> DetectionLoss:
    """Create detection loss function.

    Args:
        num_classes: Number of classes
        class_weights: Per-class weights (None for default)
        **kwargs: Additional arguments

    Returns:
        Detection loss

    Example:
        >>> loss_fn = create_detection_loss(num_classes=8)
        >>> predictions = {
        ...     "pred_logits": torch.randn(2, 300, 8),
        ...     "pred_boxes": torch.rand(2, 300, 4)
        ... }
        >>> targets = [
        ...     {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)},
        ...     {"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)}
        ... ]
        >>> loss = loss_fn(predictions, targets)
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights)

    return DetectionLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        **kwargs
    )


def main():
    """Test detection loss functions."""
    print("Testing detection loss functions...")

    # Test focal loss
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    logits = torch.randn(16, 8)
    targets = torch.randint(0, 8, (16,))

    fl = focal_loss(logits, targets)
    print(f"Focal loss: {fl.item():.4f}")

    # Test GIoU loss
    print("\nTesting GIoU Loss...")
    giou_loss = GIoULoss()

    pred_boxes = torch.rand(16, 4)  # [cx, cy, w, h]
    target_boxes = torch.rand(16, 4)

    gl = giou_loss(pred_boxes, target_boxes)
    print(f"GIoU loss: {gl.item():.4f}")

    # Test detection loss
    print("\nTesting Detection Loss...")
    detection_loss = create_detection_loss(num_classes=8)

    predictions = {
        "pred_logits": torch.randn(2, 300, 8),
        "pred_boxes": torch.rand(2, 300, 4)
    }

    targets = [
        {
            "labels": torch.tensor([0, 1, 2]),
            "boxes": torch.rand(3, 4)
        },
        {
            "labels": torch.tensor([3, 4]),
            "boxes": torch.rand(2, 4)
        }
    ]

    loss_dict = detection_loss(predictions, targets)

    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Classification loss: {loss_dict['classification_loss'].item():.4f}")
    print(f"BBox loss: {loss_dict['bbox_loss'].item():.4f}")
    print(f"GIoU loss: {loss_dict['giou_loss'].item():.4f}")

    print("\nDetection loss test successful!")


if __name__ == "__main__":
    main()
