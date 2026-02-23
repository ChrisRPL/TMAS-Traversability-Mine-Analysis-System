"""Evaluation metrics for TMAS tasks.

This module provides metrics for terrain segmentation, mine detection,
and obstacle detection evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class SegmentationMetrics:
    """Metrics for semantic segmentation evaluation."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_names: Optional[List[str]] = None
    ):
        """Initialize segmentation metrics.

        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in evaluation
            class_names: Optional class names for reporting
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch predictions.

        Args:
            predictions: Predicted class indices [B, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Flatten
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)

        # Filter valid pixels
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]

        # Update confusion matrix
        for pred, target in zip(valid_preds, valid_targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1

        self.total_samples += valid_mask.sum()

    def compute_iou(self) -> np.ndarray:
        """Compute Intersection over Union for each class.

        Returns:
            Array of IoU values per class
        """
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +  # Ground truth positives
            self.confusion_matrix.sum(axis=0) -  # Predicted positives
            intersection  # Subtract intersection (counted twice)
        )

        # Avoid division by zero
        iou = np.divide(intersection, union, where=(union != 0))
        iou[union == 0] = np.nan

        return iou

    def compute_miou(self) -> float:
        """Compute mean IoU across all classes.

        Returns:
            Mean IoU value
        """
        iou = self.compute_iou()
        # Mean over valid classes (ignore NaN)
        return np.nanmean(iou)

    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy.

        Returns:
            Pixel accuracy value
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()

        if total == 0:
            return 0.0

        return correct / total

    def compute_class_accuracy(self) -> np.ndarray:
        """Compute per-class accuracy.

        Returns:
            Array of accuracy values per class
        """
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)

        # Avoid division by zero
        accuracy = np.divide(class_correct, class_total, where=(class_total != 0))
        accuracy[class_total == 0] = np.nan

        return accuracy

    def compute_mean_accuracy(self) -> float:
        """Compute mean accuracy across classes.

        Returns:
            Mean accuracy value
        """
        class_acc = self.compute_class_accuracy()
        return np.nanmean(class_acc)

    def get_results(self) -> Dict[str, float]:
        """Get all metrics as dictionary.

        Returns:
            Dictionary with all computed metrics
        """
        iou = self.compute_iou()
        class_acc = self.compute_class_accuracy()

        results = {
            "mIoU": self.compute_miou(),
            "pixel_accuracy": self.compute_pixel_accuracy(),
            "mean_accuracy": self.compute_mean_accuracy(),
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(iou):
                results[f"IoU/{class_name}"] = float(iou[i]) if not np.isnan(iou[i]) else 0.0
                results[f"Acc/{class_name}"] = float(class_acc[i]) if not np.isnan(class_acc[i]) else 0.0

        return results

    def __str__(self) -> str:
        """String representation of metrics."""
        results = self.get_results()

        output = []
        output.append(f"mIoU: {results['mIoU']:.4f}")
        output.append(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        output.append(f"Mean Accuracy: {results['mean_accuracy']:.4f}")

        return " | ".join(output)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        """Initialize meter.

        Args:
            name: Metric name
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update meter with new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Track multiple metrics during training."""

    def __init__(self):
        """Initialize metric tracker."""
        self.metrics = defaultdict(AverageMeter)

    def update(self, metrics: Dict[str, float], n: int = 1):
        """Update multiple metrics.

        Args:
            metrics: Dictionary of metric name -> value
            n: Number of samples
        """
        for name, value in metrics.items():
            self.metrics[name].update(value, n)

    def reset(self):
        """Reset all tracked metrics."""
        for meter in self.metrics.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics.

        Returns:
            Dictionary of metric name -> average value
        """
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __str__(self) -> str:
        """String representation."""
        return " | ".join([str(meter) for meter in self.metrics.values()])


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> np.ndarray:
    """Compute confusion matrix for segmentation.

    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Create mask
    valid_mask = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)

    # Initialize confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Fill confusion matrix
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    for pred, target in zip(valid_preds, valid_targets):
        if 0 <= pred < num_classes:
            confusion[target, pred] += 1

    return confusion


def main():
    """Test metrics computation."""
    print("Testing segmentation metrics...")

    num_classes = 19
    batch_size = 2
    height, width = 180, 320

    # Create dummy predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=num_classes)

    # Update metrics
    metrics.update(predictions, targets)

    # Compute results
    results = metrics.get_results()

    print(f"\nmIoU: {results['mIoU']:.4f}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")

    # Test metric tracker
    print("\nTesting metric tracker...")
    tracker = MetricTracker()

    for i in range(5):
        tracker.update({
            "loss": np.random.rand(),
            "mIoU": 0.5 + i * 0.05,
            "accuracy": 0.6 + i * 0.04
        })

    averages = tracker.get_averages()
    print(f"Average loss: {averages['loss']:.4f}")
    print(f"Average mIoU: {averages['mIoU']:.4f}")
    print(f"Average accuracy: {averages['accuracy']:.4f}")

    print("\nMetrics test successful!")


if __name__ == "__main__":
    main()
