"""Training script for mine detection model.

This script trains the RGB-Thermal mine detection model on synthetic
and real mine datasets with the following optimizations:

- Multi-modal fusion (RGB + Thermal)
- Focal loss for class imbalance
- GIoU loss for small object localization
- Evidential uncertainty estimation
- High recall monitoring (>99.5% target)
- Mixed precision training (FP16)
- Gradient accumulation for large batches
- Wandb logging for experiment tracking

Safety-critical requirements:
- Recall > 99.5% for confirmed mines (AT/AP)
- Low confidence threshold (0.3) for high sensitivity
- Regular checkpoint saving
- Validation on diverse conditions
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tmas.detection.mine_detector import create_mine_detector
from tmas.losses.detection_loss import create_detection_loss
from tmas.data.synthetic import SyntheticMineDataset
from tmas.data.augmentation import get_mine_detection_train, get_mine_detection_val


class MineDetectionTrainer:
    """Trainer for mine detection model."""

    def __init__(self, config: Dict):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config["device"])

        # Create model
        print("Creating mine detection model...")
        self.model = create_mine_detector(
            rgb_backbone=config["model"]["rgb_backbone"],
            thermal_backbone=config["model"]["thermal_backbone"],
            num_classes=config["model"]["num_classes"],
            num_queries=config["model"]["num_queries"],
            pretrained=config["model"]["pretrained"]
        ).to(self.device)

        # Create loss function
        self.criterion = create_detection_loss(
            num_classes=config["model"]["num_classes"],
            focal_alpha=config["loss"]["focal_alpha"],
            focal_gamma=config["loss"]["focal_gamma"]
        ).to(self.device)

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"]
        )

        # Create scheduler
        if config["scheduler"]["type"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config["training"]["epochs"],
                eta_min=config["scheduler"]["eta_min"]
            )
        elif config["scheduler"]["type"] == "onecycle":
            # Will be initialized after dataloaders
            self.scheduler = None
        else:
            self.scheduler = None

        # Mixed precision training
        self.scaler = GradScaler() if config["training"]["mixed_precision"] else None

        # Metrics tracking
        self.best_recall = 0.0
        self.current_epoch = 0

        # Wandb logging
        if config["logging"]["use_wandb"]:
            wandb.init(
                project=config["logging"]["wandb_project"],
                name=config["logging"]["run_name"],
                config=config
            )

    def create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Creating dataloaders...")

        # Training dataset
        train_dataset = SyntheticMineDataset(
            data_root=self.config["data"]["train_root"],
            split="train",
            transform=get_mine_detection_train()
        )

        # Validation dataset
        val_dataset = SyntheticMineDataset(
            data_root=self.config["data"]["val_root"],
            split="val",
            transform=get_mine_detection_val()
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Initialize OneCycle scheduler if needed
        if self.config["scheduler"]["type"] == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config["optimizer"]["lr"],
                epochs=self.config["training"]["epochs"],
                steps_per_epoch=len(train_loader)
            )

        return train_loader, val_loader

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate function for dataloader.

        Args:
            batch: List of samples

        Returns:
            Batched data
        """
        rgb = torch.stack([item["image"] for item in batch])
        thermal = torch.stack([item["thermal"] for item in batch])
        targets = [item["target"] for item in batch]

        return {
            "rgb": rgb,
            "thermal": thermal,
            "targets": targets
        }

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch
            train_loader: Training dataloader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_giou_loss = 0.0

        num_batches = len(train_loader)
        accumulation_steps = self.config["training"].get("accumulation_steps", 1)

        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(self.device)
            thermal = batch["thermal"].to(self.device)
            targets = batch["targets"]

            # Move targets to device
            for target in targets:
                target["labels"] = target["labels"].to(self.device)
                target["boxes"] = target["boxes"].to(self.device)

            # Forward pass
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(rgb, thermal)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict["total_loss"] / accumulation_steps
            else:
                predictions = self.model(rgb, thermal)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict["total_loss"] / accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None and self.config["scheduler"]["type"] == "onecycle":
                    self.scheduler.step()

            # Track losses
            total_loss += loss_dict["total_loss"].item()
            total_cls_loss += loss_dict["classification_loss"].item()
            total_bbox_loss += loss_dict["bbox_loss"].item()
            total_giou_loss += loss_dict["giou_loss"].item()

            # Print progress
            if (batch_idx + 1) % self.config["logging"]["log_interval"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch}/{self.config['training']['epochs']}] "
                      f"Batch [{batch_idx + 1}/{num_batches}] "
                      f"Loss: {avg_loss:.4f}")

        # Step scheduler (if not OneCycle)
        if self.scheduler is not None and self.config["scheduler"]["type"] != "onecycle":
            self.scheduler.step()

        # Average metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_cls_loss": total_cls_loss / num_batches,
            "train_bbox_loss": total_bbox_loss / num_batches,
            "train_giou_loss": total_giou_loss / num_batches
        }

        return metrics

    def validate(self, epoch: int, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model.

        Args:
            epoch: Current epoch
            val_loader: Validation dataloader

        Returns:
            Dictionary of validation metrics including recall
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = len(val_loader)

        # Recall tracking per class
        tp_per_class = torch.zeros(self.config["model"]["num_classes"])
        total_per_class = torch.zeros(self.config["model"]["num_classes"])

        confidence_threshold = self.config["validation"]["confidence_threshold"]

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(self.device)
                thermal = batch["thermal"].to(self.device)
                targets = batch["targets"]

                # Move targets to device
                for target in targets:
                    target["labels"] = target["labels"].to(self.device)
                    target["boxes"] = target["boxes"].to(self.device)

                # Forward pass
                predictions = self.model(rgb, thermal)
                loss_dict = self.criterion(predictions, targets)

                total_loss += loss_dict["total_loss"].item()

                # Compute recall
                detections = self.model.predict(
                    rgb, thermal,
                    confidence_threshold=confidence_threshold
                )

                # Count true positives per class
                for i, target in enumerate(targets):
                    target_labels = target["labels"].cpu()
                    detected_labels = detections[i]["labels"].cpu() if len(detections[i]["labels"]) > 0 else torch.empty(0)

                    # Count ground truth per class
                    for cls in range(self.config["model"]["num_classes"]):
                        total_per_class[cls] += (target_labels == cls).sum()

                        # Count detections
                        if len(detected_labels) > 0:
                            tp_per_class[cls] += min(
                                (target_labels == cls).sum(),
                                (detected_labels == cls).sum()
                            )

        # Compute recall per class
        recall_per_class = tp_per_class / (total_per_class + 1e-7)

        # Average metrics
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_recall_mean": recall_per_class.mean().item(),
            "val_recall_at": recall_per_class[0].item(),  # AT mine
            "val_recall_ap": recall_per_class[1].item(),  # AP mine
            "val_recall_ied": (recall_per_class[2] + recall_per_class[3]).item() / 2  # IED avg
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model
        """
        checkpoint_dir = Path(self.config["checkpoints"]["save_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }

        # Save latest
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with recall: {metrics['val_recall_mean']:.4f}")

        # Save periodic
        if (epoch + 1) % self.config["checkpoints"]["save_interval"] == 0:
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Main training loop."""
        print("Starting training...")

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()

        # Training loop
        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")

            # Train
            train_metrics = self.train_epoch(epoch, train_loader)

            # Validate
            val_metrics = self.validate(epoch, val_loader)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log to wandb
            if self.config["logging"]["use_wandb"]:
                wandb.log(metrics, step=epoch)

            # Print metrics
            print(f"Train Loss: {metrics['train_loss']:.4f}")
            print(f"Val Loss: {metrics['val_loss']:.4f}")
            print(f"Val Recall (mean): {metrics['val_recall_mean']:.4f}")
            print(f"Val Recall (AT): {metrics['val_recall_at']:.4f}")
            print(f"Val Recall (AP): {metrics['val_recall_ap']:.4f}")

            # Save checkpoint
            is_best = metrics["val_recall_mean"] > self.best_recall
            if is_best:
                self.best_recall = metrics["val_recall_mean"]

            self.save_checkpoint(epoch, metrics, is_best=is_best)

        print(f"\nTraining complete! Best recall: {self.best_recall:.4f}")

        if self.config["logging"]["use_wandb"]:
            wandb.finish()


def load_config(config_path: str) -> Dict:
    """Load training configuration.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train mine detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["device"] = args.device if torch.cuda.is_available() else "cpu"

    # Create trainer
    trainer = MineDetectionTrainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
