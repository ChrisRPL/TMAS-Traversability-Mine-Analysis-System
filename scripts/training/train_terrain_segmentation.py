"""Training script for terrain segmentation model.

This script trains the TerrainSegmenter model on RELLIS-3D dataset
with proper evaluation metrics, checkpointing, and logging.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tmas.segmentation import TerrainSegmenter
from src.tmas.data.rellis3d import RELLIS3DDataset
from src.tmas.data.augmentation import get_augmentation
from src.tmas.utils.losses import create_segmentation_loss, SegmentationLossWithAux
from src.tmas.utils.metrics import SegmentationMetrics, MetricTracker
from src.tmas.utils.logging import ExperimentLogger, CheckpointManager


class Trainer:
    """Trainer for terrain segmentation."""

    def __init__(self, config: dict):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["device"])

        # Initialize W&B if enabled
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "tmas-terrain-segmentation"),
                name=config.get("exp_name", "terrain_seg_baseline"),
                config=config
            )

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Create loss, optimizer, scheduler
        self.criterion = self._create_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=config["num_classes"],
            ignore_index=config.get("ignore_index", 255)
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=config["num_classes"],
            ignore_index=config.get("ignore_index", 255)
        )
        self.metric_tracker = MetricTracker()

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config["checkpoint_dir"],
            max_checkpoints=config.get("max_checkpoints", 5)
        )

        # Best metric tracking
        self.best_miou = 0.0
        self.current_epoch = 0

    def _create_model(self) -> nn.Module:
        """Create terrain segmentation model."""
        model = TerrainSegmenter(
            backbone=self.config["backbone"],
            num_classes=self.config["num_classes"],
            pretrained=self.config.get("pretrained", True),
            hidden_dim=self.config.get("hidden_dim", 256),
            dropout=self.config.get("dropout", 0.1),
            aux_loss=self.config.get("aux_loss", True)
        )
        return model

    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        # Training augmentation
        train_aug = get_augmentation(
            "terrain_seg",
            "train",
            image_size=(self.config["image_height"], self.config["image_width"])
        )

        # Validation augmentation
        val_aug = get_augmentation(
            "terrain_seg",
            "val",
            image_size=(self.config["image_height"], self.config["image_width"])
        )

        # Training dataset
        train_dataset = RELLIS3DDataset(
            data_dir=self.config["data_dir"],
            split="train",
            transform=train_aug
        )

        # Validation dataset
        val_dataset = RELLIS3DDataset(
            data_dir=self.config["data_dir"],
            split="val",
            transform=val_aug
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("val_batch_size", self.config["batch_size"]),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True
        )

        return train_loader, val_loader

    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        main_loss = create_segmentation_loss(
            loss_type=self.config.get("loss_type", "combined"),
            ignore_index=self.config.get("ignore_index", 255),
            ce_weight=self.config.get("ce_weight", 1.0),
            dice_weight=self.config.get("dice_weight", 1.0)
        )

        # Wrap with auxiliary loss if enabled
        if self.config.get("aux_loss", True):
            return SegmentationLossWithAux(
                main_loss=main_loss,
                aux_weight=self.config.get("aux_weight", 0.4)
            )
        else:
            return main_loss

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4)
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=self.config.get("min_lr", 1e-6)
            )
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config["learning_rate"],
                epochs=self.config["epochs"],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            return None

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        self.metric_tracker.reset()

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(images, return_aux=self.config.get("aux_loss", True))

            # Compute loss
            if isinstance(outputs, dict):
                loss = self.criterion(outputs, masks)
                logits = outputs["out"]
            else:
                loss = self.criterion(outputs, masks)
                logits = outputs

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )

            self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                self.train_metrics.update(preds, masks)

            # Track loss
            self.metric_tracker.update({"loss": loss.item()}, n=images.size(0))

            # Log progress
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                avg_loss = self.metric_tracker.metrics["loss"].avg
                print(
                    f"Epoch [{epoch}/{self.config['epochs']}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

        # Epoch metrics
        epoch_time = time.time() - start_time
        train_results = self.train_metrics.get_results()
        avg_loss = self.metric_tracker.metrics["loss"].avg

        print(f"\nEpoch {epoch} Training Results:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {train_results['mIoU']:.4f}")
        print(f"  Pixel Acc: {train_results['pixel_accuracy']:.4f}")

        # Log to W&B
        if self.use_wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/mIoU": train_results["mIoU"],
                "train/pixel_accuracy": train_results["pixel_accuracy"],
                "train/epoch_time": epoch_time,
                "epoch": epoch
            })

        return train_results

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()
        val_loss_tracker = MetricTracker()

        start_time = time.time()

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(images, return_aux=False)

            # Compute loss
            if isinstance(outputs, dict):
                logits = outputs["out"]
            else:
                logits = outputs

            # Update metrics
            preds = torch.argmax(logits, dim=1)
            self.val_metrics.update(preds, masks)

        # Validation results
        val_time = time.time() - start_time
        val_results = self.val_metrics.get_results()

        print(f"\nEpoch {epoch} Validation Results:")
        print(f"  Time: {val_time:.2f}s")
        print(f"  mIoU: {val_results['mIoU']:.4f}")
        print(f"  Pixel Acc: {val_results['pixel_accuracy']:.4f}")

        # Log to W&B
        if self.use_wandb:
            log_dict = {
                "val/mIoU": val_results["mIoU"],
                "val/pixel_accuracy": val_results["pixel_accuracy"],
                "val/val_time": val_time,
                "epoch": epoch
            }

            # Log per-class IoU
            for key, value in val_results.items():
                if key.startswith("IoU/"):
                    log_dict[f"val/{key}"] = value

            wandb.log(log_dict)

        return val_results

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint,
            epoch=epoch,
            metric_value=metrics.get("mIoU", 0.0)
        )

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config["checkpoint_dir"]) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")

    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print()

        for epoch in range(1, self.config["epochs"] + 1):
            self.current_epoch = epoch

            # Training
            train_results = self.train_epoch(epoch)

            # Validation
            val_results = self.validate(epoch)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    pass  # OneCycle updates per batch
                else:
                    self.scheduler.step()

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Learning Rate: {current_lr:.6f}")

                if self.use_wandb:
                    wandb.log({"learning_rate": current_lr, "epoch": epoch})

            # Check if best model
            current_miou = val_results["mIoU"]
            is_best = current_miou > self.best_miou

            if is_best:
                self.best_miou = current_miou
                print(f"  New best mIoU: {self.best_miou:.4f}")

            # Save checkpoint
            if epoch % self.config.get("save_interval", 5) == 0 or is_best:
                self.save_checkpoint(epoch, val_results, is_best=is_best)

            print("-" * 80)

        print("\nTraining complete!")
        print(f"Best mIoU: {self.best_miou:.4f}")

        if self.use_wandb:
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train terrain segmentation model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to RELLIS-3D dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints/terrain_seg")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--exp-name", type=str, default="terrain_seg_baseline")

    args = parser.parse_args()

    # Load config or create default
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data_dir": args.data_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "device": args.device,
            "use_wandb": args.use_wandb,
            "exp_name": args.exp_name,

            # Model config
            "backbone": "efficientvit_l2",
            "num_classes": 19,
            "pretrained": True,
            "hidden_dim": 256,
            "dropout": 0.1,
            "aux_loss": True,

            # Training config
            "image_height": 720,
            "image_width": 1280,
            "ignore_index": 255,
            "loss_type": "combined",
            "ce_weight": 1.0,
            "dice_weight": 1.0,
            "aux_weight": 0.4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "scheduler": "cosine",
            "min_lr": 1e-6,

            # Logging
            "log_interval": 10,
            "save_interval": 5,
            "num_workers": 4,
            "max_checkpoints": 5,

            # W&B
            "wandb_project": "tmas-terrain-segmentation"
        }

    # Create checkpoint directory
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = Path(config["checkpoint_dir"]) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
