"""Logging utilities for experiment tracking and monitoring."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Unified logging interface for W&B and TensorBoard."""

    def __init__(
        self,
        project_name: str = "tmas",
        experiment_name: Optional[str] = None,
        log_dir: str = "runs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        """Initialize experiment logger.

        Args:
            project_name: Project name for tracking
            experiment_name: Specific experiment name
            log_dir: Directory for logs
            use_wandb: Enable Weights & Biases logging
            use_tensorboard: Enable TensorBoard logging
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self.wandb_run = None
        self.tb_writer = None

        # Initialize W&B if requested
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    dir=str(self.log_dir),
                )
                self.logger.info(f"W&B initialized: {wandb.run.url}")
            except ImportError:
                self.logger.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")

        # Initialize TensorBoard if requested
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / project_name / (experiment_name or "default")
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"TensorBoard initialized: {tb_dir}")
            except ImportError:
                self.logger.warning(
                    "tensorboard not installed, skipping TensorBoard logging"
                )

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
        """
        # Log to W&B
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to W&B: {e}")

        # Log to TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                try:
                    self.tb_writer.add_scalar(key, value, step)
                except Exception as e:
                    self.logger.warning(f"Failed to log {key} to TensorBoard: {e}")

    def log_config(self, config: Dict[str, Any]):
        """Log configuration/hyperparameters.

        Args:
            config: Configuration dictionary
        """
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.config.update(config)
            except Exception as e:
                self.logger.warning(f"Failed to log config to W&B: {e}")

    def finish(self):
        """Finish logging and cleanup."""
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
                self.logger.info("W&B run finished")
            except Exception:
                pass

        if self.tb_writer is not None:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed")


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = get_logger(__name__)
        self.checkpoints = []

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        metric_value: float,
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        Args:
            state_dict: Model state dictionary
            metric_value: Metric value for this checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        import torch

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(state_dict, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Track checkpoint
        self.checkpoints.append((checkpoint_path, metric_value, epoch))

        # Save best checkpoint separately
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(state_dict, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Save last checkpoint
        last_path = self.save_dir / "last_model.pth"
        torch.save(state_dict, last_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric value (assuming higher is better)
            self.checkpoints.sort(key=lambda x: x[1], reverse=True)

            # Remove worst checkpoints
            for checkpoint_path, _, _ in self.checkpoints[self.max_checkpoints :]:
                if checkpoint_path.exists() and "best" not in checkpoint_path.name:
                    checkpoint_path.unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")

            self.checkpoints = self.checkpoints[: self.max_checkpoints]
