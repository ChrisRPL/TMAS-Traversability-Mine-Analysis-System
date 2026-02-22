"""Unit tests for logging utilities."""

import tempfile
from pathlib import Path

import pytest

from tmas.utils import ExperimentLogger, CheckpointManager, get_logger


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"


class TestExperimentLogger:
    """Tests for ExperimentLogger class."""

    def test_experiment_logger_init_tensorboard_only(self):
        """Test initializing experiment logger with TensorBoard only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                project_name="test",
                experiment_name="test_exp",
                log_dir=tmpdir,
                use_wandb=False,
                use_tensorboard=True,
            )
            assert logger.project_name == "test"
            assert logger.experiment_name == "test_exp"
            assert logger.wandb_run is None
            # TensorBoard writer should be initialized (if torch available)

    def test_experiment_logger_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                project_name="test",
                log_dir=tmpdir,
                use_wandb=False,
                use_tensorboard=True,
            )
            metrics = {"loss": 0.5, "accuracy": 0.95}
            # Should not raise any errors
            logger.log_metrics(metrics, step=1)

    def test_experiment_logger_log_config(self):
        """Test logging configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                project_name="test",
                log_dir=tmpdir,
                use_wandb=False,
                use_tensorboard=True,
            )
            config = {"learning_rate": 0.001, "batch_size": 32}
            # Should not raise any errors
            logger.log_config(config)

    def test_experiment_logger_finish(self):
        """Test finishing experiment logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                project_name="test",
                log_dir=tmpdir,
                use_wandb=False,
                use_tensorboard=True,
            )
            # Should not raise any errors
            logger.finish()


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_checkpoint_manager_init(self):
        """Test initializing checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, max_checkpoints=3)
            assert manager.save_dir == Path(tmpdir)
            assert manager.max_checkpoints == 3
            assert len(manager.checkpoints) == 0

    def test_checkpoint_manager_save(self):
        """Test saving checkpoint."""
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, max_checkpoints=3)
            state_dict = {"model": torch.randn(10, 10), "epoch": 1}

            checkpoint_path = manager.save_checkpoint(
                state_dict=state_dict, metric_value=0.95, epoch=1, is_best=True
            )

            assert checkpoint_path.exists()
            assert (manager.save_dir / "best_model.pth").exists()
            assert (manager.save_dir / "last_model.pth").exists()

    def test_checkpoint_manager_cleanup(self):
        """Test checkpoint cleanup."""
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, max_checkpoints=2)

            # Save multiple checkpoints
            for i in range(5):
                state_dict = {"model": torch.randn(5, 5), "epoch": i}
                manager.save_checkpoint(
                    state_dict=state_dict, metric_value=0.8 + i * 0.02, epoch=i
                )

            # Should only keep 2 regular checkpoints + best + last
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pth"))
            # At most max_checkpoints regular checkpoints should remain
            assert len(manager.checkpoints) <= 2
