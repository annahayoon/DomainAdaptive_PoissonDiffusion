"""
Comprehensive tests for training system.

Tests the deterministic training loop, loss functions, metrics,
and all training-related functionality.
"""

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from poisson_training import (
    DeterministicTrainer,
    DiffusionLoss,
    ImageQualityMetrics,
    PhysicsMetrics,
    PoissonGaussianLoss,
    TrainingConfig,
    TrainingMetrics,
    create_trainer,
    get_scheduler,
    load_checkpoint,
    save_checkpoint,
    set_deterministic_mode,
    train_model,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = batch.get("electrons", batch.get("noisy"))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return {"prediction": x, "denoised": x}


class TestTrainingConfig:
    """Test training configuration."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = TrainingConfig()

        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 100
        assert config.seed == 42
        assert config.deterministic == True

    def test_config_serialization(self):
        """Test config serialization and deserialization."""
        config = TrainingConfig(batch_size=32, learning_rate=2e-4, num_epochs=50)

        # Convert to dict and back
        config_dict = config.to_dict()
        config_restored = TrainingConfig.from_dict(config_dict)

        assert config_restored.batch_size == 32
        assert config_restored.learning_rate == 2e-4
        assert config_restored.num_epochs == 50

    def test_config_save_load(self):
        """Test config save and load."""
        config = TrainingConfig(batch_size=64, learning_rate=5e-4)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            config.save(tmp.name)

            loaded_config = TrainingConfig.load(tmp.name)

            assert loaded_config.batch_size == 64
            assert loaded_config.learning_rate == 5e-4

        Path(tmp.name).unlink(missing_ok=True)


class TestPoissonGaussianLoss:
    """Test Poisson-Gaussian loss function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loss_fn = PoissonGaussianLoss()
        self.batch_size = 4
        self.height = 32
        self.width = 32

    def test_loss_initialization(self):
        """Test loss function initialization."""
        assert self.loss_fn.weights["reconstruction"] == 1.0
        assert self.loss_fn.weights["consistency"] == 0.1
        assert self.loss_fn.eps == 1e-8

    def test_poisson_gaussian_nll(self):
        """Test Poisson-Gaussian negative log-likelihood."""
        prediction = torch.rand(self.batch_size, 1, self.height, self.width)
        target = torch.rand(self.batch_size, 1, self.height, self.width) * 1000
        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)

        nll = self.loss_fn.poisson_gaussian_nll(
            prediction, target, scale, background, read_noise
        )

        assert isinstance(nll, torch.Tensor)
        assert nll.numel() == 1
        assert nll.item() >= 0

    def test_consistency_loss(self):
        """Test consistency loss."""
        prediction = torch.rand(self.batch_size, 1, self.height, self.width)
        target = torch.rand(self.batch_size, 1, self.height, self.width) * 1000
        scale = torch.tensor(1000.0)

        consistency = self.loss_fn.consistency_loss(prediction, target, scale)

        assert isinstance(consistency, torch.Tensor)
        assert consistency.numel() == 1
        assert consistency.item() >= 0

    def test_forward_pass(self):
        """Test forward pass of loss function."""
        prediction = torch.rand(self.batch_size, 1, self.height, self.width)

        outputs = {"prediction": prediction}
        batch = {
            "electrons": torch.rand(self.batch_size, 1, self.height, self.width) * 1000,
            "scale": torch.tensor(1000.0),
            "background": torch.tensor(100.0),
            "read_noise": torch.tensor(5.0),
        }

        losses = self.loss_fn(outputs, batch)

        assert "reconstruction" in losses
        assert "consistency" in losses
        assert all(isinstance(loss, torch.Tensor) for loss in losses.values())
        assert all(loss.numel() == 1 for loss in losses.values())

    def test_loss_with_different_shapes(self):
        """Test loss with different tensor shapes."""
        # Test with different batch sizes
        for batch_size in [1, 2, 8]:
            prediction = torch.rand(batch_size, 1, self.height, self.width)

            outputs = {"prediction": prediction}
            batch = {
                "electrons": torch.rand(batch_size, 1, self.height, self.width) * 1000,
                "scale": torch.tensor(1000.0),
                "background": torch.tensor(100.0),
                "read_noise": torch.tensor(5.0),
            }

            losses = self.loss_fn(outputs, batch)

            assert all(isinstance(loss, torch.Tensor) for loss in losses.values())


class TestDiffusionLoss:
    """Test diffusion loss function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loss_fn = DiffusionLoss()
        self.batch_size = 4
        self.height = 32
        self.width = 32

    def test_loss_initialization(self):
        """Test loss function initialization."""
        assert self.loss_fn.loss_type == "mse"
        assert self.loss_fn.parameterization == "noise"
        assert self.loss_fn.weighting == "uniform"

    def test_get_target_noise_param(self):
        """Test target computation for noise parameterization."""
        x0 = torch.rand(self.batch_size, 1, self.height, self.width)
        noise = torch.randn(self.batch_size, 1, self.height, self.width)
        timesteps = torch.randint(0, 1000, (self.batch_size,))
        alphas_cumprod = torch.rand(1000)

        target = self.loss_fn.get_target(x0, noise, timesteps, alphas_cumprod)

        assert target.shape == noise.shape
        assert torch.allclose(target, noise)

    def test_get_target_v_param(self):
        """Test target computation for v-parameterization."""
        loss_fn = DiffusionLoss(parameterization="v")

        x0 = torch.rand(self.batch_size, 1, self.height, self.width)
        noise = torch.randn(self.batch_size, 1, self.height, self.width)
        timesteps = torch.randint(0, 1000, (self.batch_size,))
        alphas_cumprod = torch.rand(1000)

        target = loss_fn.get_target(x0, noise, timesteps, alphas_cumprod)

        assert target.shape == noise.shape

    def test_forward_pass(self):
        """Test forward pass of diffusion loss."""
        prediction = torch.randn(self.batch_size, 1, self.height, self.width)

        outputs = {"prediction": prediction}
        batch = {
            "x0": torch.rand(self.batch_size, 1, self.height, self.width),
            "noise": torch.randn(self.batch_size, 1, self.height, self.width),
            "timesteps": torch.randint(0, 1000, (self.batch_size,)),
            "alphas_cumprod": torch.rand(1000),
        }

        losses = self.loss_fn(outputs, batch)

        assert "diffusion_loss" in losses
        assert isinstance(losses["diffusion_loss"], torch.Tensor)
        assert losses["diffusion_loss"].numel() == 1


class TestTrainingMetrics:
    """Test training metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = TrainingMetrics(window_size=10)

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        assert self.metrics.window_size == 10
        assert len(self.metrics.metrics) == 0
        assert len(self.metrics.moving_averages) == 0

    def test_update_metrics(self):
        """Test updating metrics."""
        values = {"loss": 1.5, "accuracy": 0.8}

        self.metrics.update(values, phase="train")

        assert "train" in self.metrics.metrics
        assert "loss" in self.metrics.metrics["train"]
        assert "accuracy" in self.metrics.metrics["train"]
        assert self.metrics.metrics["train"]["loss"][-1] == 1.5
        assert self.metrics.metrics["train"]["accuracy"][-1] == 0.8

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        # Add multiple values
        for i in range(5):
            values = {"loss": 1.0 + i * 0.1, "accuracy": 0.7 + i * 0.05}
            self.metrics.update(values, phase="train")

        current = self.metrics.get_current_metrics("train")

        assert "loss" in current
        assert "accuracy" in current
        assert isinstance(current["loss"], float)
        assert isinstance(current["accuracy"], float)

    def test_get_metric_history(self):
        """Test getting metric history."""
        # Add values
        for i in range(3):
            self.metrics.update({"loss": i}, phase="train")

        history = self.metrics.get_metric_history("loss", "train")

        assert len(history) == 3
        assert history == [0, 1, 2]

    def test_metrics_summary(self):
        """Test metrics summary."""
        # Add some values
        for i in range(5):
            values = {
                "total_loss": 1.0 - i * 0.1,
                "reconstruction_loss": 0.8 - i * 0.08,
            }
            self.metrics.update(values, phase="train")

        summary = self.metrics.get_summary("train")

        assert "current_metrics" in summary
        assert "latest_metrics" in summary
        assert "total_steps" in summary
        assert "total_loss_stats" in summary
        assert "reconstruction_loss_stats" in summary


class TestImageQualityMetrics:
    """Test image quality metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ImageQualityMetrics()
        self.prediction = torch.rand(2, 1, 32, 32)
        self.target = torch.rand(2, 1, 32, 32)

    def test_psnr_computation(self):
        """Test PSNR computation."""
        psnr = self.metrics.psnr(self.prediction, self.target)

        assert isinstance(psnr, torch.Tensor)
        assert psnr.numel() == 1
        assert psnr.item() > 0

    def test_psnr_identical_images(self):
        """Test PSNR with identical images."""
        psnr = self.metrics.psnr(self.prediction, self.prediction)

        assert psnr.item() == float("inf")

    def test_ssim_computation(self):
        """Test SSIM computation."""
        ssim = self.metrics.ssim(self.prediction, self.target)

        assert isinstance(ssim, torch.Tensor)
        assert ssim.numel() == 1
        assert -1 <= ssim.item() <= 1

    def test_ssim_identical_images(self):
        """Test SSIM with identical images."""
        ssim = self.metrics.ssim(self.prediction, self.prediction)

        assert ssim.item() == pytest.approx(1.0, abs=1e-6)

    def test_mae_computation(self):
        """Test MAE computation."""
        mae = self.metrics.mae(self.prediction, self.target)

        assert isinstance(mae, torch.Tensor)
        assert mae.numel() == 1
        assert mae.item() >= 0

    def test_mse_computation(self):
        """Test MSE computation."""
        mse = self.metrics.mse(self.prediction, self.target)

        assert isinstance(mse, torch.Tensor)
        assert mse.numel() == 1
        assert mse.item() >= 0


class TestPhysicsMetrics:
    """Test physics-aware metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = PhysicsMetrics()
        self.prediction = torch.rand(2, 1, 32, 32)
        self.target = torch.rand(2, 1, 32, 32) * 1000
        self.scale = torch.tensor(1000.0)
        self.background = torch.tensor(100.0)
        self.read_noise = torch.tensor(5.0)

    def test_chi_squared_consistency(self):
        """Test χ² consistency metric."""
        chi2 = self.metrics.chi_squared_consistency(
            self.prediction, self.target, self.scale, self.background, self.read_noise
        )

        assert isinstance(chi2, torch.Tensor)
        assert chi2.numel() == 1
        assert chi2.item() > 0

    def test_photon_noise_ratio(self):
        """Test photon noise ratio."""
        pnr = self.metrics.photon_noise_ratio(self.prediction, self.target, self.scale)

        assert isinstance(pnr, torch.Tensor)
        assert pnr.numel() == 1
        assert pnr.item() > 0

    def test_energy_conservation(self):
        """Test energy conservation metric."""
        ec = self.metrics.energy_conservation(self.prediction, self.target, self.scale)

        assert isinstance(ec, torch.Tensor)
        assert ec.numel() == 1
        assert ec.item() > 0


class TestSchedulers:
    """Test learning rate schedulers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        scheduler = get_scheduler(
            self.optimizer, scheduler_type="cosine", num_epochs=100, warmup_epochs=5
        )

        assert scheduler is not None

        # Test a few steps
        initial_lr = self.optimizer.param_groups[0]["lr"]
        scheduler.step()

        # LR should change after step
        assert self.optimizer.param_groups[0]["lr"] != initial_lr

    def test_linear_scheduler(self):
        """Test linear scheduler."""
        scheduler = get_scheduler(
            self.optimizer, scheduler_type="linear", num_epochs=50, warmup_epochs=0
        )

        assert scheduler is not None

    def test_plateau_scheduler(self):
        """Test plateau scheduler."""
        scheduler = get_scheduler(self.optimizer, scheduler_type="plateau", patience=5)

        assert scheduler is not None

        # Test step with metric
        scheduler.step(1.0)
        scheduler.step(0.9)  # Improvement
        scheduler.step(1.1)  # No improvement

    def test_no_scheduler(self):
        """Test no scheduler."""
        scheduler = get_scheduler(self.optimizer, scheduler_type="none")

        assert scheduler is None


class TestDeterministicTrainer:
    """Test deterministic trainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.config = TrainingConfig(
            num_epochs=2,
            batch_size=2,
            log_frequency=1,
            save_frequency=1,
            val_frequency=1,
        )

        # Create mock dataloaders
        self.train_data = []
        self.val_data = []

        for i in range(4):  # 4 batches
            batch = {
                "electrons": torch.rand(2, 1, 32, 32) * 1000,
                "scale": torch.tensor(1000.0),
                "background": torch.tensor(100.0),
                "read_noise": torch.tensor(5.0),
            }
            self.train_data.append(batch)
            self.val_data.append(batch)

        self.train_dataloader = self.train_data
        self.val_dataloader = self.val_data

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_train_step(self):
        """Test single training step."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        batch = self.train_data[0]
        loss_dict = trainer.train_step(batch)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert all(isinstance(v, float) for v in loss_dict.values())

    def test_val_step(self):
        """Test single validation step."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        batch = self.val_data[0]
        loss_dict = trainer.val_step(batch)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert all(isinstance(v, float) for v in loss_dict.values())

    @patch("training.trainer.SummaryWriter")
    def test_train_epoch(self, mock_writer):
        """Test training for one epoch."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        epoch_losses = trainer.train_epoch()

        assert isinstance(epoch_losses, dict)
        assert "total_loss" in epoch_losses
        assert all(isinstance(v, float) for v in epoch_losses.values())

    @patch("training.trainer.SummaryWriter")
    def test_validate_epoch(self, mock_writer):
        """Test validation for one epoch."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        val_losses = trainer.validate_epoch()

        assert isinstance(val_losses, dict)
        assert "total_loss" in val_losses
        assert all(isinstance(v, float) for v in val_losses.values())

    def test_early_stopping(self):
        """Test early stopping logic."""
        config = TrainingConfig(early_stopping_patience=2, early_stopping_min_delta=0.1)
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, config
        )

        # Test improving loss
        assert not trainer.should_early_stop(1.0)
        assert not trainer.should_early_stop(0.8)  # Improvement
        assert not trainer.should_early_stop(0.85)  # Small increase, within patience
        assert not trainer.should_early_stop(0.87)  # Still within patience
        assert trainer.should_early_stop(0.89)  # Should trigger early stopping

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        trainer = DeterministicTrainer(
            self.model, self.train_dataloader, self.val_dataloader, self.config
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.config.checkpoint_dir = tmp_dir

            # Save checkpoint
            trainer.current_epoch = 5
            trainer.global_step = 100
            trainer.best_val_loss = 0.5
            trainer.save_checkpoint(is_best=True)

            # Load checkpoint
            checkpoint_path = Path(tmp_dir) / "best_checkpoint.pt"
            assert checkpoint_path.exists()

            # Create new trainer and load
            new_trainer = DeterministicTrainer(
                SimpleModel(), self.train_dataloader, self.val_dataloader, self.config
            )

            new_trainer.load_checkpoint(checkpoint_path, resume_training=True)

            assert new_trainer.current_epoch == 5
            assert new_trainer.global_step == 100
            assert new_trainer.best_val_loss == 0.5


class TestUtilities:
    """Test training utilities."""

    def test_set_deterministic_mode(self):
        """Test setting deterministic mode."""
        # Set deterministic mode
        set_deterministic_mode(seed=123)

        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        py_rand1 = [random.random() for _ in range(5)]

        # Reset and generate again
        set_deterministic_mode(seed=123)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        py_rand2 = [random.random() for _ in range(5)]

        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
        assert py_rand1 == py_rand2

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load utilities."""
        data = {
            "epoch": 10,
            "model_state": {"weight": torch.randn(5, 5)},
            "optimizer_state": {"lr": 0.001},
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            save_checkpoint(data, tmp.name)

            loaded_data = load_checkpoint(tmp.name)

            assert loaded_data["epoch"] == 10
            assert "model_state" in loaded_data
            assert "optimizer_state" in loaded_data
            assert torch.allclose(
                loaded_data["model_state"]["weight"], data["model_state"]["weight"]
            )

        Path(tmp.name).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for training system."""

    def test_create_trainer(self):
        """Test trainer creation utility."""
        model = SimpleModel()
        train_data = [{"electrons": torch.rand(2, 1, 32, 32) * 1000} for _ in range(2)]

        trainer = create_trainer(model, train_data, batch_size=2, num_epochs=1)

        assert isinstance(trainer, DeterministicTrainer)
        assert trainer.config.batch_size == 2
        assert trainer.config.num_epochs == 1

    @patch("training.trainer.SummaryWriter")
    def test_train_model_function(self, mock_writer):
        """Test high-level train_model function."""
        model = SimpleModel()
        train_data = []
        val_data = []

        # Create simple datasets
        for i in range(4):
            batch = {
                "electrons": torch.rand(2, 1, 32, 32) * 1000,
                "scale": torch.tensor(1000.0),
                "background": torch.tensor(100.0),
                "read_noise": torch.tensor(5.0),
            }
            train_data.append(batch)
            val_data.append(batch)

        config = TrainingConfig(num_epochs=1, batch_size=2, log_frequency=1)

        trained_model, history = train_model(model, train_data, val_data, config)

        assert trained_model is not None
        assert isinstance(history, dict)
        assert len(history) > 0

    def test_end_to_end_training(self):
        """Test complete end-to-end training workflow."""
        # This test verifies the entire training pipeline works together
        model = SimpleModel()

        # Create minimal dataset
        train_data = []
        for i in range(2):  # Just 2 batches for quick test
            batch = {
                "electrons": torch.rand(1, 1, 16, 16) * 1000,  # Small images
                "scale": torch.tensor(1000.0),
                "background": torch.tensor(100.0),
                "read_noise": torch.tensor(5.0),
            }
            train_data.append(batch)

        config = TrainingConfig(
            num_epochs=1,
            batch_size=1,
            log_frequency=1,
            save_frequency=1,
            mixed_precision=False,  # Disable for CPU testing
            deterministic=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.checkpoint_dir = tmp_dir
            config.tensorboard_log_dir = tmp_dir

            with patch("training.trainer.SummaryWriter"):
                trainer = DeterministicTrainer(model, train_data, None, config)
                history = trainer.train()

            assert isinstance(history, dict)
            assert len(history) > 0

            # Check that checkpoint was saved
            checkpoints = list(Path(tmp_dir).glob("checkpoint_*.pt"))
            assert len(checkpoints) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
