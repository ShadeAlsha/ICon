"""
Tests for Config Overrides and Epoch GIF Generation

Tests that:
1. CLI overrides (temperature, divergence, etc.) are properly applied
2. Final config.json reflects all overrides
3. Epoch GIF generation works end-to-end
4. Frames and GIFs are created with correct structure
"""

import pytest
import numpy as np
import torch
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.playground_config import PlaygroundConfig
from playground.playground_recipes import get_recipe
from playground.epoch_gif_utils import EpochGIFManager, create_training_dynamics_gif


# ---------------------------------------------------------------------------
# Config Override Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecipeOverrides:
    """Test that recipe overrides work correctly."""

    def test_temperature_override(self):
        """Temperature override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, temperature=0.15)
        for config in configs:
            assert config.temperature == 0.15, \
                f"Expected temperature=0.15, got {config.temperature}"

    def test_embedding_dim_override(self):
        """Embedding dimension override should be applied."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, embedding_dim=64)
        for config in configs:
            assert config.embedding_dim == 64, \
                f"Expected embedding_dim=64, got {config.embedding_dim}"

    def test_multiple_overrides(self):
        """Multiple overrides should all be applied."""
        configs = get_recipe(
            "simclr_vs_sne",
            dataset="mnist",
            epochs=2,
            temperature=0.25,
            divergence="js",
            optimizer="sgd",
            embedding_dim=32,
            batch_size=64,
        )
        for config in configs:
            assert config.temperature == 0.25
            assert config.divergence == "js"
            assert config.optimizer == "sgd"
            assert config.embedding_dim == 32
            assert config.batch_size == 64

    def test_override_persistence_in_to_dict(self):
        """Overrides should persist when config is converted to dict."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, temperature=0.1)
        config_dict = configs[0].to_dict()

        assert config_dict["temperature"] == 0.1, \
            f"temperature not in dict: {config_dict}"
        assert config_dict["dataset"] == "mnist"


# ---------------------------------------------------------------------------
# Epoch GIF Manager Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEpochGIFManager:
    """Test the EpochGIFManager utility."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return tmp_path / "gif_test"

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings and labels."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 16).astype(np.float32)
        labels = np.random.randint(0, 5, 100)
        return embeddings, labels

    def test_manager_initialization(self, temp_output_dir):
        """Manager should initialize and create output directory."""
        manager = EpochGIFManager(output_dir=temp_output_dir)
        assert temp_output_dir.exists()
        assert not manager.pca_fitted

    def test_pca_fitting(self, temp_output_dir, sample_embeddings):
        """PCA should be fit on embeddings."""
        manager = EpochGIFManager(output_dir=temp_output_dir)
        embeddings, _ = sample_embeddings

        manager.fit_pca(embeddings)
        assert manager.pca_fitted
        assert manager.pca is not None
        assert manager.pca.n_components_ == 2

    def test_save_epoch_frame(self, temp_output_dir, sample_embeddings):
        """Epoch frame should be saved as PNG."""
        manager = EpochGIFManager(output_dir=temp_output_dir)
        embeddings, labels = sample_embeddings

        frame_path = manager.save_epoch_frame(
            embeddings=embeddings,
            labels=labels,
            epoch=1,
        )

        assert frame_path.exists()
        assert frame_path.name == "epoch_001.png"
        assert manager.pca_fitted  # Should auto-fit PCA

    def test_multiple_epoch_frames(self, temp_output_dir, sample_embeddings):
        """Multiple epoch frames should be saved."""
        manager = EpochGIFManager(output_dir=temp_output_dir)
        embeddings, labels = sample_embeddings

        for epoch in range(1, 4):
            frame_path = manager.save_epoch_frame(
                embeddings=embeddings,
                labels=labels,
                epoch=epoch,
            )
            assert frame_path.exists()

        assert len(manager.frame_paths) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("imageio", reason="imageio not installed"),
        reason="imageio required for GIF creation"
    )
    def test_create_gif(self, temp_output_dir, sample_embeddings):
        """GIF should be created from saved frames."""
        manager = EpochGIFManager(output_dir=temp_output_dir)
        embeddings, labels = sample_embeddings

        # Save 3 frames
        for epoch in range(1, 4):
            manager.save_epoch_frame(embeddings, labels, epoch)

        # Create GIF
        gif_path = temp_output_dir / "test.gif"
        result_path = manager.create_gif(gif_path, duration=0.5)

        assert result_path.exists()
        assert result_path == gif_path

    def test_gif_creation_without_frames_raises(self, temp_output_dir):
        """Creating GIF without frames should raise error."""
        manager = EpochGIFManager(output_dir=temp_output_dir)

        with pytest.raises(RuntimeError, match="No frames saved"):
            manager.create_gif(temp_output_dir / "test.gif")


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConfigOverrideIntegration:
    """Test config override end-to-end with actual training."""

    def test_config_saved_with_overrides(self, tmp_path):
        """Final config.json should reflect all CLI overrides."""
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            epochs=1,
            batch_size=64,
            temperature=0.123,  # Non-default
            divergence="js",    # Non-default
            embedding_dim=32,    # Non-default
            num_workers=0,
            output_dir=str(tmp_path),
            run_name="test_config_override",
        )

        # Run experiment
        results = run_playground_experiment_pure_pytorch(
            config,
            verbose=False,
            gpu=False,
        )

        # Load saved config
        run_dir = Path(results["paths"]["run_dir"])
        config_path = run_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)

        # Verify overrides are saved
        assert saved_config["temperature"] == 0.123, \
            f"Expected temperature=0.123, got {saved_config.get('temperature')}"
        assert saved_config["divergence"] == "js", \
            f"Expected divergence=js, got {saved_config.get('divergence')}"
        assert saved_config["embedding_dim"] == 32, \
            f"Expected embedding_dim=32, got {saved_config.get('embedding_dim')}"


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("imageio", reason="imageio not installed"),
    reason="imageio required for GIF tests"
)
class TestGIFGenerationIntegration:
    """Test GIF generation end-to-end."""

    def test_gif_creation_in_training(self, tmp_path):
        """GIF should be created during training with save_epoch_gifs=True."""
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            epochs=2,  # Short training for speed
            batch_size=64,
            embedding_dim=16,  # Small for speed
            num_workers=0,
            output_dir=str(tmp_path),
            run_name="test_gif",
        )

        # Run with GIF generation
        results = run_playground_experiment_pure_pytorch(
            config,
            verbose=False,
            gpu=False,
            save_epoch_gifs=True,
        )

        run_dir = Path(results["paths"]["run_dir"])

        # Check frames directory exists
        frames_dir = run_dir / "epoch_frames"
        assert frames_dir.exists(), f"Frames directory not created: {frames_dir}"

        # Check frames were created
        frame_files = list(frames_dir.glob("epoch_*.png"))
        assert len(frame_files) == 2, \
            f"Expected 2 frames, found {len(frame_files)}: {[f.name for f in frame_files]}"

        # Check GIF was created
        gif_path = run_dir / "training_dynamics.gif"
        assert gif_path.exists(), f"GIF not created: {gif_path}"
        assert gif_path.stat().st_size > 0, "GIF file is empty"

    def test_no_gif_without_flag(self, tmp_path):
        """GIF should NOT be created when save_epoch_gifs=False."""
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            epochs=2,
            batch_size=64,
            embedding_dim=16,
            num_workers=0,
            output_dir=str(tmp_path),
            run_name="test_no_gif",
        )

        results = run_playground_experiment_pure_pytorch(
            config,
            verbose=False,
            gpu=False,
            save_epoch_gifs=False,  # Explicitly disabled
        )

        run_dir = Path(results["paths"]["run_dir"])

        # Check frames directory does NOT exist
        frames_dir = run_dir / "epoch_frames"
        assert not frames_dir.exists(), \
            f"Frames directory should not exist: {frames_dir}"

        # Check GIF was NOT created
        gif_path = run_dir / "training_dynamics.gif"
        assert not gif_path.exists(), \
            f"GIF should not be created: {gif_path}"


# ---------------------------------------------------------------------------
# CLI Tests (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCLIFlagParsing:
    """Test that CLI flags are parsed correctly."""

    def test_save_epoch_gifs_flag_exists(self):
        """--save_epoch_gifs flag should be recognized."""
        from playground.playground_cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--epochs", "5",
            "--save_epoch_gifs",
        ])

        assert hasattr(args, "save_epoch_gifs")
        assert args.save_epoch_gifs is True

    def test_save_epoch_gifs_default_false(self):
        """--save_epoch_gifs should default to False."""
        from playground.playground_cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--epochs", "5",
        ])

        assert hasattr(args, "save_epoch_gifs")
        assert args.save_epoch_gifs is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
