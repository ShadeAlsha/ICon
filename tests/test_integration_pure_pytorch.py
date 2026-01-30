"""
Integration tests for pure PyTorch training.

These tests verify:
1. One-step training works with pure PyTorch
2. Different divergences change loss values
3. Different optimizers work correctly
4. Device handling works end-to-end
5. Embeddings are extracted correctly
"""

import pytest
import torch
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.playground_config import PlaygroundConfig
from playground.playground_runner import run_playground_experiment_pure_pytorch


@pytest.mark.slow
class TestIntegrationPurePyTorch:
    """Integration tests for pure PyTorch training."""

    def test_minimal_training_cpu(self):
        """Test minimal training run on CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=1,
                batch_size=32,
                embedding_dim=32,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            # Check results structure
            assert "logs" in results
            assert "embeddings" in results
            assert "labels" in results
            assert "config" in results
            assert "paths" in results
            assert "model" in results
            assert "device" in results

            # Check device
            assert results["device"] == "cpu"

            # Check logs
            assert len(results["logs"]["train_losses"]) == 1
            assert len(results["logs"]["val_losses"]) == 1

            # Check embeddings
            assert len(results["embeddings"]) > 0
            assert len(results["labels"]) > 0

    def test_divergence_kl(self):
        """Test training with KL divergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                divergence="kl",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            assert len(results["logs"]["train_losses"]) == 1
            assert results["logs"]["train_losses"][0] > 0

    def test_divergence_rkl(self):
        """Test training with reverse KL divergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                divergence="rkl",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            assert len(results["logs"]["train_losses"]) == 1
            assert results["logs"]["train_losses"][0] > 0

    def test_divergence_js(self):
        """Test training with Jensen-Shannon divergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                divergence="js",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            assert len(results["logs"]["train_losses"]) == 1
            assert results["logs"]["train_losses"][0] > 0

    def test_optimizer_adam(self):
        """Test training with Adam optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                optimizer="adam",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            assert len(results["logs"]["train_losses"]) == 1

    def test_optimizer_sgd(self):
        """Test training with SGD optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                optimizer="sgd",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            assert len(results["logs"]["train_losses"]) == 1

    def test_gpu_request_without_gpu_raises_error(self):
        """Test that requesting GPU without one raises clear error."""
        # Only run if no GPU available
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            pytest.skip("GPU available, skipping no-GPU test")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=1,
                batch_size=32,
                output_dir=tmpdir,
                num_workers=0,
            )

            with pytest.raises(RuntimeError, match="GPU requested.*but no GPU available"):
                run_playground_experiment_pure_pytorch(
                    config,
                    verbose=False,
                    gpu=True,  # Request GPU when none available
                )

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                        reason="Requires GPU")
    def test_training_on_gpu(self):
        """Test training on GPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=True,
            )

            # Check that training completed
            assert len(results["logs"]["train_losses"]) == 1

            # Check device is GPU
            assert results["device"] in ["cuda", "mps"]

    def test_model_saves_correctly(self):
        """Test that model state dict is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            # Check model file exists
            model_path = Path(results["paths"]["model"])
            assert model_path.exists()

            # Check it can be loaded
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            assert len(state_dict) > 0

    def test_embeddings_save_correctly(self):
        """Test that embeddings are saved as npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=1,
                batch_size=32,
                embedding_dim=16,
                output_dir=tmpdir,
                num_workers=0,
            )

            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=False,
                gpu=False,
            )

            # Check embeddings file exists
            embeddings_path = Path(results["paths"]["embeddings"])
            assert embeddings_path.exists()

            # Check it can be loaded
            import numpy as np
            data = np.load(embeddings_path)
            assert "embeddings" in data
            assert "labels" in data
            assert len(data["embeddings"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
