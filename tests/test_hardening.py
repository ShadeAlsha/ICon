"""
Hardening Tests for I-Con Playground

Tests for the robustness improvements added during the hardening phase:
- Probe tensor/numpy handling
- Recipe CLI overrides
- Experiment manifest generation
- Device assertions and debug mode

Run with:
    pytest tests/test_hardening.py -v
    pytest tests/test_hardening.py -v -m "unit"  # Fast tests only
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.playground_probes import (
    _ensure_numpy,
    run_linear_probe,
    run_knn_probe,
    analyze_class_separability,
)
from playground.playground_recipes import get_recipe, list_recipes
from playground.playground_config import PlaygroundConfig


# ---------------------------------------------------------------------------
# _ensure_numpy Helper Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureNumpy:
    """Tests for the _ensure_numpy helper function."""

    def test_numpy_array_passthrough(self):
        """Numpy arrays should pass through unchanged."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = _ensure_numpy(arr, "test")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_torch_tensor_cpu(self):
        """CPU torch tensors should be converted to numpy."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _ensure_numpy(tensor, "test")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, tensor.numpy())

    def test_torch_tensor_with_grad(self):
        """Torch tensors with gradients should be detached and converted."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result = _ensure_numpy(tensor, "test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_invalid_type_raises(self):
        """Invalid types should raise TypeError with helpful message."""
        with pytest.raises(TypeError, match="must be numpy array or torch tensor"):
            _ensure_numpy([1, 2, 3], "test_list")

    def test_error_message_includes_name(self):
        """Error message should include the provided name."""
        with pytest.raises(TypeError, match="'my_embeddings'"):
            _ensure_numpy("invalid", "my_embeddings")


# ---------------------------------------------------------------------------
# Probe Tensor Handling Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProbesTensorHandling:
    """Test that probes accept both numpy arrays and torch tensors."""

    @pytest.fixture
    def sample_data_numpy(self):
        """Create sample numpy data for testing."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 16).astype(np.float32)
        labels = np.random.randint(0, 5, 100)
        return embeddings, labels

    @pytest.fixture
    def sample_data_torch(self, sample_data_numpy):
        """Create sample torch data for testing."""
        embeddings, labels = sample_data_numpy
        return torch.from_numpy(embeddings), torch.from_numpy(labels)

    def test_linear_probe_numpy(self, sample_data_numpy):
        """Linear probe works with numpy arrays."""
        embeddings, labels = sample_data_numpy
        result = run_linear_probe(embeddings, labels, verbose=False)
        assert "test_accuracy" in result
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_linear_probe_torch(self, sample_data_torch):
        """Linear probe works with torch tensors."""
        embeddings, labels = sample_data_torch
        result = run_linear_probe(embeddings, labels, verbose=False)
        assert "test_accuracy" in result
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_linear_probe_mixed(self, sample_data_numpy, sample_data_torch):
        """Linear probe works with mixed numpy/torch inputs."""
        embeddings_np, _ = sample_data_numpy
        _, labels_torch = sample_data_torch
        result = run_linear_probe(embeddings_np, labels_torch, verbose=False)
        assert "test_accuracy" in result

    def test_knn_probe_numpy(self, sample_data_numpy):
        """kNN probe works with numpy arrays."""
        embeddings, labels = sample_data_numpy
        result = run_knn_probe(embeddings, labels, k=3, verbose=False)
        assert "test_accuracy" in result
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_knn_probe_torch(self, sample_data_torch):
        """kNN probe works with torch tensors."""
        embeddings, labels = sample_data_torch
        result = run_knn_probe(embeddings, labels, k=3, verbose=False)
        assert "test_accuracy" in result
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_separability_numpy(self, sample_data_numpy):
        """Separability analysis works with numpy arrays."""
        embeddings, labels = sample_data_numpy
        result = analyze_class_separability(embeddings, labels, max_samples=50, verbose=False)
        assert "separability_ratio" in result
        assert result["separability_ratio"] > 0

    def test_separability_torch(self, sample_data_torch):
        """Separability analysis works with torch tensors."""
        embeddings, labels = sample_data_torch
        result = analyze_class_separability(embeddings, labels, max_samples=50, verbose=False)
        assert "separability_ratio" in result
        assert result["separability_ratio"] > 0


# ---------------------------------------------------------------------------
# Probe Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProbeValidation:
    """Test probe input validation."""

    def test_linear_probe_empty_raises(self):
        """Linear probe should raise on empty embeddings."""
        empty_emb = np.array([]).reshape(0, 16)
        empty_labels = np.array([])
        with pytest.raises(ValueError, match="empty"):
            run_linear_probe(empty_emb, empty_labels, verbose=False)

    def test_knn_probe_empty_raises(self):
        """kNN probe should raise on empty embeddings."""
        empty_emb = np.array([]).reshape(0, 16)
        empty_labels = np.array([])
        with pytest.raises(ValueError, match="empty"):
            run_knn_probe(empty_emb, empty_labels, verbose=False)

    def test_separability_empty_raises(self):
        """Separability analysis should raise on empty embeddings."""
        empty_emb = np.array([]).reshape(0, 16)
        empty_labels = np.array([])
        with pytest.raises(ValueError, match="empty"):
            analyze_class_separability(empty_emb, empty_labels, verbose=False)

    def test_linear_probe_1d_raises(self):
        """Linear probe should raise on 1D embeddings."""
        emb_1d = np.random.randn(100)
        labels = np.random.randint(0, 5, 100)
        with pytest.raises(ValueError, match="2D"):
            run_linear_probe(emb_1d, labels, verbose=False)

    def test_linear_probe_mismatched_length_raises(self):
        """Linear probe should raise on mismatched lengths."""
        embeddings = np.random.randn(100, 16)
        labels = np.random.randint(0, 5, 50)  # Wrong length
        with pytest.raises(ValueError, match="same length"):
            run_linear_probe(embeddings, labels, verbose=False)


# ---------------------------------------------------------------------------
# Recipe CLI Override Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecipeCLIOverrides:
    """Test that CLI overrides are applied to recipe configs."""

    def test_divergence_override(self):
        """Divergence override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, divergence="js")
        for config in configs:
            assert config.divergence == "js"

    def test_optimizer_override(self):
        """Optimizer override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, optimizer="sgd")
        for config in configs:
            assert config.optimizer == "sgd"

    def test_weight_decay_override(self):
        """Weight decay override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, weight_decay=0.01)
        for config in configs:
            assert config.weight_decay == 0.01

    def test_batch_size_override(self):
        """Batch size override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, batch_size=64)
        for config in configs:
            assert config.batch_size == 64

    def test_learning_rate_override(self):
        """Learning rate override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, learning_rate=0.001)
        for config in configs:
            assert config.learning_rate == 0.001

    def test_num_workers_override(self):
        """Num workers override should be applied to all configs."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1, num_workers=4)
        for config in configs:
            assert config.num_workers == 4

    def test_multiple_overrides(self):
        """Multiple overrides should all be applied."""
        configs = get_recipe(
            "simclr_vs_sne",
            dataset="mnist",
            epochs=1,
            divergence="rkl",
            optimizer="adamw",
            weight_decay=0.05,
            batch_size=128
        )
        for config in configs:
            assert config.divergence == "rkl"
            assert config.optimizer == "adamw"
            assert config.weight_decay == 0.05
            assert config.batch_size == 128

    def test_no_override_preserves_defaults(self):
        """Without overrides, default values should be preserved."""
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1)
        for config in configs:
            # Check defaults
            assert config.divergence in ["kl", None]  # Default KL or None
            assert config.optimizer in ["adam", "adamw", None]


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigValidation:
    """Test config validation for new fields."""

    def test_valid_divergence(self):
        """Valid divergence values should be accepted."""
        for div in ["kl", "rkl", "js"]:
            config = PlaygroundConfig(dataset="mnist", backbone="mlp", divergence=div)
            config.validate()  # Validation only happens when explicitly called
            assert config.divergence == div

    def test_invalid_divergence_raises(self):
        """Invalid divergence should raise ValueError when validated."""
        config = PlaygroundConfig(dataset="mnist", backbone="mlp", divergence="invalid")
        with pytest.raises(ValueError, match="[Dd]ivergence"):
            config.validate()

    def test_valid_optimizer(self):
        """Valid optimizer values should be accepted."""
        for opt in ["adam", "adamw", "sgd"]:
            config = PlaygroundConfig(dataset="mnist", backbone="mlp", optimizer=opt)
            config.validate()  # Validation only happens when explicitly called
            assert config.optimizer == opt

    def test_invalid_optimizer_raises(self):
        """Invalid optimizer should raise ValueError when validated."""
        config = PlaygroundConfig(dataset="mnist", backbone="mlp", optimizer="invalid")
        with pytest.raises(ValueError, match="[Oo]ptimizer"):
            config.validate()


# ---------------------------------------------------------------------------
# Pure PyTorch Trainer Device Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeviceManager:
    """Test DeviceManager device selection."""

    def test_force_cpu(self):
        """gpu=False should force CPU."""
        from playground.pure_torch_trainer import DeviceManager
        device = DeviceManager.get_device(gpu=False, verbose=False)
        assert device.type == "cpu"

    def test_auto_select_returns_device(self):
        """Auto-select (gpu=None) should return a valid device."""
        from playground.pure_torch_trainer import DeviceManager
        device = DeviceManager.get_device(gpu=None, verbose=False)
        assert device.type in ["cpu", "cuda", "mps"]

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="No GPU available"
    )
    def test_require_gpu_with_gpu(self):
        """gpu=True with available GPU should succeed."""
        from playground.pure_torch_trainer import DeviceManager
        device = DeviceManager.get_device(gpu=True, verbose=False)
        assert device.type in ["cuda", "mps"]

    @pytest.mark.skipif(
        torch.cuda.is_available() or torch.backends.mps.is_available(),
        reason="GPU is available"
    )
    def test_require_gpu_without_gpu_raises(self):
        """gpu=True without GPU should raise RuntimeError."""
        from playground.pure_torch_trainer import DeviceManager
        with pytest.raises(RuntimeError, match="GPU requested"):
            DeviceManager.get_device(gpu=True, verbose=False)


@pytest.mark.unit
class TestDeviceAssertions:
    """Test device assertion helpers."""

    def test_assert_tensor_on_device_pass(self):
        """Assertion should pass when tensor is on expected device."""
        from playground.pure_torch_trainer import assert_tensor_on_device
        tensor = torch.tensor([1, 2, 3])
        device = torch.device("cpu")
        # Should not raise
        assert_tensor_on_device(tensor, device, "test_tensor")

    def test_assert_tensor_on_device_fail(self):
        """Assertion should fail when tensor is on wrong device."""
        from playground.pure_torch_trainer import assert_tensor_on_device
        tensor = torch.tensor([1, 2, 3])  # On CPU
        device = torch.device("cuda")  # Expect CUDA
        with pytest.raises(RuntimeError, match="Device mismatch"):
            assert_tensor_on_device(tensor, device, "test_tensor")


# ---------------------------------------------------------------------------
# All Recipes Test
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAllRecipes:
    """Test that all recipes can be loaded."""

    def test_all_recipes_loadable(self):
        """All listed recipes should be loadable."""
        recipe_names = list_recipes()
        for name in recipe_names:
            configs = get_recipe(name, dataset="mnist", epochs=1)
            assert isinstance(configs, list)
            assert len(configs) > 0
            for config in configs:
                assert isinstance(config, PlaygroundConfig)

    def test_recipe_configs_have_required_fields(self):
        """All recipe configs should have required fields."""
        recipe_names = list_recipes()
        for name in recipe_names:
            configs = get_recipe(name, dataset="mnist", epochs=1)
            for config in configs:
                assert hasattr(config, "dataset")
                assert hasattr(config, "backbone")
                assert hasattr(config, "icon_mode")
                assert hasattr(config, "epochs")
                assert hasattr(config, "run_name")


# ---------------------------------------------------------------------------
# Experiment Manifest Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExperimentManifest:
    """Test experiment manifest generation."""

    def test_manifest_function_exists(self):
        """Manifest creation function should exist."""
        from playground.playground_runner import _create_experiment_manifest
        assert callable(_create_experiment_manifest)

    def test_git_hash_function_exists(self):
        """Git hash function should exist."""
        from playground.playground_runner import _get_git_commit_hash
        assert callable(_get_git_commit_hash)

    def test_git_hash_returns_string_or_none(self):
        """Git hash should return string or None."""
        from playground.playground_runner import _get_git_commit_hash
        result = _get_git_commit_hash()
        assert result is None or isinstance(result, str)

    def test_manifest_has_required_fields(self):
        """Manifest should have all required fields."""
        from playground.playground_runner import _create_experiment_manifest
        import torch

        # Mock inputs
        pg_config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            epochs=1,
        )
        device = torch.device("cpu")
        logs = {
            "train_losses": [0.5],
            "val_losses": [0.6],
            "epoch_logs": [{"epoch": 0}]
        }
        run_dir = Path("/tmp/test_run")

        manifest = _create_experiment_manifest(pg_config, device, logs, run_dir)

        assert "timestamp" in manifest
        assert "config" in manifest
        assert "environment" in manifest
        assert "results" in manifest
        assert "artifacts" in manifest

        env = manifest["environment"]
        assert "torch_version" in env
        assert "python_version" in env
        assert "platform" in env
        assert "device" in env
        assert "git_commit" in env

        results = manifest["results"]
        assert "final_train_loss" in results
        assert "final_val_loss" in results
        assert "num_epochs_completed" in results
