"""
Playground Smoke Tests

Minimal sanity checks for the I-Con Playground to ensure basic functionality.

Test Markers:
    unit: Fast tests with no training or dataset downloads (~1s total)
    integration: Tests with 1-epoch MNIST training (~30s total)
    slow: Tests with t-SNE or multi-run recipes (minutes)

Run with:
    pytest tests/test_playground_smoke.py -v              # All tests
    pytest tests/test_playground_smoke.py -v -m "unit"    # Fast unit tests only
    pytest tests/test_playground_smoke.py -v -m "not slow"  # Unit + integration (<30s)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.playground_config import PlaygroundConfig
from playground.playground_runner import run_playground_experiment
from playground.playground_probes import run_linear_probe, run_knn_probe, analyze_class_separability
from playground.playground_viz import plot_training_curves, plot_embeddings_2d
from playground.playground_recipes import get_recipe, list_recipes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """Create a minimal config for fast testing."""
    return PlaygroundConfig(
        dataset="mnist",
        backbone="mlp",
        icon_mode="simclr_like",
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        temperature=0.5,
        embedding_dim=32,  # Small embedding for speed
        output_dir="test_outputs",
        num_workers=0,  # No multiprocessing for tests
        seed=42,
    )


@pytest.fixture
def minimal_results(minimal_config):
    """Run a minimal experiment and return results."""
    results = run_playground_experiment(
        minimal_config,
        verbose=False,
        save_checkpoints=False,
        gpu=False,  # Force CPU for tests
    )
    return results


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_config_creation():
    """Test that PlaygroundConfig can be created with defaults."""
    config = PlaygroundConfig()
    assert config.dataset == "cifar10"
    assert config.backbone == "resnet18"
    assert config.icon_mode == "simclr_like"
    assert config.epochs == 10


@pytest.mark.unit
def test_config_validation_invalid_dataset():
    """Test that invalid dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        PlaygroundConfig(dataset="invalid_dataset")


@pytest.mark.unit
def test_config_validation_invalid_backbone():
    """Test that invalid backbone raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        PlaygroundConfig(backbone="invalid_backbone")


@pytest.mark.unit
def test_config_validation_invalid_icon_mode():
    """Test that invalid icon_mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported icon_mode"):
        PlaygroundConfig(icon_mode="invalid_mode")


@pytest.mark.unit
def test_config_to_icon_config(minimal_config):
    """Test that PlaygroundConfig can be converted to I-Con Config."""
    icon_config = minimal_config.to_icon_config()
    assert icon_config is not None
    assert hasattr(icon_config, "mapper")
    assert hasattr(icon_config, "supervisory_distribution")
    assert hasattr(icon_config, "learned_distribution")


@pytest.mark.unit
def test_config_all_icon_modes():
    """Test that all supported I-Con modes can be instantiated."""
    modes = [
        "simclr_like", "sne_like", "tsne_like", "supervised", "cluster_like",
        "barlow_twins_like", "vicreg_like", "debiasing_like"
    ]
    for mode in modes:
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode=mode,
            epochs=1,
        )
        icon_config = config.to_icon_config()
        assert icon_config is not None, f"Failed to create config for mode: {mode}"


# ---------------------------------------------------------------------------
# Experiment Runner Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_minimal_experiment(minimal_config):
    """Test that a minimal experiment can run to completion."""
    results = run_playground_experiment(
        minimal_config,
        verbose=False,
        save_checkpoints=False,
        gpu=False,
    )

    assert "logs" in results
    assert "embeddings" in results
    assert "labels" in results
    assert "config" in results
    assert "paths" in results


@pytest.mark.integration
def test_embeddings_and_labels_non_empty(minimal_results):
    """Test that embeddings and labels are non-empty arrays."""
    embeddings = minimal_results["embeddings"]
    labels = minimal_results["labels"]

    assert isinstance(embeddings, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert embeddings.shape[0] > 0
    assert labels.shape[0] > 0
    assert embeddings.shape[0] == labels.shape[0]


@pytest.mark.integration
def test_logs_has_epoch_data(minimal_results):
    """Test that logs contain at least 1 epoch of data."""
    logs = minimal_results["logs"]

    assert "train_losses" in logs
    assert "epoch_logs" in logs
    assert len(logs["train_losses"]) >= 1
    assert len(logs["epoch_logs"]) >= 1


@pytest.mark.integration
def test_embedding_dimensionality(minimal_results):
    """Test that embeddings have correct dimensionality."""
    embeddings = minimal_results["embeddings"]
    config = minimal_results["config"]

    expected_dim = config["embedding_dim"]
    assert embeddings.shape[1] == expected_dim


# ---------------------------------------------------------------------------
# Probe Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_linear_probe(minimal_results):
    """Test that linear probe runs and returns valid accuracy."""
    embeddings = minimal_results["embeddings"]
    labels = minimal_results["labels"]

    probe_results = run_linear_probe(
        embeddings,
        labels,
        test_size=0.2,
        verbose=False,
    )

    assert "train_accuracy" in probe_results
    assert "test_accuracy" in probe_results
    assert 0.0 <= probe_results["train_accuracy"] <= 1.0
    assert 0.0 <= probe_results["test_accuracy"] <= 1.0


@pytest.mark.integration
def test_knn_probe(minimal_results):
    """Test that kNN probe runs and returns valid accuracy."""
    embeddings = minimal_results["embeddings"]
    labels = minimal_results["labels"]

    knn_results = run_knn_probe(
        embeddings,
        labels,
        k=5,
        test_size=0.2,
        verbose=False,
    )

    assert "train_accuracy" in knn_results
    assert "test_accuracy" in knn_results
    assert 0.0 <= knn_results["train_accuracy"] <= 1.0
    assert 0.0 <= knn_results["test_accuracy"] <= 1.0


@pytest.mark.integration
def test_separability_analysis(minimal_results):
    """Test that separability analysis runs and returns valid metrics."""
    embeddings = minimal_results["embeddings"]
    labels = minimal_results["labels"]

    sep_results = analyze_class_separability(
        embeddings,
        labels,
        max_samples=500,
        verbose=False,
    )

    assert "within_class_distance_mean" in sep_results
    assert "between_class_distance_mean" in sep_results
    assert "separability_ratio" in sep_results
    assert sep_results["separability_ratio"] > 0


# ---------------------------------------------------------------------------
# Visualization Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_plot_training_curves(minimal_results):
    """Test that training curves can be plotted without errors."""
    logs = minimal_results["logs"]

    fig = plot_training_curves(logs, show=False)
    assert fig is not None


@pytest.mark.integration
def test_plot_embeddings_2d_pca(minimal_results):
    """Test that 2D embeddings can be plotted with PCA."""
    embeddings = minimal_results["embeddings"]
    labels = minimal_results["labels"]

    fig = plot_embeddings_2d(
        embeddings,
        labels,
        method="pca",
        show=False,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Recipe Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_list_recipes():
    """Test that recipe listing works."""
    recipes = list_recipes()
    assert isinstance(recipes, list)
    assert len(recipes) > 0
    assert "simclr_vs_sne" in recipes


@pytest.mark.unit
def test_get_recipe_simclr_vs_sne():
    """Test that simclr_vs_sne recipe returns valid configs."""
    configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1)

    assert isinstance(configs, list)
    assert len(configs) == 2
    assert all(isinstance(cfg, PlaygroundConfig) for cfg in configs)
    assert configs[0].icon_mode == "simclr_like"
    assert configs[1].icon_mode == "sne_like"


@pytest.mark.unit
def test_get_recipe_invalid_name():
    """Test that invalid recipe name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown recipe"):
        get_recipe("invalid_recipe_name")


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_integration(tmp_path):
    """Test complete pipeline: config -> train -> probe -> viz."""
    config = PlaygroundConfig(
        dataset="mnist",
        backbone="mlp",
        icon_mode="simclr_like",
        epochs=1,
        batch_size=32,
        embedding_dim=32,
        output_dir=str(tmp_path / "test_run"),
        num_workers=0,
    )

    results = run_playground_experiment(config, verbose=False, gpu=False)
    assert results["embeddings"].shape[0] > 0

    linear_acc = run_linear_probe(
        results["embeddings"],
        results["labels"],
        verbose=False,
    )["test_accuracy"]

    knn_acc = run_knn_probe(
        results["embeddings"],
        results["labels"],
        k=5,
        verbose=False,
    )["test_accuracy"]

    assert 0.0 <= linear_acc <= 1.0
    assert 0.0 <= knn_acc <= 1.0

    fig1 = plot_training_curves(results["logs"], show=False)
    fig2 = plot_embeddings_2d(results["embeddings"], results["labels"], method="pca", show=False)

    assert fig1 is not None
    assert fig2 is not None

    print(f"\n✓ Full pipeline test passed!")
    print(f"  Linear probe: {linear_acc:.2%}")
    print(f"  kNN probe: {knn_acc:.2%}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_outputs():
    """Clean up test output directories after all tests."""
    yield

    import shutil
    test_dirs = ["test_outputs", "playground_runs"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            try:
                shutil.rmtree(test_dir)
                print(f"\nCleaned up: {test_dir}")
            except Exception as e:
                print(f"\nWarning: Could not clean up {test_dir}: {e}")
