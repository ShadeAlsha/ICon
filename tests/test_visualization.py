"""
Tests for the Visualization Pipeline

Comprehensive tests for GIF generation, embedding collection, and projection.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json


# ============================================================================
# Unit Tests - Fast, no external dependencies
# ============================================================================

@pytest.mark.unit
class TestVizConfig:
    """Test VizConfig creation and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        from playground.viz.config import VizConfig, VizMode, GifMethod, GifOverlay

        config = VizConfig()
        assert config.viz_mode == VizMode.BOTH
        assert config.gif_every == 1
        assert config.gif_method == GifMethod.PCA
        assert config.gif_fps == 2.0
        assert config.gif_max_points == 5000
        assert config.gif_overlay == GifOverlay.EPOCH

    def test_string_to_enum_conversion(self):
        """Test that string inputs are converted to enums."""
        from playground.viz.config import VizConfig, VizMode, GifMethod

        config = VizConfig(viz_mode="gif", gif_method="tsne")
        assert config.viz_mode == VizMode.GIF
        assert config.gif_method == GifMethod.TSNE

    def test_invalid_gif_every(self):
        """Test validation for gif_every < 1."""
        from playground.viz.config import VizConfig

        with pytest.raises(ValueError, match="gif_every must be >= 1"):
            VizConfig(gif_every=0)

    def test_invalid_gif_fps(self):
        """Test validation for gif_fps <= 0."""
        from playground.viz.config import VizConfig

        with pytest.raises(ValueError, match="gif_fps must be > 0"):
            VizConfig(gif_fps=0)

    def test_invalid_gif_max_points(self):
        """Test validation for gif_max_points < 100."""
        from playground.viz.config import VizConfig

        with pytest.raises(ValueError, match="gif_max_points must be >= 100"):
            VizConfig(gif_max_points=50)

    def test_should_generate_gif(self):
        """Test should_generate_gif property."""
        from playground.viz.config import VizConfig, VizMode

        assert VizConfig(viz_mode=VizMode.GIF).should_generate_gif is True
        assert VizConfig(viz_mode=VizMode.BOTH).should_generate_gif is True
        assert VizConfig(viz_mode=VizMode.STATIC).should_generate_gif is False
        assert VizConfig(viz_mode=VizMode.NONE).should_generate_gif is False

    def test_should_generate_static(self):
        """Test should_generate_static property."""
        from playground.viz.config import VizConfig, VizMode

        assert VizConfig(viz_mode=VizMode.STATIC).should_generate_static is True
        assert VizConfig(viz_mode=VizMode.BOTH).should_generate_static is True
        assert VizConfig(viz_mode=VizMode.GIF).should_generate_static is False
        assert VizConfig(viz_mode=VizMode.NONE).should_generate_static is False

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        from playground.viz.config import VizConfig, GifMethod

        original = VizConfig(gif_every=5, gif_method=GifMethod.TSNE, gif_fps=1.0)
        d = original.to_dict()
        restored = VizConfig.from_dict(d)

        assert restored.gif_every == original.gif_every
        assert restored.gif_method == original.gif_method
        assert restored.gif_fps == original.gif_fps


@pytest.mark.unit
class TestProjector:
    """Test Projector class."""

    def test_pca_fit_transform(self):
        """Test PCA projection fit and transform."""
        from playground.viz.projector import Projector, GifMethod

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)

        projector = Projector(method=GifMethod.PCA, random_state=42)
        assert projector.fitted is False

        projector.fit(embeddings)
        assert projector.fitted is True
        assert projector.explained_variance_ratio is not None
        assert len(projector.explained_variance_ratio) == 2

        result = projector.transform(embeddings)
        assert result.shape == (100, 2)

    def test_pca_determinism(self):
        """Test that PCA projection is deterministic with same seed."""
        from playground.viz.projector import Projector, GifMethod

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)

        proj1 = Projector(method=GifMethod.PCA, random_state=42)
        proj1.fit(embeddings)
        result1 = proj1.transform(embeddings)

        proj2 = Projector(method=GifMethod.PCA, random_state=42)
        proj2.fit(embeddings)
        result2 = proj2.transform(embeddings)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pca_reuse(self):
        """Test that PCA is fit once and reused for multiple transforms."""
        from playground.viz.projector import Projector, GifMethod

        np.random.seed(42)
        ref_embeddings = np.random.randn(100, 64)
        other_embeddings = np.random.randn(50, 64)

        projector = Projector(method=GifMethod.PCA, random_state=42)
        projector.fit(ref_embeddings)

        # Transform multiple times - should use same PCA
        result1 = projector.transform(ref_embeddings)
        result2 = projector.transform(other_embeddings)

        assert result1.shape == (100, 2)
        assert result2.shape == (50, 2)

        # Verify the projector wasn't re-fit
        assert projector.total_variance_explained is not None

    def test_transform_without_fit_raises(self):
        """Test that transform raises error if not fitted."""
        from playground.viz.projector import Projector

        projector = Projector()
        embeddings = np.random.randn(100, 64)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            projector.transform(embeddings)

    def test_fit_transform_shortcut(self):
        """Test fit_transform convenience method."""
        from playground.viz.projector import Projector, GifMethod

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)

        projector = Projector(method=GifMethod.PCA, random_state=42)
        result = projector.fit_transform(embeddings)

        assert result.shape == (100, 2)
        assert projector.fitted is True

    def test_get_info(self):
        """Test get_info returns expected information."""
        from playground.viz.projector import Projector, GifMethod

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)

        projector = Projector(method=GifMethod.PCA, random_state=42)
        info_before = projector.get_info()
        assert info_before['fitted'] is False
        assert info_before['method'] == 'pca'

        projector.fit(embeddings)
        info_after = projector.get_info()
        assert info_after['fitted'] is True
        assert 'explained_variance_ratio' in info_after
        assert 'total_variance_explained' in info_after


@pytest.mark.unit
class TestEpochEmbeddingCollector:
    """Test EpochEmbeddingCollector class."""

    def test_should_collect_default(self):
        """Test should_collect with gif_every=1."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(
                output_dir=Path(tmpdir),
                gif_every=1
            )

            # Should collect every epoch
            assert collector.should_collect(1) is True
            assert collector.should_collect(2) is True
            assert collector.should_collect(3) is True

    def test_should_collect_every_5(self):
        """Test should_collect with gif_every=5."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(
                output_dir=Path(tmpdir),
                gif_every=5
            )

            # First epoch always collected
            assert collector.should_collect(1) is True
            assert collector.should_collect(2) is False
            assert collector.should_collect(3) is False
            assert collector.should_collect(4) is False
            assert collector.should_collect(5) is True
            assert collector.should_collect(10) is True
            assert collector.should_collect(11) is False

    def test_save_and_load_epoch(self):
        """Test saving and loading epoch embeddings."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        labels = np.random.randint(0, 10, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(output_dir=Path(tmpdir))
            collector.save_epoch(epoch=1, embeddings=embeddings, labels=labels)

            # Load and verify
            data = collector.get_epoch(1)
            assert data is not None
            np.testing.assert_array_almost_equal(data['embeddings'], embeddings)
            np.testing.assert_array_equal(data['labels'], labels)
            assert data['epoch'] == 1

    def test_get_all_epochs(self):
        """Test loading all epoch embeddings."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(output_dir=Path(tmpdir))

            # Save 3 epochs
            for epoch in [1, 5, 10]:
                embeddings = np.random.randn(100, 64)
                labels = np.random.randint(0, 10, 100)
                collector.save_epoch(epoch=epoch, embeddings=embeddings, labels=labels)

            all_epochs = collector.get_all_epochs()
            assert len(all_epochs) == 3
            assert 1 in all_epochs
            assert 5 in all_epochs
            assert 10 in all_epochs

    def test_subsampling(self):
        """Test that large embeddings are subsampled."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)
        embeddings = np.random.randn(10000, 64)
        labels = np.random.randint(0, 10, 10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(
                output_dir=Path(tmpdir),
                max_points=1000
            )
            collector.save_epoch(epoch=1, embeddings=embeddings, labels=labels)

            data = collector.get_epoch(1)
            assert data['embeddings'].shape[0] == 1000
            assert data['labels'].shape[0] == 1000

    def test_check_embedding_changes_detects_change(self):
        """Test sanity check detects changing embeddings."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(output_dir=Path(tmpdir))

            # Save two different epochs with different embeddings
            np.random.seed(42)
            collector.save_epoch(
                epoch=1,
                embeddings=np.random.randn(100, 64),
                labels=np.random.randint(0, 10, 100)
            )
            collector.save_epoch(
                epoch=2,
                embeddings=np.random.randn(100, 64) + 10,  # Clearly different
                labels=np.random.randint(0, 10, 100)
            )

            check = collector.check_embedding_changes()
            assert check['changed'] is True
            assert check['warning'] is None

    def test_check_embedding_changes_detects_unchanged(self):
        """Test sanity check warns on unchanged embeddings."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)
        same_embeddings = np.random.randn(100, 64)
        same_labels = np.random.randint(0, 10, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EpochEmbeddingCollector(output_dir=Path(tmpdir))

            # Save same embeddings twice
            collector.save_epoch(epoch=1, embeddings=same_embeddings, labels=same_labels)
            collector.save_epoch(epoch=2, embeddings=same_embeddings, labels=same_labels)

            check = collector.check_embedding_changes()
            assert check['changed'] is False
            assert check['warning'] is not None
            assert "unchanged" in check['warning'].lower()

    def test_load_from_existing(self):
        """Test loading collector from existing run directory."""
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            collector1 = EpochEmbeddingCollector(
                output_dir=Path(tmpdir),
                gif_every=5
            )
            embeddings = np.random.randn(100, 64)
            labels = np.random.randint(0, 10, 100)
            collector1.save_epoch(epoch=5, embeddings=embeddings, labels=labels)

            # Load
            collector2 = EpochEmbeddingCollector.load(Path(tmpdir))
            assert collector2.gif_every == 5
            assert 5 in collector2.get_collected_epochs()


# ============================================================================
# Integration Tests - Test GIF generation with dummy data
# ============================================================================

@pytest.mark.integration
class TestGifGeneration:
    """Test GIF generation pipeline."""

    def test_generate_training_gif_basic(self):
        """Test basic GIF generation with dummy embeddings."""
        from playground.viz import generate_training_gif, VizConfig, VizMode, GifMethod

        np.random.seed(42)

        # Create dummy embeddings by epoch
        embeddings_by_epoch = {}
        for epoch in [1, 2, 3]:
            embeddings_by_epoch[epoch] = {
                'embeddings': np.random.randn(100, 64) + epoch * 0.5,  # Different each epoch
                'labels': np.random.randint(0, 5, 100),
            }

        labels = embeddings_by_epoch[1]['labels']

        config = VizConfig(
            viz_mode=VizMode.GIF,
            gif_method=GifMethod.PCA,
            gif_fps=2.0,
            gif_max_points=100,
            save_frames=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=labels,
                config=config,
                output_dir=Path(tmpdir),
                random_state=42,
            )

            # Check GIF was created
            assert result['gif_path'].exists()
            assert result['n_epochs'] == 3

            # Check frames were saved
            assert len(result['frame_paths']) == 3

            # Check projector info
            assert 'projector_info' in result
            assert result['projector_info']['method'] == 'pca'
            assert result['projector_info']['fitted'] is True

            # Check sanity check passed
            assert result['sanity_check']['changed'] is True

    def test_generate_gif_with_metadata_overlay(self):
        """Test GIF generation with loss overlay."""
        from playground.viz import generate_training_gif, VizConfig, GifOverlay

        np.random.seed(42)

        embeddings_by_epoch = {
            1: {'embeddings': np.random.randn(50, 32), 'labels': np.random.randint(0, 3, 50)},
            2: {'embeddings': np.random.randn(50, 32), 'labels': np.random.randint(0, 3, 50)},
        }

        epoch_metadata = {
            1: {'train_loss': 1.5, 'val_loss': 1.6},
            2: {'train_loss': 1.0, 'val_loss': 1.1},
        }

        config = VizConfig(gif_overlay=GifOverlay.LOSS, gif_max_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
                epoch_metadata=epoch_metadata,
            )

            assert result['gif_path'].exists()

    def test_generate_gif_without_saving_frames(self):
        """Test GIF generation with save_frames=False."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        embeddings_by_epoch = {
            1: {'embeddings': np.random.randn(50, 32), 'labels': np.random.randint(0, 3, 50)},
            2: {'embeddings': np.random.randn(50, 32), 'labels': np.random.randint(0, 3, 50)},
        }

        config = VizConfig(save_frames=False, gif_max_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
            )

            # GIF should exist
            assert result['gif_path'].exists()
            # Frames should be cleaned up
            assert result['frame_paths'] == []

    def test_deterministic_gif_generation(self):
        """Test that GIF generation is deterministic."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        labels = np.random.randint(0, 5, 100)

        embeddings_by_epoch = {
            1: {'embeddings': embeddings, 'labels': labels},
            2: {'embeddings': embeddings + 1, 'labels': labels},
        }

        config = VizConfig(gif_max_points=100)

        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = generate_training_gif(
                    embeddings_by_epoch=embeddings_by_epoch,
                    labels=labels,
                    config=config,
                    output_dir=Path(tmpdir),
                    random_state=42,
                )
                results.append(result)

        # Projector info should be identical
        assert results[0]['projector_info']['total_variance_explained'] == \
               results[1]['projector_info']['total_variance_explained']

    def test_variable_sized_epochs_uses_anchor_tracking(self):
        """Test GIF generation with variable-sized epochs uses anchor-based tracking."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        # Create epochs with DIFFERENT sizes (simulates custom dataset behavior)
        embeddings_by_epoch = {
            1: {
                'embeddings': np.random.randn(100, 64),
                'labels': np.random.randint(0, 5, 100),
            },
            2: {
                'embeddings': np.random.randn(80, 64) + 0.5,  # Smaller epoch
                'labels': np.random.randint(0, 5, 80),
            },
            3: {
                'embeddings': np.random.randn(120, 64) + 1.0,  # Larger epoch
                'labels': np.random.randint(0, 5, 120),
            },
        }

        config = VizConfig(gif_max_points=150)  # More than largest epoch

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
                random_state=42,
            )

            # GIF should be created successfully
            assert result['gif_path'].exists()

            # Check anchor_strategy is present and correct
            assert 'anchor_strategy' in result
            anchor_info = result['anchor_strategy']
            assert anchor_info['strategy'] == 'fixed_anchor_tracking'
            assert anchor_info['variable_sizes'] is True
            assert anchor_info['min_epoch_count'] == 80
            assert anchor_info['max_epoch_count'] == 120
            # Anchor count should be min(gif_max_points, min_epoch_count)
            assert anchor_info['anchor_count'] == 80

    def test_variable_sized_epochs_no_silent_drops(self):
        """Test that variable-sized epochs don't silently drop data."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        # Create 5 epochs with different sizes
        embeddings_by_epoch = {
            1: {'embeddings': np.random.randn(100, 64), 'labels': np.random.randint(0, 5, 100)},
            2: {'embeddings': np.random.randn(90, 64) + 0.2, 'labels': np.random.randint(0, 5, 90)},
            3: {'embeddings': np.random.randn(95, 64) + 0.4, 'labels': np.random.randint(0, 5, 95)},
            4: {'embeddings': np.random.randn(85, 64) + 0.6, 'labels': np.random.randint(0, 5, 85)},
            5: {'embeddings': np.random.randn(100, 64) + 0.8, 'labels': np.random.randint(0, 5, 100)},
        }

        config = VizConfig(gif_max_points=200, save_frames=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
                random_state=42,
            )

            # All 5 epochs should be included (none dropped)
            assert result['n_epochs'] == 5
            assert len(result['frame_paths']) == 5

    def test_uniform_sized_epochs_still_works(self):
        """Test that uniform-sized epochs still work correctly."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        # All epochs have the same size
        embeddings_by_epoch = {
            1: {'embeddings': np.random.randn(100, 64), 'labels': np.random.randint(0, 5, 100)},
            2: {'embeddings': np.random.randn(100, 64) + 0.5, 'labels': np.random.randint(0, 5, 100)},
            3: {'embeddings': np.random.randn(100, 64) + 1.0, 'labels': np.random.randint(0, 5, 100)},
        }

        config = VizConfig(gif_max_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
                random_state=42,
            )

            assert result['gif_path'].exists()
            assert 'anchor_strategy' in result
            # variable_sizes should be False for uniform epochs
            assert result['anchor_strategy']['variable_sizes'] is False

    def test_anchor_tracking_determinism(self):
        """Test that anchor-based tracking produces deterministic results."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        # Variable-sized epochs
        embeddings_by_epoch = {
            1: {'embeddings': np.random.randn(100, 64), 'labels': np.random.randint(0, 5, 100)},
            2: {'embeddings': np.random.randn(80, 64) + 0.5, 'labels': np.random.randint(0, 5, 80)},
        }

        config = VizConfig(gif_max_points=150)

        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = generate_training_gif(
                    embeddings_by_epoch=embeddings_by_epoch,
                    labels=embeddings_by_epoch[1]['labels'],
                    config=config,
                    output_dir=Path(tmpdir),
                    random_state=42,  # Same seed
                )
                results.append(result)

        # Anchor strategy should be identical
        assert results[0]['anchor_strategy'] == results[1]['anchor_strategy']
        # Projector info should be identical
        assert results[0]['projector_info']['total_variance_explained'] == \
               results[1]['projector_info']['total_variance_explained']

    def test_sanity_check_uses_anchors(self):
        """Test that embedding change sanity check uses anchor indices."""
        from playground.viz import generate_training_gif, VizConfig

        np.random.seed(42)

        # Create base embeddings with consistent range
        base_embeddings = np.random.randn(100, 64)

        # Epochs with different sizes but clearly changing embeddings
        # Use scaled shifts to keep data in similar range for consistent frame sizes
        embeddings_by_epoch = {
            1: {'embeddings': base_embeddings.copy(), 'labels': np.random.randint(0, 5, 100)},
            2: {'embeddings': base_embeddings[:80] * 2 + 5, 'labels': np.random.randint(0, 5, 80)},  # Clear shift
            3: {'embeddings': base_embeddings[:90] * 1.5 + 3, 'labels': np.random.randint(0, 5, 90)},  # Add 3rd epoch
        }

        config = VizConfig(gif_max_points=150)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embeddings_by_epoch[1]['labels'],
                config=config,
                output_dir=Path(tmpdir),
                random_state=42,
            )

            # Sanity check should detect that embeddings changed
            assert result['sanity_check']['changed'] is True
            # Should report anchor count
            assert 'anchor_count' in result['sanity_check']


@pytest.mark.integration
class TestRegenGif:
    """Test GIF regeneration from saved embeddings."""

    def test_regenerate_gif_from_run(self):
        """Test regenerating GIF from a saved run."""
        from playground.viz import regenerate_gif_from_run, VizConfig
        from playground.viz.embedding_collector import EpochEmbeddingCollector

        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create a mock run with saved embeddings
            collector = EpochEmbeddingCollector(output_dir=run_dir)
            for epoch in [1, 5, 10]:
                embeddings = np.random.randn(100, 64) + epoch * 0.5
                labels = np.random.randint(0, 5, 100)
                collector.save_epoch(epoch=epoch, embeddings=embeddings, labels=labels)

            # Save mock logs
            logs = {
                'train_losses': [1.5, 1.2, 1.0],
                'val_losses': [1.6, 1.3, 1.1],
                'epoch_logs': [
                    {'epoch': 0, 'train_loss': 1.5, 'val_loss': 1.6},
                    {'epoch': 4, 'train_loss': 1.2, 'val_loss': 1.3},
                    {'epoch': 9, 'train_loss': 1.0, 'val_loss': 1.1},
                ]
            }
            with open(run_dir / 'logs.json', 'w') as f:
                json.dump(logs, f)

            # Save mock config
            config = {'seed': 42}
            with open(run_dir / 'config.json', 'w') as f:
                json.dump(config, f)

            # Regenerate GIF
            result = regenerate_gif_from_run(
                run_dir=run_dir,
                config=VizConfig(gif_fps=1.0, gif_max_points=100),
            )

            assert result['gif_path'].exists()
            assert result['n_epochs'] == 3


# ============================================================================
# CLI Tests
# ============================================================================

@pytest.mark.unit
class TestPlaygroundConfigViz:
    """Test PlaygroundConfig visualization settings."""

    def test_default_viz_settings(self):
        """Test default visualization settings in PlaygroundConfig."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig()
        assert config.viz_mode == 'both'
        assert config.gif_every == 1
        assert config.gif_method == 'pca'
        assert config.gif_fps == 2.0

    def test_get_viz_config(self):
        """Test get_viz_config method."""
        from playground.playground_config import PlaygroundConfig

        pg_config = PlaygroundConfig(
            viz_mode='gif',
            gif_every=5,
            gif_method='tsne',
            gif_fps=1.0,
        )

        viz_config = pg_config.get_viz_config()

        from playground.viz.config import VizMode, GifMethod
        assert viz_config.viz_mode == VizMode.GIF
        assert viz_config.gif_every == 5
        assert viz_config.gif_method == GifMethod.TSNE
        assert viz_config.gif_fps == 1.0

    def test_should_generate_gif_property(self):
        """Test should_generate_gif property."""
        from playground.playground_config import PlaygroundConfig

        assert PlaygroundConfig(viz_mode='gif').should_generate_gif is True
        assert PlaygroundConfig(viz_mode='both').should_generate_gif is True
        assert PlaygroundConfig(viz_mode='static').should_generate_gif is False
        assert PlaygroundConfig(viz_mode='none').should_generate_gif is False

    def test_viz_settings_in_to_dict(self):
        """Test that viz settings are included in to_dict."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(
            viz_mode='gif',
            gif_every=5,
            gif_method='tsne',
        )

        d = config.to_dict()
        assert d['viz_mode'] == 'gif'
        assert d['gif_every'] == 5
        assert d['gif_method'] == 'tsne'

    def test_invalid_viz_mode(self):
        """Test validation for invalid viz_mode."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(viz_mode='invalid')
        with pytest.raises(ValueError, match="viz_mode must be one of"):
            config.validate()

    def test_invalid_gif_method(self):
        """Test validation for invalid gif_method."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(gif_method='invalid')
        with pytest.raises(ValueError, match="gif_method must be one of"):
            config.validate()


# ============================================================================
# Custom Dataset Tests
# ============================================================================

@pytest.mark.unit
class TestCustomDatasets:
    """Test custom dataset functionality."""

    def test_folder_dataset_two_classes(self):
        """Test FolderImageDataset with a 2-class folder structure."""
        from playground.custom_datasets import FolderImageDataset
        from PIL import Image
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2-class folder structure
            class_a_dir = Path(tmpdir) / "class_a"
            class_b_dir = Path(tmpdir) / "class_b"
            class_a_dir.mkdir()
            class_b_dir.mkdir()

            # Create dummy images (small 8x8 RGB images)
            for i in range(3):
                img = Image.new('RGB', (8, 8), color=(i * 50, i * 50, i * 50))
                img.save(class_a_dir / f"img_{i}.png")

            for i in range(2):
                img = Image.new('RGB', (8, 8), color=(100 + i * 50, 100 + i * 50, 100 + i * 50))
                img.save(class_b_dir / f"img_{i}.png")

            # Load dataset - should not raise any errors
            dataset = FolderImageDataset(
                root=tmpdir,
                contrastive=False,
                num_views=1,
            )

            # Verify dataset length > 0
            assert len(dataset) > 0, "Dataset should have at least one sample"
            assert len(dataset) == 5, f"Expected 5 samples (3+2), got {len(dataset)}"

            # Verify number of classes == 2
            assert len(dataset.classes) == 2, f"Expected 2 classes, got {len(dataset.classes)}"
            assert set(dataset.classes) == {"class_a", "class_b"}

            # Verify we can access a sample without error
            sample = dataset[0]
            assert 'image' in sample
            assert 'label' in sample
            assert 'index' in sample

    def test_folder_dataset_with_class_name_in_kwargs(self):
        """Test that get_custom_dataloaders doesn't fail when class_name is in kwargs for folder type."""
        from playground.custom_datasets import get_custom_dataloaders, FolderImageDataset
        from PIL import Image
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal 2-class folder structure
            class_a_dir = Path(tmpdir) / "class_a"
            class_b_dir = Path(tmpdir) / "class_b"
            class_a_dir.mkdir()
            class_b_dir.mkdir()

            # Create dummy images
            for i in range(5):
                img = Image.new('RGB', (8, 8), color=(i * 50, i * 50, i * 50))
                img.save(class_a_dir / f"img_{i}.png")
                img.save(class_b_dir / f"img_{i}.png")

            # This should NOT raise "unexpected keyword argument 'class_name'"
            train_loader, val_loader = get_custom_dataloaders(
                dataset_type='folder',
                path=tmpdir,
                batch_size=4,
                num_workers=0,
                contrastive=False,
                num_views=1,
                class_name="should_be_ignored",  # This was causing the bug
            )

            # Verify loaders work
            assert len(train_loader) > 0 or len(val_loader) > 0

    def test_folder_dataset_contrastive_mode(self):
        """Test FolderImageDataset returns multiple views in contrastive mode."""
        from playground.custom_datasets import FolderImageDataset
        from PIL import Image
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create single class folder
            class_dir = Path(tmpdir) / "class_a"
            class_dir.mkdir()

            # Create dummy images
            for i in range(3):
                img = Image.new('RGB', (8, 8), color=(i * 50, i * 50, i * 50))
                img.save(class_dir / f"img_{i}.png")

            # Load in contrastive mode
            dataset = FolderImageDataset(
                root=tmpdir,
                contrastive=True,
                num_views=2,
            )

            sample = dataset[0]
            assert 'image' in sample
            assert 'image1' in sample, "Contrastive mode should return 'image1'"
            assert 'label' in sample
            assert 'index' in sample


    def test_custom_dataset_config_shows_custom_not_cifar(self):
        """Test that PlaygroundConfig with custom_dataset shows CUSTOM in describe(), not CIFAR."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(
            dataset="custom",
            backbone="resnet18",
            icon_mode="simclr_like",
            custom_dataset_type="folder",
            custom_dataset_path="/path/to/data",
        )

        description = config.describe()

        # Should show "CUSTOM (folder)", NOT "CIFAR10"
        assert "CUSTOM (folder)" in description, f"Expected 'CUSTOM (folder)' in description, got:\n{description}"
        assert "CIFAR" not in description, f"Description should not mention CIFAR:\n{description}"
        assert "MNIST" not in description, f"Description should not mention MNIST:\n{description}"

    def test_custom_dataset_allows_simplecnn_backbone(self):
        """Test that custom dataset allows simplecnn backbone (bypasses MNIST-only check)."""
        from playground.playground_config import PlaygroundConfig

        # This should NOT raise "Backbone 'simplecnn' is designed for MNIST only"
        config = PlaygroundConfig(
            dataset="custom",
            backbone="simplecnn",
            icon_mode="simclr_like",
            custom_dataset_type="folder",
            custom_dataset_path="/path/to/data",
        )

        # Should pass validation
        config.validate()

        description = config.describe()
        assert "simplecnn" in description
        assert "CUSTOM (folder)" in description

    def test_custom_dataset_run_name_prefix(self):
        """Test that custom dataset uses 'custom_dataset' prefix in run_name."""
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(
            dataset="custom",
            backbone="resnet18",
            icon_mode="simclr_like",
            custom_dataset_type="embeddings",
            custom_dataset_path="/path/to/embeddings.npz",
        )

        # run_name should use "custom_dataset" prefix, not literal "custom"
        assert config.run_name.startswith("custom_dataset_"), f"Expected run_name to start with 'custom_dataset_', got: {config.run_name}"
        assert "custom_dataset_resnet18_simclr_like" == config.run_name


@pytest.mark.unit
class TestSNELossShapes:
    """Test SNE/t-SNE loss shape correctness."""

    def test_distance_kernel_rejects_non_2d_input(self):
        """Test that DistanceKernel raises error for non-2D input (flatten must happen upstream)."""
        import torch
        from distributions.distance import Gaussian

        batch_size = 16

        gaussian = Gaussian(
            sigma=0.5,
            metric="euclidean",
            input_key="embedding",
            mask_diagonal=True,
            normalize=True
        )

        # 2D input should work
        embeddings_2d = torch.randn(batch_size, 128)
        q = gaussian._compute(embeddings_2d)
        assert q.shape == (batch_size, batch_size)

        # 4D input should raise error (flatten must happen in Model._compute_loss)
        embeddings_4d = torch.randn(batch_size, 3, 8, 8)
        with pytest.raises(ValueError, match="requires 2D tensors"):
            gaussian._compute(embeddings_4d)

    def test_model_compute_loss_flattens_embeddings(self):
        """Test that Model._compute_loss flattens 4D embeddings to 2D before distribution computation."""
        import torch
        # Use playground config to avoid circular imports (establishes correct import order)
        from playground.playground_config import PlaygroundConfig

        batch_size = 8
        embedding_dim = 64

        # Create config via PlaygroundConfig to avoid circular import issues
        pg_config = PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="sne_like",  # Use sne_like to test the fix
            embedding_dim=embedding_dim,
        )
        config = pg_config.to_icon_config()

        from model.kernel_model import Model
        model = Model(config)
        model.eval()

        # Create a test batch with proper image dimensions
        batch = {
            "image": torch.randn(batch_size, 3, 32, 32),
            "index": torch.arange(batch_size),
            "label": torch.zeros(batch_size, dtype=torch.long),
        }

        # Run compute_loss - this should not raise any errors
        with torch.no_grad():
            result = model._compute_loss(batch)

        # Verify embeddings in batch are 2D after _compute_loss
        assert batch["embedding"].dim() == 2, \
            f"Embedding should be 2D after _compute_loss, got shape {batch['embedding'].shape}"
        assert batch["embedding"].shape == (batch_size, embedding_dim), \
            f"Expected embedding shape ({batch_size}, {embedding_dim}), got {batch['embedding'].shape}"

        # Verify learned_distribution (q) has correct shape
        q = result["metrics"]["learned_distribution"]
        assert q.shape == (batch_size, batch_size), \
            f"q should have shape ({batch_size}, {batch_size}), got {q.shape}"

    def test_all_icon_modes_receive_2d_embeddings(self):
        """Test that all I-Con modes (simclr_like, sne_like, tsne_like) receive 2D embeddings."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        for icon_mode in ["simclr_like", "sne_like", "tsne_like"]:
            config = PlaygroundConfig(
                dataset="cifar10",
                backbone="resnet18",
                icon_mode=icon_mode,
                embedding_dim=64,
            )

            icon_config = config.to_icon_config()
            from model.kernel_model import Model
            model = Model(icon_config)
            model.eval()

            # Create test batch
            batch = {
                "image": torch.randn(batch_size, 3, 32, 32),
                "index": torch.arange(batch_size),
                "label": torch.zeros(batch_size, dtype=torch.long),
            }

            with torch.no_grad():
                result = model._compute_loss(batch)

            # Verify embedding is 2D
            assert batch["embedding"].dim() == 2, \
                f"[{icon_mode}] Embedding should be 2D, got shape {batch['embedding'].shape}"

            # Verify q has shape (B, B)
            q = result["metrics"]["learned_distribution"]
            assert q.shape == (batch_size, batch_size), \
                f"[{icon_mode}] q should have shape ({batch_size}, {batch_size}), got {q.shape}"

    def test_p_and_q_shapes_match(self):
        """Test that p and q have matching shapes for contrastive learning."""
        import torch
        from distributions.distance import Gaussian
        from distributions.graph import Augmentation

        batch_size = 16

        # Create supervisory distribution (p)
        augmentation = Augmentation(
            input_key="index",
            mask_diagonal=True,
            normalize=True
        )

        # Create learned distribution (q)
        gaussian = Gaussian(
            sigma=0.5,
            metric="euclidean",
            input_key="embedding",
            mask_diagonal=True,
            normalize=True
        )

        # Simulate batch with 2D embeddings (as required)
        batch = {
            "index": torch.arange(batch_size),
            "embedding": torch.randn(batch_size, 128),
        }

        p = augmentation(batch)
        q = gaussian(batch)

        assert p.shape == q.shape, \
            f"p and q shapes must match! p: {p.shape}, q: {q.shape}"
        assert p.shape == (batch_size, batch_size), \
            f"Both p and q should be ({batch_size}, {batch_size}), got {p.shape}"

    def test_simclr_with_4d_backbone_output_produces_correct_shapes(self):
        """Test that SimCLR with 4D backbone output produces p and q of shape (B, B)."""
        import torch
        import torch.nn as nn
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        # Create a simclr_like config
        config = PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="simclr_like",
            embedding_dim=64,
        )
        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)
        model.eval()

        # Create test batch
        batch = {
            "image": torch.randn(batch_size, 3, 32, 32),
            "index": torch.arange(batch_size),
            "label": torch.zeros(batch_size, dtype=torch.long),
        }

        with torch.no_grad():
            result = model._compute_loss(batch)

        p = result["metrics"]["supervisory_distribution"]
        q = result["metrics"]["learned_distribution"]

        # Verify both p and q have shape (B, B)
        assert p.shape == (batch_size, batch_size), \
            f"SimCLR p should have shape ({batch_size}, {batch_size}), got {p.shape}"
        assert q.shape == (batch_size, batch_size), \
            f"SimCLR q should have shape ({batch_size}, {batch_size}), got {q.shape}"
        assert p.shape == q.shape, \
            f"SimCLR p and q must match: p={p.shape}, q={q.shape}"

    def test_custom_folder_simplecnn_simclr_no_shape_mismatch(self):
        """Test that custom folder dataset + simplecnn + simclr_like runs without shape mismatch."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        # Create config for custom folder + simplecnn + simclr_like
        config = PlaygroundConfig(
            dataset="custom",
            backbone="simplecnn",
            icon_mode="simclr_like",
            embedding_dim=64,
            custom_dataset_type="folder",
            custom_dataset_path="/tmp/test",  # Path doesn't need to exist for this test
        )

        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)
        model.eval()

        # SimpleCNN expects 28x28 grayscale images
        batch = {
            "image": torch.randn(batch_size, 1, 28, 28),
            "index": torch.arange(batch_size),
            "label": torch.zeros(batch_size, dtype=torch.long),
        }

        # This should NOT raise any shape mismatch errors
        with torch.no_grad():
            result = model._compute_loss(batch)

        # Verify embedding is 2D
        assert batch["embedding"].dim() == 2, \
            f"Embedding should be 2D, got {batch['embedding'].dim()}D"
        assert batch["embedding"].shape == (batch_size, 64), \
            f"Embedding should be ({batch_size}, 64), got {batch['embedding'].shape}"

        # Verify p and q have correct shapes
        p = result["metrics"]["supervisory_distribution"]
        q = result["metrics"]["learned_distribution"]

        assert p.shape == (batch_size, batch_size), \
            f"p should have shape ({batch_size}, {batch_size}), got {p.shape}"
        assert q.shape == (batch_size, batch_size), \
            f"q should have shape ({batch_size}, {batch_size}), got {q.shape}"
        assert p.shape == q.shape, \
            f"p and q shapes must match: p={p.shape}, q={q.shape}"

    def test_simclr_cnn_custom_dataset_produces_correct_q_shape(self):
        """Test that SimCLR + CNN backbone + custom dataset produces q with shape (B, B)."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 16

        # Test with ResNet (CNN) backbone and custom dataset
        config = PlaygroundConfig(
            dataset="custom",
            backbone="resnet18",
            icon_mode="simclr_like",
            embedding_dim=128,
            custom_dataset_type="folder",
            custom_dataset_path="/tmp/test",
        )

        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)
        model.eval()

        # Custom folder uses 224x224 RGB images by default
        batch = {
            "image": torch.randn(batch_size, 3, 224, 224),
            "index": torch.arange(batch_size),
            "label": torch.zeros(batch_size, dtype=torch.long),
        }

        with torch.no_grad():
            result = model._compute_loss(batch)

        # Verify q has shape (B, B), not (B*C*H*W, B*C*H*W)
        q = result["metrics"]["learned_distribution"]
        assert q.shape == (batch_size, batch_size), \
            f"SimCLR q must have shape ({batch_size}, {batch_size}), got {q.shape}. " \
            f"This indicates raw backbone feature maps were used instead of projection head output."

    def test_simclr_never_receives_tensors_with_dim_greater_than_2(self):
        """Test that SimCLR (and all I-Con objectives) never receive tensors with dim > 2."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        for icon_mode in ["simclr_like", "sne_like", "tsne_like"]:
            config = PlaygroundConfig(
                dataset="cifar10",
                backbone="resnet18",
                icon_mode=icon_mode,
                embedding_dim=64,
            )

            icon_config = config.to_icon_config()

            from model.kernel_model import Model
            model = Model(icon_config)
            model.eval()

            batch = {
                "image": torch.randn(batch_size, 3, 32, 32),
                "index": torch.arange(batch_size),
                "label": torch.zeros(batch_size, dtype=torch.long),
            }

            with torch.no_grad():
                result = model._compute_loss(batch)

            # After _compute_loss, batch["embedding"] must be 2D
            z = batch["embedding"]
            assert z.dim() == 2, \
                f"[{icon_mode}] Embedding must be 2D after _compute_loss, got {z.dim()}D with shape {z.shape}"

            # q must have shape (B, B)
            q = result["metrics"]["learned_distribution"]
            assert q.dim() == 2, \
                f"[{icon_mode}] q must be 2D, got {q.dim()}D"
            assert q.shape[0] == batch_size and q.shape[1] == batch_size, \
                f"[{icon_mode}] q must have shape ({batch_size}, {batch_size}), got {q.shape}"


@pytest.mark.unit
class TestIConInvariants:
    """
    Regression tests that make I-Con shape mismatch bugs PROVABLY IMPOSSIBLE.

    These tests enforce the absolute invariants:
    1) All I-Con objectives operate ONLY on (B, D) embeddings
    2) p and q MUST ALWAYS have shape (B, B)
    3) No learned_distribution may EVER see tensors with dim > 2
    4) Violations fail fast with clear error messages
    """

    def test_kernel_base_rejects_non_2d_tensor(self):
        """Test that DistanceKernel rejects non-2D tensors with clear error."""
        import torch
        from distributions.distance import Gaussian

        batch_size = 8
        gaussian = Gaussian(
            sigma=0.5,
            metric="euclidean",
            input_key="embedding",
            mask_diagonal=True,
            normalize=True
        )

        # 4D tensor should raise ValueError
        batch_4d = {"embedding": torch.randn(batch_size, 3, 8, 8)}

        with pytest.raises(ValueError, match="requires 2D tensors"):
            gaussian(batch_4d)

    def test_distance_kernel_rejects_non_2d_directly(self):
        """Test that DistanceKernel rejects non-2D tensors directly."""
        import torch
        from distributions.distance import DistanceKernel

        kernel = DistanceKernel(metric="euclidean")

        # Direct call with 4D tensor
        x_4d = torch.randn(8, 3, 8, 8)

        with pytest.raises(ValueError, match="DistanceKernel requires 2D tensors"):
            kernel._compute(x_4d)

    def test_model_compute_loss_rejects_forbidden_keys(self):
        """Test that _compute_loss rejects forbidden keys like 'features'."""
        import torch
        import torch.nn as nn
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="simclr_like",
            embedding_dim=64,
        )
        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)

        # Patch mapper to return forbidden key
        original_mapper = model.mapper

        class BadMapper(nn.Module):
            def __init__(self, real_mapper):
                super().__init__()
                self.real_mapper = real_mapper
                self.output_dim = real_mapper.output_dim

            def forward(self, batch):
                result = self.real_mapper(batch)
                result['features'] = torch.randn(8, 3, 8, 8)  # Forbidden!
                return result

            def parameters(self):
                return self.real_mapper.parameters()

        model.mapper = BadMapper(original_mapper)

        batch = {
            "image": torch.randn(8, 3, 32, 32),
            "index": torch.arange(8),
            "label": torch.zeros(8, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="forbidden key"):
            model._compute_loss(batch)

    def test_model_compute_loss_requires_embedding_key(self):
        """Test that _compute_loss requires 'embedding' key from mapper."""
        import torch
        import torch.nn as nn
        from playground.playground_config import PlaygroundConfig

        config = PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="simclr_like",
            embedding_dim=64,
        )
        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)

        # Patch mapper to NOT return embedding key
        original_mapper = model.mapper

        class NoEmbeddingMapper(nn.Module):
            def __init__(self, real_mapper):
                super().__init__()
                self.real_mapper = real_mapper
                self.output_dim = real_mapper.output_dim

            def forward(self, batch):
                return {'wrong_key': torch.randn(8, 64)}

            def parameters(self):
                return self.real_mapper.parameters()

        model.mapper = NoEmbeddingMapper(original_mapper)

        batch = {
            "image": torch.randn(8, 3, 32, 32),
            "index": torch.arange(8),
            "label": torch.zeros(8, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="Mapper must return 'embedding' key"):
            model._compute_loss(batch)

    def test_all_icon_modes_enforce_p_equals_q_shape(self):
        """Test that ALL icon modes produce p.shape == q.shape == (B, B)."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 16

        for icon_mode in ["simclr_like", "sne_like", "tsne_like"]:
            for backbone in ["resnet18", "simplecnn"]:
                if backbone == "simplecnn":
                    dataset = "mnist"
                    image_shape = (batch_size, 1, 28, 28)
                else:
                    dataset = "cifar10"
                    image_shape = (batch_size, 3, 32, 32)

                config = PlaygroundConfig(
                    dataset=dataset,
                    backbone=backbone,
                    icon_mode=icon_mode,
                    embedding_dim=64,
                )

                icon_config = config.to_icon_config()

                from model.kernel_model import Model
                model = Model(icon_config)
                model.eval()

                batch = {
                    "image": torch.randn(*image_shape),
                    "index": torch.arange(batch_size),
                    "label": torch.zeros(batch_size, dtype=torch.long),
                }

                with torch.no_grad():
                    result = model._compute_loss(batch)

                p = result["metrics"]["supervisory_distribution"]
                q = result["metrics"]["learned_distribution"]

                # INVARIANT: p.shape == q.shape == (B, B)
                assert p.shape == q.shape, \
                    f"[{icon_mode}+{backbone}] INVARIANT VIOLATED: p.shape={p.shape} != q.shape={q.shape}"
                assert p.shape == (batch_size, batch_size), \
                    f"[{icon_mode}+{backbone}] INVARIANT VIOLATED: expected ({batch_size},{batch_size}), got p={p.shape}"

    def test_q_shape_never_bchw_bchw(self):
        """Test that q NEVER has shape (B*C*H*W, B*C*H*W) - the fatal bug pattern."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        for icon_mode in ["simclr_like", "sne_like", "tsne_like"]:
            config = PlaygroundConfig(
                dataset="custom",
                backbone="resnet18",
                icon_mode=icon_mode,
                embedding_dim=128,
                custom_dataset_type="folder",
                custom_dataset_path="/tmp/test",
            )

            icon_config = config.to_icon_config()

            from model.kernel_model import Model
            model = Model(icon_config)
            model.eval()

            # Use larger images to make the bug more obvious if it exists
            batch = {
                "image": torch.randn(batch_size, 3, 224, 224),
                "index": torch.arange(batch_size),
                "label": torch.zeros(batch_size, dtype=torch.long),
            }

            with torch.no_grad():
                result = model._compute_loss(batch)

            q = result["metrics"]["learned_distribution"]

            # The bug pattern: q.shape[0] == B * C * H * W instead of B
            # For 224x224 RGB: B*C*H*W = 8*3*224*224 = 1204224 (huge!)
            # For correct: q.shape == (8, 8)
            assert q.shape[0] == batch_size, \
                f"[{icon_mode}] FATAL BUG: q.shape[0]={q.shape[0]} != batch_size={batch_size}. " \
                f"This indicates feature maps were used instead of embeddings!"
            assert q.shape == (batch_size, batch_size), \
                f"[{icon_mode}] q must be ({batch_size}, {batch_size}), got {q.shape}"

    def test_custom_folder_resnet_simclr_produces_correct_q(self):
        """Regression test: CNN backbone + simclr_like + custom folder = (B, B) q."""
        import torch
        from playground.playground_config import PlaygroundConfig

        batch_size = 8

        config = PlaygroundConfig(
            dataset="custom",
            backbone="resnet18",
            icon_mode="simclr_like",
            embedding_dim=128,
            custom_dataset_type="folder",
            custom_dataset_path="/tmp/test",
        )

        icon_config = config.to_icon_config()

        from model.kernel_model import Model
        model = Model(icon_config)
        model.eval()

        batch = {
            "image": torch.randn(batch_size, 3, 224, 224),
            "index": torch.arange(batch_size),
            "label": torch.zeros(batch_size, dtype=torch.long),
        }

        with torch.no_grad():
            result = model._compute_loss(batch)

        p = result["metrics"]["supervisory_distribution"]
        q = result["metrics"]["learned_distribution"]

        assert p.shape == (batch_size, batch_size)
        assert q.shape == (batch_size, batch_size)
        assert p.shape == q.shape

    def test_embedding_always_2d_after_compute_loss(self):
        """Test that batch['embedding'] is ALWAYS 2D after _compute_loss."""
        import torch
        from playground.playground_config import PlaygroundConfig

        for icon_mode in ["simclr_like", "sne_like", "tsne_like"]:
            for backbone in ["resnet18", "simplecnn"]:
                if backbone == "simplecnn":
                    dataset = "mnist"
                    image_shape = (8, 1, 28, 28)
                else:
                    dataset = "cifar10"
                    image_shape = (8, 3, 32, 32)

                config = PlaygroundConfig(
                    dataset=dataset,
                    backbone=backbone,
                    icon_mode=icon_mode,
                    embedding_dim=64,
                )

                icon_config = config.to_icon_config()

                from model.kernel_model import Model
                model = Model(icon_config)
                model.eval()

                batch = {
                    "image": torch.randn(*image_shape),
                    "index": torch.arange(8),
                    "label": torch.zeros(8, dtype=torch.long),
                }

                with torch.no_grad():
                    model._compute_loss(batch)

                z = batch["embedding"]
                assert z.dim() == 2, \
                    f"[{icon_mode}+{backbone}] embedding must be 2D, got {z.dim()}D with shape {z.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
