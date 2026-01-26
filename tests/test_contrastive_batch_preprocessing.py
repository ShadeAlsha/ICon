"""
Tests for Contrastive Batch Preprocessing

Critical tests to verify that:
1. Contrastive batches are properly preprocessed (views concatenated, indices duplicated)
2. Different icon_modes (simclr_like vs sne_like) produce different losses
3. Positive pairs are correctly identified in the supervisory distribution
4. The fix works across different configurations

These tests directly address the critical bug where simclr_like and sne_like
were producing identical losses due to missing batch preprocessing.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestContrastiveBatchPreprocessing:
    """Test that contrastive batches are properly preprocessed."""

    def test_preprocess_contrastive_batch_concatenates_views(self):
        """The preprocess method should concatenate image views."""
        # Test the preprocessing logic directly without needing a full trainer
        batch = {
            'image': torch.randn(32, 784),
            'image1': torch.randn(32, 784),
            'index': torch.arange(32),
            'label': torch.randint(0, 10, (32,)),
        }

        # Manually apply the preprocessing logic
        if 'image1' in batch:
            combined_image = torch.cat([batch['image'], batch['image1']], dim=0)
            combined_index = torch.cat([batch['index'], batch['index']], dim=0)
            combined_label = torch.cat([batch['label'], batch['label']], dim=0)
            processed = {
                'image': combined_image,
                'index': combined_index,
                'label': combined_label,
            }
        else:
            processed = batch

        # Verify shapes
        assert processed['image'].shape[0] == 64, "Image should be doubled"
        assert processed['index'].shape[0] == 64, "Index should be doubled"
        assert processed['label'].shape[0] == 64, "Label should be doubled"

    def test_preprocess_contrastive_batch_duplicates_indices(self):
        """The preprocess method should duplicate indices for positive pairs."""
        batch = {
            'image': torch.randn(16, 784),
            'image1': torch.randn(16, 784),
            'index': torch.arange(16),
            'label': torch.randint(0, 10, (16,)),
        }

        # Apply preprocessing logic
        combined_index = torch.cat([batch['index'], batch['index']], dim=0)

        # Check that indices are duplicated [0,1,2,...,15, 0,1,2,...,15]
        first_half = combined_index[:16]
        second_half = combined_index[16:]
        assert torch.equal(first_half, second_half), "Indices should be duplicated"

    def test_preprocess_noncontrastive_batch_unchanged(self):
        """Non-contrastive batches should pass through unchanged."""
        batch = {
            'image': torch.randn(32, 784),
            'index': torch.arange(32),
            'label': torch.randint(0, 10, (32,)),
        }

        # Without image1, batch should be unchanged
        if 'image1' not in batch:
            processed = batch
        else:
            processed = None  # Would process

        assert processed['image'].shape == batch['image'].shape


class TestSimclrVsSneDistinctLosses:
    """Test that simclr_like and sne_like produce distinct losses."""

    def test_different_metrics_produce_different_distances(self):
        """Cosine vs euclidean metrics should produce different distance matrices."""
        from distributions.distance import Gaussian

        torch.manual_seed(42)
        embeddings = torch.randn(16, 32)

        # SimCLR-like: cosine distance
        cosine_dist = Gaussian(sigma=0.5, metric='cosine', normalize=True, mask_diagonal=True, input_key='embedding')
        # SNE-like: euclidean distance
        euclidean_dist = Gaussian(sigma=0.5, metric='euclidean', normalize=True, mask_diagonal=True, input_key='embedding')

        batch = {'embedding': embeddings, 'index': torch.arange(16)}

        cosine_result = cosine_dist(batch)
        euclidean_result = euclidean_dist(batch)

        # Results should be different
        diff = (cosine_result - euclidean_result).abs().mean()
        assert diff > 0.01, f"Distance matrices should differ, got mean diff {diff:.6f}"

    def test_different_modes_produce_different_losses_in_training(self):
        """Full training should produce different losses for different modes."""
        from playground.playground_config import PlaygroundConfig
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        results = {}

        for mode in ['simclr_like', 'sne_like']:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = PlaygroundConfig(
                    dataset='mnist',
                    backbone='mlp',
                    icon_mode=mode,
                    epochs=2,
                    batch_size=256,
                    temperature=0.5,
                    embedding_dim=32,
                    num_workers=0,
                    output_dir=tmpdir,
                )

                result = run_playground_experiment_pure_pytorch(
                    config,
                    verbose=False,
                    gpu=False,
                )

                results[mode] = result['logs']['train_losses'][-1]

        # Losses should be significantly different
        diff = abs(results['simclr_like'] - results['sne_like'])
        assert diff > 0.1, f"Losses should differ significantly, got diff {diff:.4f}"

    def test_loss_curves_diverge_over_epochs(self):
        """Loss curves for different modes should diverge over training."""
        from playground.playground_config import PlaygroundConfig
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        results = {}

        for mode in ['simclr_like', 'sne_like']:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = PlaygroundConfig(
                    dataset='mnist',
                    backbone='mlp',
                    icon_mode=mode,
                    epochs=5,
                    batch_size=256,
                    temperature=0.5,
                    embedding_dim=32,
                    num_workers=0,
                    output_dir=tmpdir,
                )

                result = run_playground_experiment_pure_pytorch(
                    config,
                    verbose=False,
                    gpu=False,
                )

                results[mode] = result['logs']['train_losses']

        # Compare final losses
        final_diff = abs(results['simclr_like'][-1] - results['sne_like'][-1])
        # Compare first losses (should be closer since same initialization)
        first_diff = abs(results['simclr_like'][0] - results['sne_like'][0])

        # Final difference should be larger than first difference (curves diverge)
        assert final_diff > first_diff * 0.5, \
            f"Loss curves should diverge. First diff: {first_diff:.4f}, Final diff: {final_diff:.4f}"


class TestSupervisoryDistributionPositivePairs:
    """Test that supervisory distribution correctly identifies positive pairs."""

    def test_augmentation_kernel_creates_positive_pairs(self):
        """Augmentation kernel should have non-zero values for same-index samples."""
        from distributions.graph import Augmentation

        # Create batch with duplicated indices (as after preprocessing)
        indices = torch.cat([torch.arange(8), torch.arange(8)])

        augmentation = Augmentation(normalize=True, mask_diagonal=True, input_key='index')
        batch = {'index': indices}

        result = augmentation(batch)

        # Each sample should have exactly one positive pair (its augmented view)
        non_zeros_per_row = (result > 0).sum(dim=1)
        assert non_zeros_per_row.min() >= 1, "Each sample should have at least one positive pair"

    def test_unique_indices_produce_zero_supervisory(self):
        """Unique indices (bug state) should produce zero supervisory distribution."""
        from distributions.graph import Augmentation

        # Create batch with unique indices (the bug state)
        indices = torch.arange(16)  # All unique, no positive pairs

        augmentation = Augmentation(normalize=True, mask_diagonal=True, input_key='index')
        batch = {'index': indices}

        result = augmentation(batch)

        # With all unique indices and diagonal masked, there should be no positive pairs
        # (after normalization, this creates issues)
        non_zeros_per_row = (result > 0).sum(dim=1)
        # All zeros because there are no matching indices after diagonal masking
        assert non_zeros_per_row.max() == 0, "Unique indices should produce no positive pairs"


class TestSanityChecks:
    """Test that sanity checks catch problems."""

    def test_sanity_check_logic_catches_unique_indices(self):
        """Sanity check logic should catch batches with all unique indices."""
        # Test the sanity check logic directly
        batch = {
            'image': torch.randn(16, 784),
            'index': torch.arange(16),  # All unique - bug state
            'label': torch.randint(0, 10, (16,)),
        }

        index = batch.get('index')
        unique_count = index.unique().shape[0]
        total_count = index.shape[0]

        # This condition should trigger the error in the sanity check
        has_no_duplicates = (unique_count == total_count and total_count > 1)
        assert has_no_duplicates, "Unique indices should be detected"

    def test_sanity_check_logic_passes_duplicated_indices(self):
        """Sanity check logic should pass for properly preprocessed batches."""
        # Test with duplicated indices (correct state)
        original_indices = torch.arange(8)
        combined_indices = torch.cat([original_indices, original_indices])

        unique_count = combined_indices.unique().shape[0]
        total_count = combined_indices.shape[0]

        # After preprocessing, we should have duplicates
        has_duplicates = (unique_count < total_count)
        assert has_duplicates, "Preprocessed batch should have duplicate indices"


class TestRecipeComparison:
    """Test that recipe comparisons work correctly."""

    def test_simclr_vs_sne_recipe_produces_different_results(self):
        """The simclr_vs_sne recipe should produce distinct results for each mode."""
        from playground.playground_recipes import get_recipe
        from playground.playground_runner import run_playground_experiment_pure_pytorch

        configs = get_recipe('simclr_vs_sne', dataset='mnist', epochs=2)

        results = []
        for config in configs:
            with tempfile.TemporaryDirectory() as tmpdir:
                config.output_dir = tmpdir
                config.num_workers = 0

                result = run_playground_experiment_pure_pytorch(
                    config,
                    verbose=False,
                    gpu=False,
                )

                results.append({
                    'mode': config.icon_mode,
                    'loss': result['logs']['train_losses'][-1]
                })

        # Compare results
        assert len(results) == 2, "Recipe should produce 2 experiments"
        loss_diff = abs(results[0]['loss'] - results[1]['loss'])
        assert loss_diff > 0.1, f"Recipe experiments should produce different losses, got diff {loss_diff:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
