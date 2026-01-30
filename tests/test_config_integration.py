"""
Tests for playground config integration with divergences and optimizers.

These tests verify:
1. PlaygroundConfig accepts and validates divergence parameters
2. PlaygroundConfig accepts and validates optimizer parameters
3. Config properly passes these to I-Con Config
4. Invalid values are rejected
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.playground_config import PlaygroundConfig, SUPPORTED_DIVERGENCES


class TestConfigIntegration:
    """Test suite for config integration."""

    def test_default_divergence(self):
        """Test that default divergence is 'kl'."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
        )
        assert config.divergence == "kl"

    def test_custom_divergence_kl(self):
        """Test KL divergence configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="kl",
        )
        icon_config = config.to_icon_config()
        assert icon_config.divergence == "kl"

    def test_custom_divergence_rkl(self):
        """Test reverse KL divergence configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="rkl",
        )
        icon_config = config.to_icon_config()
        assert icon_config.divergence == "rkl"

    def test_custom_divergence_js(self):
        """Test Jensen-Shannon divergence configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="js",
        )
        icon_config = config.to_icon_config()
        assert icon_config.divergence == "js"

    def test_divergence_case_insensitive(self):
        """Test that divergence names are case-insensitive."""
        config1 = PlaygroundConfig(dataset="mnist", backbone="mlp", divergence="KL")
        config2 = PlaygroundConfig(dataset="mnist", backbone="mlp", divergence="Kl")
        config3 = PlaygroundConfig(dataset="mnist", backbone="mlp", divergence="kl")

        assert config1.divergence == config2.divergence == config3.divergence == "kl"

    def test_invalid_divergence_raises_error(self):
        """Test that invalid divergence raises ValueError."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="invalid",
        )

        with pytest.raises(ValueError, match="Unsupported divergence"):
            config.to_icon_config()

    def test_default_optimizer(self):
        """Test that default optimizer is 'adamw'."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
        )
        assert config.optimizer == "adamw"

    def test_custom_optimizer_adam(self):
        """Test Adam optimizer configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            optimizer="adam",
        )
        icon_config = config.to_icon_config()
        assert icon_config.optimizer == "adam"

    def test_custom_optimizer_sgd(self):
        """Test SGD optimizer configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            optimizer="sgd",
        )
        icon_config = config.to_icon_config()
        assert icon_config.optimizer == "sgd"

    def test_invalid_optimizer_raises_error(self):
        """Test that invalid optimizer raises ValueError."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            optimizer="invalid",
        )

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            config.to_icon_config()

    def test_custom_weight_decay(self):
        """Test custom weight decay configuration."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            weight_decay=0.01,
        )
        icon_config = config.to_icon_config()
        assert icon_config.weight_decay == 0.01

    def test_config_to_dict_includes_new_fields(self):
        """Test that to_dict includes divergence, optimizer, and weight_decay."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="rkl",
            optimizer="adam",
            weight_decay=0.001,
        )

        config_dict = config.to_dict()

        assert "divergence" in config_dict
        assert "optimizer" in config_dict
        assert "weight_decay" in config_dict
        assert config_dict["divergence"] == "rkl"
        assert config_dict["optimizer"] == "adam"
        assert config_dict["weight_decay"] == 0.001

    def test_all_divergences_supported(self):
        """Test that all supported divergences work."""
        for div in SUPPORTED_DIVERGENCES:
            config = PlaygroundConfig(
                dataset="mnist",
                backbone="mlp",
                icon_mode="simclr_like",
                divergence=div,
            )
            icon_config = config.to_icon_config()
            assert icon_config.divergence == div

    def test_combined_divergence_and_optimizer(self):
        """Test that divergence and optimizer can be set together."""
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            divergence="js",
            optimizer="sgd",
            weight_decay=0.0001,
        )

        icon_config = config.to_icon_config()

        assert icon_config.divergence == "js"
        assert icon_config.optimizer == "sgd"
        assert icon_config.weight_decay == 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
