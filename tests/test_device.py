"""
Tests for device selection module.

These tests verify device selection logic without requiring actual GPU hardware.
They use mocking to simulate different hardware configurations.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from playground.device import select_device, get_dataloader_config


class TestDeviceSelection:
    """Test device selection with different hardware configurations."""

    def test_explicit_cpu_selection(self):
        """Test that device='cpu' always returns CPU."""
        device = select_device("cpu", verbose=False)
        assert device.type == "cpu"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_auto_selects_cpu_when_no_gpu(self, mock_mps, mock_cuda):
        """Test that device='auto' selects CPU when no GPU is available."""
        device = select_device("auto", verbose=False)
        assert device.type == "cpu"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Mock GPU")
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_auto_selects_cuda_when_available(self, mock_mps, mock_name, mock_cuda):
        """Test that device='auto' prefers CUDA over CPU."""
        device = select_device("auto", verbose=False)
        assert device.type == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_auto_selects_mps_when_cuda_unavailable(self, mock_mps, mock_cuda):
        """Test that device='auto' selects MPS when CUDA is unavailable."""
        device = select_device("auto", verbose=False)
        assert device.type == "mps"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Mock GPU")
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_auto_prefers_cuda_over_mps(self, mock_mps, mock_name, mock_cuda):
        """Test that device='auto' prefers CUDA over MPS."""
        device = select_device("auto", verbose=False)
        assert device.type == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_fails_when_unavailable(self, mock_cuda):
        """Test that device='cuda' raises error when CUDA is unavailable."""
        with pytest.raises(RuntimeError, match="CUDA GPU requested.*but CUDA is not available"):
            select_device("cuda", verbose=False)

    @patch('torch.backends.mps.is_available', return_value=False)
    def test_mps_fails_when_unavailable(self, mock_mps):
        """Test that device='mps' raises error when MPS is unavailable."""
        with pytest.raises(RuntimeError, match="Apple Silicon GPU requested.*but MPS is not available"):
            select_device("mps", verbose=False)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Tesla V100")
    def test_explicit_cuda_selection(self, mock_name, mock_cuda):
        """Test that device='cuda' returns CUDA when available."""
        device = select_device("cuda", verbose=False)
        assert device.type == "cuda"

    @patch('torch.backends.mps.is_available', return_value=True)
    def test_explicit_mps_selection(self, mock_mps):
        """Test that device='mps' returns MPS when available."""
        device = select_device("mps", verbose=False)
        assert device.type == "mps"

    def test_invalid_device_raises_error(self):
        """Test that invalid device string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            select_device("gpu", verbose=False)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="RTX 3090")
    def test_verbose_output_cuda(self, mock_name, mock_cuda, capsys):
        """Test that verbose=True prints device info for CUDA."""
        select_device("cuda", verbose=True)
        captured = capsys.readouterr()
        assert "Using device: cuda" in captured.out
        assert "RTX 3090" in captured.out

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_verbose_output_cpu(self, mock_mps, mock_cuda, capsys):
        """Test that verbose=True prints device info for CPU."""
        select_device("auto", verbose=True)
        captured = capsys.readouterr()
        assert "Using device: cpu" in captured.out


class TestDataLoaderConfig:
    """Test DataLoader configuration based on device type."""

    def test_cuda_enables_pin_memory(self):
        """Test that CUDA device enables pin_memory and persistent_workers."""
        device = torch.device("cuda")
        config = get_dataloader_config(device)
        assert config["pin_memory"] is True
        assert config["persistent_workers"] is True

    def test_cpu_disables_pin_memory(self):
        """Test that CPU device disables pin_memory."""
        device = torch.device("cpu")
        config = get_dataloader_config(device)
        assert config["pin_memory"] is False
        assert "persistent_workers" not in config

    def test_mps_disables_pin_memory(self):
        """Test that MPS device disables pin_memory."""
        device = torch.device("mps")
        config = get_dataloader_config(device)
        assert config["pin_memory"] is False
        assert "persistent_workers" not in config


class TestBackwardCompatibility:
    """Test backward compatibility with old device selection patterns."""

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_auto_equivalent_to_none(self, mock_mps, mock_cuda):
        """Test that device='auto' behaves like old gpu=None behavior."""
        device = select_device("auto", verbose=False)
        assert device.type == "cpu"

    def test_cpu_equivalent_to_gpu_false(self):
        """Test that device='cpu' behaves like old gpu=False behavior."""
        device = select_device("cpu", verbose=False)
        assert device.type == "cpu"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Mock GPU")
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_auto_equivalent_to_gpu_true(self, mock_mps, mock_name, mock_cuda):
        """Test that device='auto' behaves like old gpu=True behavior (selects best available)."""
        device = select_device("auto", verbose=False)
        assert device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
