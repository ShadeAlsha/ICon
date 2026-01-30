"""
Tests for explicit device handling.

These tests verify:
1. Device selection works correctly (CPU, CUDA, MPS)
2. GPU request without GPU raises clear error
3. Model and distributions are moved to correct device
4. Batch tensors are moved to correct device
5. Loss tensors are on correct device
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.pure_torch_trainer import DeviceManager


class TestDeviceManager:
    """Test suite for device management."""

    def test_force_cpu(self):
        """Test forcing CPU device."""
        device = DeviceManager.get_device(gpu=False, verbose=False)
        assert device.type == "cpu"

    def test_auto_select_returns_valid_device(self):
        """Test that auto-select returns a valid device."""
        device = DeviceManager.get_device(gpu=None, verbose=False)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_gpu_request_with_no_gpu_raises_error(self):
        """Test that requesting GPU without one raises RuntimeError."""
        # Only run if no GPU available
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            with pytest.raises(RuntimeError, match="GPU requested.*but no GPU available"):
                DeviceManager.get_device(gpu=True, verbose=False)
        else:
            pytest.skip("GPU available, skipping no-GPU test")

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                        reason="Requires GPU (CUDA or MPS)")
    def test_gpu_request_with_gpu_works(self):
        """Test that requesting GPU with available GPU works."""
        device = DeviceManager.get_device(gpu=True, verbose=False)
        assert device.type in ["cuda", "mps"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cuda_device_selection(self):
        """Test CUDA device selection."""
        device = DeviceManager.get_device(gpu=True, verbose=False)
        assert device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
    def test_mps_device_selection(self):
        """Test MPS device selection on Apple Silicon."""
        # Only test if CUDA is not available (otherwise CUDA takes precedence)
        if not torch.cuda.is_available():
            device = DeviceManager.get_device(gpu=True, verbose=False)
            assert device.type == "mps"

    def test_move_tensor_to_device(self):
        """Test moving tensors to device."""
        device = torch.device("cpu")
        tensor = torch.randn(4, 4)

        moved_tensor = DeviceManager.move_to_device(tensor, device)

        assert moved_tensor.device.type == device.type

    def test_move_module_to_device(self):
        """Test moving nn.Module to device."""
        device = torch.device("cpu")
        module = nn.Linear(10, 10)

        moved_module = DeviceManager.move_to_device(module, device)

        # Check that module parameters are on correct device
        for param in moved_module.parameters():
            assert param.device.type == device.type

    def test_move_non_tensor_unchanged(self):
        """Test that non-tensor objects are returned unchanged."""
        device = torch.device("cpu")
        obj = {"key": "value", "number": 42}

        result = DeviceManager.move_to_device(obj, device)

        assert result == obj
        assert result is obj  # Should be same object

    def test_device_manager_verbose_output(self, capsys):
        """Test that verbose=True prints device info."""
        DeviceManager.get_device(gpu=False, verbose=True)

        captured = capsys.readouterr()
        assert "Device:" in captured.out
        assert "CPU" in captured.out


class TestModelDevicePlacement:
    """Test that models and distributions are placed on correct device."""

    def test_model_on_cpu(self):
        """Test model placement on CPU."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

        device = torch.device("cpu")
        model = model.to(device)

        # Check all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                        reason="Requires GPU")
    def test_model_on_gpu(self):
        """Test model placement on GPU."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

        device = DeviceManager.get_device(gpu=True, verbose=False)
        model = model.to(device)

        # Check all parameters are on GPU
        for param in model.parameters():
            assert param.device.type == device.type

    def test_batch_dict_device_movement(self):
        """Test moving batch dictionary to device."""
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.tensor([0, 1, 2, 3]),
            "index": torch.tensor([0, 1, 2, 3]),
            "metadata": "some string",  # Non-tensor
        }

        device = torch.device("cpu")

        # Move batch to device
        moved_batch = {
            key: val.to(device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }

        # Check tensors moved
        assert moved_batch["image"].device.type == "cpu"
        assert moved_batch["label"].device.type == "cpu"
        assert moved_batch["index"].device.type == "cpu"

        # Check non-tensor unchanged
        assert moved_batch["metadata"] == "some string"

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                        reason="Requires GPU")
    def test_loss_tensor_on_gpu(self):
        """Test that loss computation produces tensors on GPU."""
        device = DeviceManager.get_device(gpu=True, verbose=False)

        # Create simple model
        model = nn.Linear(10, 10).to(device)

        # Create input on GPU
        x = torch.randn(4, 10).to(device)
        target = torch.randn(4, 10).to(device)

        # Forward pass
        output = model(x)

        # Compute loss
        loss = nn.functional.mse_loss(output, target)

        # Check loss is on GPU
        assert loss.device.type == device.type

    @pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                        reason="Requires GPU")
    def test_gradient_on_gpu(self):
        """Test that gradients are computed on GPU."""
        device = DeviceManager.get_device(gpu=True, verbose=False)

        # Create simple model
        model = nn.Linear(10, 10).to(device)

        # Create input on GPU
        x = torch.randn(4, 10).to(device)
        target = torch.randn(4, 10).to(device)

        # Forward pass
        output = model(x)
        loss = nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients are on GPU
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.device.type == device.type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
