"""
Centralized Device Selection for I-Con Playground

Provides robust, research-grade device selection for:
- CPU-only machines
- Apple Silicon (MPS)
- CUDA GPUs (local, Colab, MIT/lab clusters)
- Headless Linux servers

This module ensures consistent device handling across all components.
"""

import torch
from typing import Literal, Optional


DeviceType = Literal["auto", "cpu", "cuda", "mps"]


def select_device(device: DeviceType = "auto", verbose: bool = True) -> torch.device:
    """
    Select training device with automatic detection and explicit overrides.

    Priority:
    1. If device is explicitly specified (cpu/cuda/mps), use that or fail clearly
    2. If device is "auto", select: CUDA > MPS > CPU

    Args:
        device: Device selection strategy
            - "auto": Automatically select CUDA > MPS > CPU
            - "cpu": Force CPU execution
            - "cuda": Force CUDA GPU (fails if unavailable)
            - "mps": Force Apple Silicon GPU (fails if unavailable)
        verbose: Whether to print device selection info

    Returns:
        torch.device object configured for the selected device

    Raises:
        RuntimeError: If explicitly requested device is unavailable

    Examples:
        >>> # Auto-select best available device
        >>> device = select_device("auto")
        Using device: cuda

        >>> # Force CPU for debugging
        >>> device = select_device("cpu")
        Using device: cpu

        >>> # Require CUDA on a GPU cluster
        >>> device = select_device("cuda")
        Using device: cuda
    """
    if device == "cpu":
        # Explicit CPU request - always succeeds
        selected = torch.device("cpu")
        if verbose:
            print("Using device: cpu")
        return selected

    elif device == "cuda":
        # Explicit CUDA request - must be available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU requested (--device cuda) but CUDA is not available.\n"
                "\n"
                "Troubleshooting:\n"
                "  1. Verify CUDA installation: nvidia-smi\n"
                "  2. Check PyTorch CUDA support: python -c 'import torch; print(torch.cuda.is_available())'\n"
                "  3. Reinstall PyTorch with CUDA: https://pytorch.org/get-started/locally/\n"
                "  4. Use --device auto or --device cpu for CPU training"
            )
        selected = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using device: cuda")
            print(f"  GPU: {gpu_name}")
        return selected

    elif device == "mps":
        # Explicit MPS request - must be available
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Apple Silicon GPU requested (--device mps) but MPS is not available.\n"
                "\n"
                "Requirements:\n"
                "  - macOS 12.3+ on Apple Silicon (M1/M2/M3)\n"
                "  - PyTorch 1.12+ with MPS support\n"
                "\n"
                "Troubleshooting:\n"
                "  1. Check MPS support: python -c 'import torch; print(torch.backends.mps.is_available())'\n"
                "  2. Update PyTorch: pip install --upgrade torch\n"
                "  3. Use --device auto or --device cpu for CPU training"
            )
        selected = torch.device("mps")
        if verbose:
            print("Using device: mps")
            print("  GPU: Apple Silicon")
        return selected

    elif device == "auto":
        # Automatic selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            selected = torch.device("cuda")
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using device: cuda")
                print(f"  GPU: {gpu_name}")
        elif torch.backends.mps.is_available():
            selected = torch.device("mps")
            if verbose:
                print("Using device: mps")
                print("  GPU: Apple Silicon")
        else:
            selected = torch.device("cpu")
            if verbose:
                print("Using device: cpu")
        return selected

    else:
        raise ValueError(
            f"Unknown device: {device}. "
            f"Must be one of: 'auto', 'cpu', 'cuda', 'mps'"
        )


def get_dataloader_config(device: torch.device) -> dict:
    """
    Get DataLoader configuration optimized for the given device.

    pin_memory should ONLY be enabled for CUDA. It provides no benefit
    (and may cause issues) on CPU or MPS.

    Args:
        device: torch.device object

    Returns:
        Dictionary with DataLoader configuration
            - pin_memory: bool
            - persistent_workers: bool (if num_workers > 0)

    Examples:
        >>> device = torch.device("cuda")
        >>> config = get_dataloader_config(device)
        >>> config
        {'pin_memory': True, 'persistent_workers': True}

        >>> device = torch.device("cpu")
        >>> config = get_dataloader_config(device)
        >>> config
        {'pin_memory': False}
    """
    config = {}

    # pin_memory: CUDA only
    if device.type == "cuda":
        config["pin_memory"] = True
        # persistent_workers helps with pin_memory overhead
        config["persistent_workers"] = True
    else:
        config["pin_memory"] = False

    return config


def log_device_info(verbose: bool = True) -> None:
    """
    Log detailed device and PyTorch configuration info.

    Useful for debugging GPU issues on clusters.

    Args:
        verbose: Whether to print info
    """
    if not verbose:
        return

    print("\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("="*60 + "\n")


def verify_device_usage(device: torch.device) -> None:
    """
    Print verification message for how users confirm GPU usage.

    Args:
        device: torch.device object
    """
    if device.type == "cuda":
        print("\nTo verify GPU usage on a cluster:")
        print("  1. Check this log shows 'Using device: cuda'")
        print("  2. Run nvidia-smi in another terminal")
        print("  3. Look for python process using GPU memory\n")
    elif device.type == "mps":
        print("\nTo verify GPU usage on Apple Silicon:")
        print("  1. Check this log shows 'Using device: mps'")
        print("  2. Open Activity Monitor > Window > GPU History")
        print("  3. Look for python process using GPU\n")
