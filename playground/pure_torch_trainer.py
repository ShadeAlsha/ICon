"""
Pure PyTorch Training Loop for I-Con Playground

Replaces PyTorch Lightning with explicit PyTorch training.
Provides full control over device placement and training process.

Key features:
- Explicit device handling (CPU, CUDA, MPS)
- No silent fallbacks when GPU requested
- Clear error messages
- Compatible with existing playground infrastructure
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import json
from tqdm import tqdm

if TYPE_CHECKING:
    from playground.epoch_gif_utils import EpochGIFManager


def assert_tensor_on_device(tensor: torch.Tensor, expected_device: torch.device, name: str = "tensor"):
    """
    Assert that a tensor is on the expected device.

    Args:
        tensor: Tensor to check
        expected_device: Expected device
        name: Name for error message

    Raises:
        RuntimeError: If tensor is on wrong device
    """
    actual_device = tensor.device
    expected_type = expected_device.type

    if actual_device.type != expected_type:
        raise RuntimeError(
            f"Device mismatch for '{name}':\n"
            f"  Expected: {expected_device} ({expected_type})\n"
            f"  Actual:   {actual_device}\n"
            f"\n"
            f"This usually means a tensor wasn't moved to the correct device.\n"
            f"Try:\n"
            f"  - Ensure all data is moved with .to(device)\n"
            f"  - Check custom datasets return tensors correctly\n"
            f"  - Use --num_workers 0 if using MPS"
        )


class DeviceManager:
    """Manages device selection and validation."""

    @staticmethod
    def get_device(gpu: Optional[bool] = None, verbose: bool = True) -> torch.device:
        """
        Get training device with explicit error handling.

        Args:
            gpu: None (auto-select), True (require GPU), False (force CPU)
            verbose: Whether to print device info

        Returns:
            torch.device object

        Raises:
            RuntimeError: If GPU requested but not available
        """
        if gpu is False:
            # Explicitly requested CPU
            device = torch.device("cpu")
            if verbose:
                print(f"Device: CPU (forced)")
            return device

        if gpu is True:
            # Explicitly requested GPU - MUST have one
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if verbose:
                    print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
                return device
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                if verbose:
                    print(f"Device: Apple Silicon GPU (MPS)")
                return device
            else:
                raise RuntimeError(
                    "GPU requested (--gpu) but no GPU available. "
                    "Available options:\n"
                    "  1. Run without --gpu flag for CPU training\n"
                    "  2. Run with --cpu flag to explicitly use CPU\n"
                    "  3. Ensure CUDA/MPS is properly installed"
                )

        # Auto-select (gpu is None)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if verbose:
                print(f"Device: CUDA GPU (auto-selected)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            if verbose:
                print(f"Device: MPS (auto-selected)")
        else:
            device = torch.device("cpu")
            if verbose:
                print(f"Device: CPU (auto-selected)")

        return device

    @staticmethod
    def move_to_device(obj: Any, device: torch.device) -> Any:
        """Move object to device if it's a tensor or nn.Module."""
        if isinstance(obj, (torch.Tensor, nn.Module)):
            return obj.to(device)
        return obj


class PureTorchTrainer:
    """
    Pure PyTorch trainer for I-Con models.

    Replaces PyTorch Lightning with explicit training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config,
        verbose: bool = True,
        debug_device: bool = False,
        epoch_frames_manager: Optional['EpochGIFManager'] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: I-Con model (should be a regular nn.Module, not LightningModule)
            device: Device to train on
            config: Model configuration
            verbose: Whether to print progress
            debug_device: Print device placement info for debugging
            epoch_frames_manager: Optional manager for saving epoch-by-epoch frames
        """
        self.device = device
        self.config = config
        self.verbose = verbose
        self.debug_device = debug_device
        self.epoch_frames_manager = epoch_frames_manager

        # Move model to device
        self.model = model.to(device)

        # Move distributions to device
        if hasattr(model, 'supervisory_distribution'):
            self.model.supervisory_distribution = model.supervisory_distribution.to(device)
        if hasattr(model, 'learned_distribution'):
            self.model.learned_distribution = model.learned_distribution.to(device)

        # Verify model is on correct device
        self._verify_model_device()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

    def _verify_model_device(self):
        """Verify all model components are on the correct device."""
        # Check model parameters
        for name, param in self.model.named_parameters():
            if param.device.type != self.device.type:
                raise RuntimeError(
                    f"Model parameter '{name}' is on {param.device}, expected {self.device}.\n"
                    f"This is a bug - please report it."
                )

        if self.debug_device:
            print(f"\n{'='*50}")
            print("DEBUG: Device Placement Verification")
            print(f"{'='*50}")
            print(f"  Target device: {self.device}")
            print(f"  Model params:  {next(self.model.parameters()).device}")
            if hasattr(self.model, 'supervisory_distribution'):
                sup_params = list(self.model.supervisory_distribution.parameters())
                if sup_params:
                    print(f"  Supervisory:   {sup_params[0].device}")
                else:
                    print(f"  Supervisory:   (no learnable params)")
            if hasattr(self.model, 'learned_distribution'):
                learn_params = list(self.model.learned_distribution.parameters())
                if learn_params:
                    print(f"  Learned dist:  {learn_params[0].device}")
                else:
                    print(f"  Learned dist:  (no learnable params)")
            print(f"{'='*50}\n")

        # Tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.epoch_logs: List[Dict[str, float]] = []

        # For collecting embeddings
        self._val_embeddings: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._collect_embeddings = False

    def _setup_optimizer(self):
        """Setup optimizer with proper parameter groups."""
        param_groups = [
            {'params': self.model.mapper.parameters(), 'lr': self.config.lr},
        ]

        if self.config.linear_probe:
            param_groups.append({
                'params': self.model.linear_probe.parameters(),
                'lr': self.config.lr * 5
            })

        if hasattr(self.model.learned_distribution, 'learnable_gamma') and \
           self.model.learned_distribution.learnable_gamma:
            param_groups.append({
                'params': [self.model.learned_distribution.gamma],
                'lr': 0.001 * self.config.lr
            })

        if self.config.optimizer == 'adam':
            return torch.optim.Adam(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(param_groups, weight_decay=self.config.weight_decay, momentum=0)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported")

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all tensors in batch to device."""
        return {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }

    def _preprocess_contrastive_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess contrastive batch by concatenating augmented views.

        The dataloader returns batches with:
        - 'image': first augmented view [batch_size, ...]
        - 'image1': second augmented view [batch_size, ...]
        - 'index': sample indices [batch_size]
        - 'label': class labels [batch_size]

        The model expects:
        - 'image': concatenated views [2*batch_size, ...]
        - 'index': duplicated indices [2*batch_size] so same sample has same index
        - 'label': duplicated labels [2*batch_size]

        This is CRITICAL for contrastive learning: the supervisory distribution
        uses indices to identify positive pairs (samples with the same index).
        """
        if 'image1' not in batch:
            # Not a contrastive batch, return as-is
            return batch

        # Concatenate image views: [view1, view2]
        image = batch['image']
        image1 = batch['image1']
        combined_image = torch.cat([image, image1], dim=0)

        # Duplicate indices: [idx1, idx2, ..., idx1, idx2, ...]
        # This ensures both views of sample i have the same index
        index = batch['index']
        combined_index = torch.cat([index, index], dim=0)

        # Duplicate labels
        label = batch['label']
        combined_label = torch.cat([label, label], dim=0)

        return {
            'image': combined_image,
            'index': combined_index,
            'label': combined_label,
        }

    def _sanity_check_contrastive_batch(
        self,
        batch: Dict[str, torch.Tensor],
        loss_dict: Dict[str, Any]
    ) -> None:
        """
        Sanity checks for contrastive learning batches.

        Verifies that:
        1. Batch indices have duplicates (indicating positive pairs exist)
        2. Icon loss is non-zero (supervisory distribution has positive pairs)
        3. Different objectives should produce different losses

        These checks catch common bugs like missing batch preprocessing.
        """
        index = batch.get('index')
        if index is None:
            return

        # Check 1: Indices should have duplicates for contrastive learning
        unique_count = index.unique().shape[0]
        total_count = index.shape[0]

        if unique_count == total_count and total_count > 1:
            # All indices are unique - no positive pairs!
            raise RuntimeError(
                f"Contrastive batch has no duplicate indices!\n"
                f"  Total indices: {total_count}\n"
                f"  Unique indices: {unique_count}\n"
                f"\n"
                f"This means no positive pairs exist for contrastive learning.\n"
                f"The batch preprocessing may have failed. Check that:\n"
                f"  1. _preprocess_contrastive_batch is called before _compute_loss\n"
                f"  2. The dataloader returns 'image1' for the second view\n"
                f"  3. Indices are correctly duplicated when concatenating views"
            )

        # Check 2: Icon loss should be non-zero for contrastive learning
        icon_loss = loss_dict['losses'].get('icon_loss')
        if icon_loss is not None:
            icon_loss_val = icon_loss.item()
            if abs(icon_loss_val) < 1e-8:
                print(
                    f"\n*** WARNING: icon_loss is near zero ({icon_loss_val:.2e}) ***\n"
                    f"This may indicate:\n"
                    f"  - Supervisory distribution has no positive pairs\n"
                    f"  - Batch preprocessing issue\n"
                    f"  - Degenerate embeddings\n"
                )

        if self.verbose:
            print(f"  Sanity check: batch has {total_count - unique_count} duplicate indices (positive pairs exist)")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        first_batch = True

        iterator = tqdm(train_loader, desc="Training", disable=not self.verbose)

        for batch in iterator:
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Preprocess contrastive batch (concatenate views, duplicate indices)
            batch = self._preprocess_contrastive_batch(batch)

            # Debug device info on first batch
            if first_batch and self.debug_device:
                self._debug_batch_devices(batch)

            # Forward pass
            self.optimizer.zero_grad()

            # Compute loss (this calls model's forward/loss computation)
            loss_dict = self.model._compute_loss(batch)
            loss = sum(loss_dict['losses'].values())

            # Sanity checks on first batch
            if first_batch:
                first_batch = False
                self._sanity_check_contrastive_batch(batch, loss_dict)

            # Verify loss is on correct device (critical check)
            if loss.device.type != self.device.type:
                raise RuntimeError(
                    f"Loss tensor is on {loss.device}, but model is on {self.device}.\n"
                    f"This indicates a device placement bug in the model.\n"
                    f"Check that all operations happen on the same device."
                )

            # Check for NaN with helpful error
            if torch.isnan(loss):
                raise ValueError(
                    f"NaN loss detected at batch {num_batches}.\n"
                    f"Possible causes:\n"
                    f"  - Learning rate too high (try --lr 1e-4)\n"
                    f"  - Numerical instability in divergence computation\n"
                    f"  - Invalid data (check for NaN in inputs)\n"
                    f"Try running with --debug_device for more info."
                )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            # Optimizer step
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if self.verbose:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        return {"train_loss": avg_loss}

    def _debug_batch_devices(self, batch: Dict[str, Any]):
        """Print device info for a batch (used with --debug_device)."""
        print(f"\n{'='*50}")
        print("DEBUG: First Batch Device Placement")
        print(f"{'='*50}")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  Batch['{key}']: {val.device} (shape: {val.shape})")
            else:
                print(f"  Batch['{key}']: {type(val).__name__}")
        print(f"{'='*50}\n")

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Preprocess contrastive batch if needed (no-op for non-contrastive)
                batch = self._preprocess_contrastive_batch(batch)

                # Forward pass
                loss_dict = self.model._compute_loss(batch)
                loss = sum(loss_dict['losses'].values())

                epoch_loss += loss.item()
                num_batches += 1

                # Collect embeddings if requested
                if self._collect_embeddings:
                    metrics = loss_dict['metrics']
                    if 'embedding' in metrics:
                        self._val_embeddings.append(metrics['embedding'].cpu())
                    if 'label' in metrics:
                        self._val_labels.append(metrics['label'].cpu())
                    if 'index' in metrics:
                        self._val_indices.append(metrics['index'].cpu())

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        return {"val_loss": avg_loss}

    def fit(self, train_loader, val_loader, epochs: int):
        """Train for multiple epochs."""
        if self.verbose:
            print(f"\nStarting training for {epochs} epochs...")
            print(f"Training device: {self.device}")
            print(f"Model on device: {next(self.model.parameters()).device}")

        for epoch in range(epochs):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics["val_loss"])

            # Save epoch frame if manager is provided
            if self.epoch_frames_manager is not None:
                if self.verbose:
                    print(f"  Saving epoch {epoch + 1} frame...")
                # Collect embeddings for this epoch
                self.start_embedding_collection()
                _ = self.validate_epoch(val_loader)
                epoch_data = self.get_collected_embeddings()

                if epoch_data["embeddings"].shape[0] > 0:
                    self.epoch_frames_manager.save_epoch_frame(
                        embeddings=epoch_data["embeddings"],
                        labels=epoch_data["labels"],
                        epoch=epoch + 1,
                    )

            # Log
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
            }
            self.epoch_logs.append(epoch_log)

            if self.verbose:
                print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")

    def start_embedding_collection(self):
        """Enable embedding collection for next validation pass."""
        self._collect_embeddings = True
        self._val_embeddings = []
        self._val_labels = []
        self._val_indices = []

    def get_collected_embeddings(self) -> Dict[str, Any]:
        """Get collected embeddings as numpy arrays."""
        import numpy as np

        result = {}
        if self._val_embeddings:
            result["embeddings"] = torch.cat(self._val_embeddings, dim=0).numpy()
        if self._val_labels:
            result["labels"] = torch.cat(self._val_labels, dim=0).numpy()
        if self._val_indices:
            result["indices"] = torch.cat(self._val_indices, dim=0).numpy()

        return result

    def get_logs(self) -> Dict[str, Any]:
        """Get all collected logs."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "epoch_logs": self.epoch_logs,
        }
