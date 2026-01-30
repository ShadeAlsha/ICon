"""
Playground Runner Module

Handles the execution of I-Con experiments using the playground configuration.
Wraps the I-Con training loop with additional logging and embedding extraction.
"""

import os
import sys
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import warnings

# Add parent directory to path so we can import from the main I-Con modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kernel_model import Model
from dataloaders import get_dataloaders
from .playground_config import PlaygroundConfig
from .pure_torch_trainer import PureTorchTrainer, DeviceManager


# ---------------------------------------------------------------------------
# Custom callback to collect training logs and embeddings
# ---------------------------------------------------------------------------

class PlaygroundLogger(Callback):
    """
    A lightweight callback to collect training metrics and embeddings.

    This callback captures loss values during training and collects
    embeddings during validation for later analysis.
    """

    def __init__(self):
        super().__init__()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.epoch_logs: List[Dict[str, float]] = []

        # For collecting embeddings at the end of training
        self._val_embeddings: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._collect_embeddings = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Record training metrics at the end of each epoch."""
        # Get logged metrics
        metrics = trainer.callback_metrics

        epoch_log = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss", torch.tensor(0.0)).item(),
        }

        if "train_accuracy" in metrics:
            epoch_log["train_accuracy"] = metrics["train_accuracy"].item()
            self.train_accuracies.append(epoch_log["train_accuracy"])

        self.train_losses.append(epoch_log["train_loss"])
        self.epoch_logs.append(epoch_log)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Record validation metrics and optionally collect embeddings."""
        metrics = trainer.callback_metrics

        if "val_loss" in metrics:
            self.val_losses.append(metrics["val_loss"].item())
        if "val_accuracy" in metrics:
            self.val_accuracies.append(metrics["val_accuracy"].item())

        # Update epoch log with validation metrics
        if self.epoch_logs:
            self.epoch_logs[-1]["val_loss"] = metrics.get("val_loss", torch.tensor(0.0)).item()
            if "val_accuracy" in metrics:
                self.epoch_logs[-1]["val_accuracy"] = metrics["val_accuracy"].item()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Collect embeddings during the final validation epoch."""
        if self._collect_embeddings and outputs is not None:
            if "embedding" in outputs:
                self._val_embeddings.append(outputs["embedding"].cpu())
            if "label" in outputs:
                self._val_labels.append(outputs["label"].cpu())
            if "index" in outputs:
                self._val_indices.append(outputs["index"].cpu())

    def start_embedding_collection(self):
        """Enable embedding collection for the next validation pass."""
        self._collect_embeddings = True
        self._val_embeddings = []
        self._val_labels = []
        self._val_indices = []

    def get_collected_embeddings(self) -> Dict[str, np.ndarray]:
        """Get the collected embeddings as numpy arrays."""
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


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_playground_experiment(
    pg_config: PlaygroundConfig,
    verbose: bool = True,
    save_checkpoints: Optional[bool] = None,
    gpu: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run an I-Con experiment using the playground configuration.

    This function:
    1. Converts the PlaygroundConfig to an internal I-Con Config
    2. Sets up data loaders for the specified dataset
    3. Trains the model using PyTorch Lightning
    4. Extracts embeddings from the validation set
    5. Saves all artifacts to the output directory

    Args:
        pg_config: A PlaygroundConfig instance specifying the experiment
        verbose: Whether to print progress information
        save_checkpoints: Whether to save model checkpoints (defaults to config value)
        gpu: Device selection (None=auto-select, True=request GPU with fallback, False=force CPU)

    Returns:
        A dictionary containing:
        - "logs": Training logs (losses, accuracies per epoch)
        - "embeddings": Learned embeddings for the validation set (numpy array)
        - "labels": Corresponding labels (numpy array)
        - "config": The configuration used (as dict)
        - "paths": Paths to saved artifacts (embeddings, logs, etc.)

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails

    Example:
        >>> config = PlaygroundConfig(dataset="cifar10", icon_mode="simclr_like", epochs=5)
        >>> results = run_playground_experiment(config, gpu=True)
        >>> print(f"Final val loss: {results['logs']['val_losses'][-1]:.4f}")
    """
    pl.seed_everything(pg_config.seed)

    if verbose:
        print(pg_config.describe())
        print("\nStarting experiment...\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{pg_config.run_name}_{timestamp}" if pg_config.run_name else f"{pg_config.dataset}_{pg_config.backbone}_{pg_config.icon_mode}_{timestamp}"
    run_dir = Path(pg_config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(pg_config.to_dict(), f, indent=2)

    icon_config = pg_config.to_icon_config()

    if verbose:
        print(f"Loading {pg_config.dataset} dataset...")

    use_contrastive = pg_config.icon_mode in [
        "simclr_like", "sne_like", "tsne_like", "cluster_like",
        "barlow_twins_like", "vicreg_like", "debiasing_like"
    ]

    train_loader, val_loader = get_dataloaders(
        dataset_name=pg_config.dataset,
        batch_size=pg_config.batch_size,
        num_views=2 if use_contrastive else 1,
        num_workers=pg_config.num_workers,
        size=pg_config.image_size,
        with_augmentation=pg_config.use_augmentation,
        contrastive=use_contrastive,
        shuffle_train=True,
        shuffle_test=False,
    )

    if verbose:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = Model(icon_config)

    callbacks = []
    playground_logger = PlaygroundLogger()
    callbacks.append(playground_logger)

    if save_checkpoints is None:
        save_checkpoints = pg_config.save_checkpoints

    if save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="epoch_{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=max(1, pg_config.epochs // 5),
        )
        callbacks.append(checkpoint_callback)

    if gpu is False:
        accelerator, devices = "cpu", 1
        if verbose:
            print("Device: CPU (forced)")
    elif gpu is True:
        if torch.cuda.is_available():
            accelerator, devices = "gpu", 1
            if verbose:
                print("Device: GPU (CUDA)")
        elif torch.backends.mps.is_available():
            accelerator, devices = "mps", 1
            if verbose:
                print("Device: MPS (Apple Silicon)")
        else:
            accelerator, devices = "cpu", 1
            if verbose:
                warnings.warn("GPU requested but unavailable. Using CPU.", UserWarning)
                print("Device: CPU (fallback)")
    else:
        if torch.cuda.is_available():
            accelerator, devices = "gpu", 1
            if verbose:
                print("Device: GPU (auto-selected)")
        elif torch.backends.mps.is_available():
            accelerator, devices = "mps", 1
            if verbose:
                print("Device: MPS (auto-selected)")
        else:
            accelerator, devices = "cpu", 1
            if verbose:
                print("Device: CPU (auto-selected)")

    trainer = pl.Trainer(
        max_epochs=pg_config.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        logger=False,
        gradient_clip_val=icon_config.gradient_clip_val,
    )

    if verbose:
        print("\nTraining...")

    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}. Try: reduce batch_size, use smaller backbone, or run self-check.") from e

    if verbose:
        print("\nExtracting embeddings...")

    playground_logger.start_embedding_collection()
    trainer.validate(model, val_loader, verbose=False)
    embedding_data = playground_logger.get_collected_embeddings()

    logs = playground_logger.get_logs()

    embeddings_path = run_dir / "embeddings.npz"
    np.savez(
        embeddings_path,
        embeddings=embedding_data.get("embeddings", np.array([])),
        labels=embedding_data.get("labels", np.array([])),
        indices=embedding_data.get("indices", np.array([])),
    )

    logs_path = run_dir / "logs.json"
    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=2)

    model_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"\nExperiment complete!")
        print(f"Results saved to: {run_dir}")
        if logs["val_losses"]:
            print(f"Final validation loss: {logs['val_losses'][-1]:.4f}")
        if logs["val_accuracies"]:
            print(f"Final validation accuracy: {logs['val_accuracies'][-1]:.4f}")

    results = {
        "logs": logs,
        "embeddings": embedding_data.get("embeddings", np.array([])),
        "labels": embedding_data.get("labels", np.array([])),
        "config": pg_config.to_dict(),
        "paths": {
            "run_dir": str(run_dir),
            "embeddings": str(embeddings_path),
            "logs": str(logs_path),
            "config": str(config_path),
            "model": str(model_path),
        },
        "model": model,
    }

    return results


def _get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _create_experiment_manifest(
    pg_config: 'PlaygroundConfig',
    device: torch.device,
    logs: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Create a comprehensive experiment manifest for reproducibility.

    This manifest contains everything needed to understand and reproduce the experiment:
    - Full configuration
    - Environment details (PyTorch version, device, git commit)
    - Final results summary
    """
    import platform

    # Get device details
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    elif device.type == "mps":
        device_name = "Apple Silicon MPS"
        cuda_version = None
    else:
        device_name = "CPU"
        cuda_version = None

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": pg_config.to_dict(),
        "environment": {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "device_name": device_name,
            "cuda_version": cuda_version,
            "git_commit": _get_git_commit_hash(),
        },
        "results": {
            "final_train_loss": logs["train_losses"][-1] if logs["train_losses"] else None,
            "final_val_loss": logs["val_losses"][-1] if logs["val_losses"] else None,
            "num_epochs_completed": len(logs["train_losses"]),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "embeddings": "embeddings.npz",
            "logs": "logs.json",
            "config": "config.json",
            "model": "final_model.pt",
        },
        "visualization": {
            "viz_mode": pg_config.viz_mode,
            "gif_every": pg_config.gif_every,
            "gif_method": pg_config.gif_method,
            "gif_fps": pg_config.gif_fps,
            "gif_max_points": pg_config.gif_max_points,
            "gif_overlay": pg_config.gif_overlay,
        },
    }

    return manifest


def run_playground_experiment_pure_pytorch(
    pg_config: PlaygroundConfig,
    verbose: bool = True,
    save_checkpoints: Optional[bool] = None,
    device: str = "auto",
    debug_device: bool = False,
    gpu: Optional[bool] = None,  # Deprecated, for backward compatibility
) -> Dict[str, Any]:
    """
    Run an I-Con experiment using pure PyTorch (no Lightning).

    This is the new, explicit-device-handling version that:
    1. Uses pure PyTorch training loop (no Lightning)
    2. Has explicit device handling with NO silent fallbacks
    3. Supports divergence selection (KL, RKL, JS)
    4. Uses configurable optimizers (Adam, AdamW, SGD)
    5. Provides clear error messages

    Args:
        pg_config: A PlaygroundConfig instance specifying the experiment
        verbose: Whether to print progress information
        save_checkpoints: Whether to save model checkpoints (defaults to config value)
        device: Device selection string
            - "auto": Automatically select CUDA > MPS > CPU
            - "cuda": Force CUDA GPU (fails if unavailable)
            - "mps": Force Apple Silicon GPU (fails if unavailable)
            - "cpu": Force CPU execution
        debug_device: Print detailed device placement info for debugging
        gpu: [DEPRECATED] Use device parameter instead

    Returns:
        A dictionary containing:
        - "logs": Training logs (losses, accuracies per epoch)
        - "embeddings": Learned embeddings for the validation set (numpy array)
        - "labels": Corresponding labels (numpy array)
        - "config": The configuration used (as dict)
        - "paths": Paths to saved artifacts (embeddings, logs, etc.)
        - "model": Trained model

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If requested device is unavailable
        RuntimeError: If training fails

    Example:
        >>> config = PlaygroundConfig(dataset="cifar10", icon_mode="simclr_like",
        ...                           divergence="js", optimizer="adam", epochs=5)
        >>> results = run_playground_experiment_pure_pytorch(config, device="cuda", debug_device=True)
        >>> print(f"Final val loss: {results['logs']['val_losses'][-1]:.4f}")
    """
    # Handle backward compatibility with gpu parameter
    if gpu is not None:
        import warnings
        warnings.warn(
            "The 'gpu' parameter is deprecated. Use 'device' parameter instead. "
            "'gpu=True' → 'device=\"auto\"', 'gpu=False' → 'device=\"cpu\"'",
            DeprecationWarning,
            stacklevel=2
        )
        if gpu is False:
            device = "cpu"
        elif gpu is True:
            device = "auto"
    torch.manual_seed(pg_config.seed)
    np.random.seed(pg_config.seed)

    if verbose:
        print(pg_config.describe())
        print(f"\n{'='*60}")
        print("FINAL RESOLVED CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Dataset:       {pg_config.dataset}")
        print(f"  Backbone:      {pg_config.backbone}")
        print(f"  Icon mode:     {pg_config.icon_mode}")
        print(f"  Divergence:    {pg_config.divergence}")
        print(f"  Optimizer:     {pg_config.optimizer}")
        print(f"  Learning rate: {pg_config.learning_rate}")
        print(f"  Weight decay:  {pg_config.weight_decay}")
        print(f"  Temperature:   {pg_config.temperature}")
        print(f"  Embedding dim: {pg_config.embedding_dim}")
        print(f"  Batch size:    {pg_config.batch_size}")
        print(f"  Epochs:        {pg_config.epochs}")
        print(f"  Num workers:   {pg_config.num_workers}")
        print(f"{'='*60}")
        print("\nStarting experiment (Pure PyTorch mode)...\n")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{pg_config.run_name}_{timestamp}" if pg_config.run_name else \
               f"{pg_config.dataset}_{pg_config.backbone}_{pg_config.icon_mode}_{timestamp}"
    run_dir = Path(pg_config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(pg_config.to_dict(), f, indent=2)

    # Convert to I-Con config
    icon_config = pg_config.to_icon_config()

    # Get device using centralized device selection
    from playground.device import select_device, log_device_info, verify_device_usage

    if debug_device:
        log_device_info(verbose=verbose)

    torch_device = select_device(device, verbose=verbose)

    if debug_device:
        verify_device_usage(torch_device)

    # Load dataset
    use_contrastive = pg_config.icon_mode in [
        "simclr_like", "sne_like", "tsne_like", "cluster_like",
        "barlow_twins_like", "vicreg_like", "debiasing_like"
    ]

    # Load dataset (standard or custom)
    if pg_config.custom_dataset_type is not None:
        if verbose:
            print(f"\nLoading custom dataset ({pg_config.custom_dataset_type})...")
            print(f"  Path: {pg_config.custom_dataset_path}")

        from playground.custom_datasets import get_custom_dataloaders

        train_loader, val_loader = get_custom_dataloaders(
            dataset_type=pg_config.custom_dataset_type,
            path=pg_config.custom_dataset_path,
            batch_size=pg_config.batch_size,
            num_workers=pg_config.num_workers,
            contrastive=use_contrastive,
            num_views=2 if use_contrastive else 1,
            device=torch_device,
            class_name=pg_config.custom_dataset_class,
        )
    else:
        if verbose:
            print(f"\nLoading {pg_config.dataset} dataset...")

        train_loader, val_loader = get_dataloaders(
            dataset_name=pg_config.dataset,
            batch_size=pg_config.batch_size,
            num_views=2 if use_contrastive else 1,
            num_workers=pg_config.num_workers,
            size=pg_config.image_size,
            with_augmentation=pg_config.use_augmentation,
            contrastive=use_contrastive,
            shuffle_train=True,
            shuffle_test=False,
            device=torch_device,
        )

    if verbose:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = Model(icon_config)

    # Setup epoch embedding collector if GIF generation is enabled
    embedding_collector = None
    if pg_config.should_generate_gif:
        from playground.viz import EpochEmbeddingCollector
        embedding_collector = EpochEmbeddingCollector(
            output_dir=run_dir,
            gif_every=pg_config.gif_every,
            max_points=pg_config.gif_max_points,
        )
        if verbose:
            print(f"\nEpoch GIF generation enabled")
            print(f"  Method: {pg_config.gif_method.upper()}")
            print(f"  Save every: {pg_config.gif_every} epoch(s)")
            print(f"  Max points: {pg_config.gif_max_points}")

    # Create trainer
    trainer = PureTorchTrainer(
        model=model,
        device=torch_device,
        config=icon_config,
        verbose=verbose,
        debug_device=debug_device,
        embedding_collector=embedding_collector,
    )

    # Train
    if verbose:
        print("\nTraining...")

    try:
        trainer.fit(train_loader, val_loader, epochs=pg_config.epochs)
    except Exception as e:
        raise RuntimeError(
            f"Training failed: {e}. "
            f"Try: reduce batch_size, use smaller backbone, or run self-check."
        ) from e

    # Extract embeddings
    if verbose:
        print("\nExtracting embeddings...")

    trainer.start_embedding_collection()
    trainer.validate_epoch(val_loader)
    embedding_data = trainer.get_collected_embeddings()

    # Get logs
    logs = trainer.get_logs()

    # Save embeddings
    embeddings_path = run_dir / "embeddings.npz"
    np.savez(
        embeddings_path,
        embeddings=embedding_data.get("embeddings", np.array([])),
        labels=embedding_data.get("labels", np.array([])),
        indices=embedding_data.get("indices", np.array([])),
    )

    # Save logs
    logs_path = run_dir / "logs.json"
    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=2)

    # Save model
    model_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)

    # Save experiment manifest (for reproducibility)
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = _create_experiment_manifest(
        pg_config=pg_config,
        device=torch_device,
        logs=logs,
        run_dir=run_dir,
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate GIF from collected epoch embeddings if enabled
    gif_result = None
    if pg_config.should_generate_gif and embedding_collector is not None:
        from playground.viz import generate_training_gif

        # Get epoch metadata from logs
        epoch_metadata = {}
        for i, epoch_log in enumerate(logs.get("epoch_logs", [])):
            epoch_num = epoch_log.get("epoch", i) + 1
            epoch_metadata[epoch_num] = {
                "train_loss": epoch_log.get("train_loss"),
                "val_loss": epoch_log.get("val_loss"),
            }

        viz_config = pg_config.get_viz_config()
        embeddings_by_epoch = embedding_collector.get_all_epochs()

        if embeddings_by_epoch:
            gif_result = generate_training_gif(
                embeddings_by_epoch=embeddings_by_epoch,
                labels=embedding_data.get("labels", np.array([])),
                config=viz_config,
                output_dir=run_dir,
                epoch_metadata=epoch_metadata,
                random_state=pg_config.seed,
            )

            # Check for sanity warnings
            if gif_result.get("sanity_check", {}).get("warning"):
                if verbose:
                    print(f"\nWARNING: {gif_result['sanity_check']['warning']}")

    if verbose:
        print(f"\nExperiment complete!")
        print(f"Results saved to: {run_dir}")
        if logs["val_losses"]:
            print(f"Final validation loss: {logs['val_losses'][-1]:.4f}")

    # Return results
    paths_dict = {
        "run_dir": str(run_dir),
        "embeddings": str(embeddings_path),
        "logs": str(logs_path),
        "config": str(config_path),
        "model": str(model_path),
    }

    # Add GIF path if generated
    if gif_result is not None:
        paths_dict["gif"] = str(gif_result.get("gif_path", ""))

    results = {
        "logs": logs,
        "embeddings": embedding_data.get("embeddings", np.array([])),
        "labels": embedding_data.get("labels", np.array([])),
        "config": pg_config.to_dict(),
        "paths": paths_dict,
        "model": model,
        "device": str(device),
        "gif_result": gif_result,
    }

    return results


# ---------------------------------------------------------------------------
# Utility functions for loading saved experiments
# ---------------------------------------------------------------------------

def load_experiment_results(run_dir: str) -> Dict[str, Any]:
    """
    Load results from a previously saved experiment.

    Args:
        run_dir: Path to the experiment run directory

    Returns:
        Dictionary containing logs, embeddings, labels, and config
    """
    run_dir = Path(run_dir)

    results = {}

    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            results["config"] = json.load(f)

    # Load logs
    logs_path = run_dir / "logs.json"
    if logs_path.exists():
        with open(logs_path, "r") as f:
            results["logs"] = json.load(f)

    # Load embeddings
    embeddings_path = run_dir / "embeddings.npz"
    if embeddings_path.exists():
        data = np.load(embeddings_path)
        results["embeddings"] = data["embeddings"]
        results["labels"] = data["labels"]
        if "indices" in data:
            results["indices"] = data["indices"]

    results["paths"] = {
        "run_dir": str(run_dir),
        "embeddings": str(embeddings_path),
        "logs": str(logs_path),
        "config": str(config_path),
    }

    return results


def list_experiments(output_dir: str = "playground_runs") -> List[Dict[str, str]]:
    """
    List all saved experiments in the output directory.

    Args:
        output_dir: Base directory where experiments are saved

    Returns:
        List of dictionaries with experiment info (name, path, timestamp)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    experiments = []
    for run_dir in sorted(output_dir.iterdir()):
        if run_dir.is_dir():
            config_path = run_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                experiments.append({
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "dataset": config.get("dataset", "unknown"),
                    "icon_mode": config.get("icon_mode", "unknown"),
                    "epochs": config.get("epochs", 0),
                })

    return experiments
