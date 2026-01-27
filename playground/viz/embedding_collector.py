"""
Epoch Embedding Collector

Collects and stores embeddings during training for later visualization.
This module decouples embedding collection from both training and visualization,
enabling post-hoc GIF regeneration without retraining.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class EpochEmbeddingCollector:
    """
    Collects embeddings at specified epochs during training.

    This class is responsible for:
    1. Collecting embeddings at configurable intervals (gif_every)
    2. Storing embeddings and metadata to disk
    3. Providing embeddings for visualization

    The key design principle is separation of concerns: the training loop
    calls collector.save_epoch() and the visualization system calls
    collector.get_all_epochs() independently.

    Attributes:
        output_dir: Directory to save embeddings
        gif_every: Collect embeddings every N epochs
        max_points: Maximum points to store per epoch

    Example:
        # During training
        >>> collector = EpochEmbeddingCollector(output_dir, gif_every=5)
        >>> for epoch in range(epochs):
        ...     if collector.should_collect(epoch):
        ...         collector.save_epoch(epoch, embeddings, labels, metadata)

        # For visualization (can be called later)
        >>> collector = EpochEmbeddingCollector.load(output_dir)
        >>> embeddings_by_epoch = collector.get_all_epochs()
    """

    def __init__(
        self,
        output_dir: Path,
        gif_every: int = 1,
        max_points: int = 5000,
    ):
        """
        Initialize the embedding collector.

        Args:
            output_dir: Directory to save epoch embeddings
            gif_every: Collect embeddings every N epochs
            max_points: Maximum points to store per epoch (for performance)
        """
        self.output_dir = Path(output_dir)
        self.gif_every = gif_every
        self.max_points = max_points

        # Create output directory
        self.embeddings_dir = self.output_dir / "epoch_embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Track collected epochs
        self._collected_epochs: List[int] = []
        self._metadata: Dict[str, Any] = {
            "gif_every": gif_every,
            "max_points": max_points,
            "epochs": [],
        }

    def should_collect(self, epoch: int) -> bool:
        """
        Check if embeddings should be collected for this epoch.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            True if this epoch should be collected
        """
        return epoch % self.gif_every == 0 or epoch == 1

    def save_epoch(
        self,
        epoch: int,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save embeddings for a single epoch.

        Args:
            epoch: Epoch number (1-indexed)
            embeddings: Embedding vectors (N, D)
            labels: Class labels (N,)
            metadata: Optional per-epoch metadata (e.g., loss, accuracy)

        Returns:
            Path to saved file
        """
        # Subsample if needed
        if len(embeddings) > self.max_points:
            indices = np.random.choice(
                len(embeddings),
                self.max_points,
                replace=False
            )
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Save embeddings
        epoch_path = self.embeddings_dir / f"epoch_{epoch:04d}.npz"
        np.savez(
            epoch_path,
            embeddings=embeddings,
            labels=labels,
            epoch=epoch,
        )

        # Update tracking
        if epoch not in self._collected_epochs:
            self._collected_epochs.append(epoch)
            self._collected_epochs.sort()

        # Update metadata
        epoch_info = {
            "epoch": epoch,
            "n_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
        }
        if metadata:
            epoch_info.update(metadata)

        # Add or update epoch info
        self._metadata["epochs"] = [
            e for e in self._metadata["epochs"] if e["epoch"] != epoch
        ]
        self._metadata["epochs"].append(epoch_info)
        self._metadata["epochs"].sort(key=lambda x: x["epoch"])

        # Save metadata
        self._save_metadata()

        return epoch_path

    def _save_metadata(self) -> None:
        """Save collection metadata to disk."""
        metadata_path = self.embeddings_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def get_epoch(self, epoch: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings for a specific epoch.

        Args:
            epoch: Epoch number to load

        Returns:
            Dictionary with 'embeddings', 'labels', 'epoch' or None if not found
        """
        epoch_path = self.embeddings_dir / f"epoch_{epoch:04d}.npz"
        if not epoch_path.exists():
            return None

        data = np.load(epoch_path)
        return {
            "embeddings": data["embeddings"],
            "labels": data["labels"],
            "epoch": int(data["epoch"]),
        }

    def get_all_epochs(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load all collected epoch embeddings.

        Returns:
            Dictionary mapping epoch number to embedding data
        """
        result = {}

        # Find all epoch files
        epoch_files = sorted(self.embeddings_dir.glob("epoch_*.npz"))

        for epoch_path in epoch_files:
            data = np.load(epoch_path)
            epoch = int(data["epoch"])
            result[epoch] = {
                "embeddings": data["embeddings"],
                "labels": data["labels"],
                "epoch": epoch,
            }

        return result

    def get_collected_epochs(self) -> List[int]:
        """Get list of collected epoch numbers."""
        epoch_files = sorted(self.embeddings_dir.glob("epoch_*.npz"))
        return [int(p.stem.split("_")[1]) for p in epoch_files]

    def get_metadata(self) -> Dict[str, Any]:
        """Get collection metadata."""
        metadata_path = self.embeddings_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return self._metadata

    @classmethod
    def load(cls, run_dir: Path) -> 'EpochEmbeddingCollector':
        """
        Load an existing collector from a run directory.

        Args:
            run_dir: Path to experiment run directory

        Returns:
            EpochEmbeddingCollector with loaded state
        """
        run_dir = Path(run_dir)
        embeddings_dir = run_dir / "epoch_embeddings"

        if not embeddings_dir.exists():
            raise FileNotFoundError(
                f"No epoch embeddings found in {run_dir}\n"
                f"Expected directory: {embeddings_dir}\n"
                f"Make sure the experiment was run with GIF generation enabled."
            )

        # Load metadata
        metadata_path = embeddings_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"gif_every": 1, "max_points": 5000, "epochs": []}

        # Create collector
        collector = cls(
            output_dir=run_dir,
            gif_every=metadata.get("gif_every", 1),
            max_points=metadata.get("max_points", 5000),
        )
        collector._metadata = metadata
        collector._collected_epochs = collector.get_collected_epochs()

        return collector

    def check_embedding_changes(self) -> Dict[str, Any]:
        """
        Sanity check: Verify that embeddings actually change across epochs.

        If embeddings don't change, this indicates a potential bug in the
        training loop (e.g., model not being updated).

        Returns:
            Dictionary with change analysis
        """
        epochs_data = self.get_all_epochs()

        if len(epochs_data) < 2:
            return {
                "changed": True,
                "message": "Insufficient epochs for comparison",
                "n_epochs": len(epochs_data),
            }

        epochs = sorted(epochs_data.keys())
        first_epoch = epochs[0]
        last_epoch = epochs[-1]

        first_emb = epochs_data[first_epoch]["embeddings"]
        last_emb = epochs_data[last_epoch]["embeddings"]

        # Compare embeddings (use smaller of the two sizes)
        n = min(len(first_emb), len(last_emb))
        first_emb = first_emb[:n]
        last_emb = last_emb[:n]

        # Compute change metrics
        diff = np.abs(first_emb - last_emb)
        mean_change = np.mean(diff)
        max_change = np.max(diff)
        relative_change = mean_change / (np.mean(np.abs(first_emb)) + 1e-10)

        # Check if embeddings are essentially unchanged
        is_unchanged = relative_change < 0.01  # Less than 1% change

        return {
            "changed": not is_unchanged,
            "mean_absolute_change": float(mean_change),
            "max_absolute_change": float(max_change),
            "relative_change": float(relative_change),
            "first_epoch": first_epoch,
            "last_epoch": last_epoch,
            "n_epochs_compared": len(epochs),
            "warning": (
                "Embeddings appear unchanged across epochs! "
                "This may indicate a training issue (frozen model, zero gradients, etc.)."
                if is_unchanged else None
            ),
        }

    def describe(self) -> str:
        """Get human-readable description of collected embeddings."""
        epochs = self.get_collected_epochs()
        n_epochs = len(epochs)

        lines = [
            f"Epoch Embedding Collector",
            f"=" * 40,
            f"  Output dir:  {self.embeddings_dir}",
            f"  gif_every:   {self.gif_every}",
            f"  max_points:  {self.max_points}",
            f"  Collected:   {n_epochs} epochs",
        ]

        if n_epochs > 0:
            lines.append(f"  Epochs:      {min(epochs)} to {max(epochs)}")

            # Show first epoch info
            first_data = self.get_epoch(min(epochs))
            if first_data:
                lines.append(f"  Embedding dim: {first_data['embeddings'].shape[1]}")
                lines.append(f"  Samples/epoch: {len(first_data['embeddings'])}")

        return "\n".join(lines)
