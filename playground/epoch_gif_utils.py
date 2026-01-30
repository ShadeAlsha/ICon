"""
Epoch-by-Epoch GIF Utilities for I-Con Playground

Generates animated visualizations showing how embeddings evolve during training.
This is critical for understanding learning dynamics - static plots hide temporal patterns.

Key features:
- Generate PCA frames for each epoch
- Maintain consistent PCA projection across all epochs
- Stitch frames into animated GIFs
- Configurable subsampling for speed
- Pure Python (no external dependencies like ffmpeg)

Example usage:
    manager = EpochGIFManager(output_dir="runs/experiment_1/epoch_frames")

    # After each epoch
    manager.save_epoch_frame(
        embeddings=val_embeddings,
        labels=val_labels,
        epoch=epoch_num
    )

    # After training completes
    manager.create_gif("training_dynamics.gif")
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA
import warnings


class EpochGIFManager:
    """
    Manages epoch-by-epoch embedding visualization and GIF generation.

    This class handles:
    1. Consistent PCA projection across all epochs (fit once, transform many)
    2. Frame generation with matplotlib
    3. GIF stitching with imageio

    The key insight: We need to fit PCA once (on final embeddings or all concatenated)
    to ensure the projection is consistent across epochs. Otherwise, each epoch's
    PCA would have different axes, making the animation meaningless.
    """

    def __init__(
        self,
        output_dir: Path,
        max_samples: int = 2000,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100,
    ):
        """
        Initialize the GIF manager.

        Args:
            output_dir: Directory to save frames and GIF
            max_samples: Maximum samples to plot (subsample for speed)
            figsize: Figure size for each frame
            dpi: DPI for saved frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_samples = max_samples
        self.figsize = figsize
        self.dpi = dpi

        # PCA state (will be fit on first epoch or explicitly)
        self.pca: Optional[PCA] = None
        self.pca_fitted = False

        # Track saved frames
        self.frame_paths: List[Path] = []

    def fit_pca(self, embeddings: np.ndarray) -> None:
        """
        Fit PCA on reference embeddings for consistent projection.

        This should be called once, typically with final epoch embeddings
        or all concatenated embeddings across epochs.

        Args:
            embeddings: Reference embeddings to fit PCA on (N, D)
        """
        if embeddings.shape[0] > self.max_samples:
            # Subsample for PCA fitting
            indices = np.random.choice(len(embeddings), self.max_samples, replace=False)
            embeddings = embeddings[indices]

        self.pca = PCA(n_components=2)
        self.pca.fit(embeddings)
        self.pca_fitted = True

        explained_var = self.pca.explained_variance_ratio_
        print(f"  PCA fitted: {explained_var[0]:.2%} + {explained_var[1]:.2%} = "
              f"{explained_var.sum():.2%} variance explained")

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to 2D using fitted PCA.

        Args:
            embeddings: Embeddings to transform (N, D)

        Returns:
            2D embeddings (N, 2)
        """
        if not self.pca_fitted:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")

        return self.pca.transform(embeddings)

    def save_epoch_frame(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epoch: int,
        title_prefix: str = "Epoch",
        auto_fit_pca: bool = True,
    ) -> Path:
        """
        Generate and save a PCA frame for one epoch.

        Args:
            embeddings: Embeddings for this epoch (N, D)
            labels: Corresponding labels (N,)
            epoch: Epoch number
            title_prefix: Prefix for plot title
            auto_fit_pca: If True and PCA not fitted, fit on this data

        Returns:
            Path to saved frame
        """
        # Fit PCA if not already done
        if not self.pca_fitted and auto_fit_pca:
            print(f"  Fitting PCA on epoch {epoch} embeddings...")
            self.fit_pca(embeddings)

        if not self.pca_fitted:
            raise RuntimeError("PCA not fitted. Call fit_pca() or set auto_fit_pca=True.")

        # Subsample if needed
        if len(embeddings) > self.max_samples:
            indices = np.random.choice(len(embeddings), self.max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Transform to 2D
        embeddings_2d = self.transform(embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Scatter plot colored by class
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f"Class {int(label)}",
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_title(f"{title_prefix} {epoch}", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Save frame
        frame_path = self.output_dir / f"epoch_{epoch:03d}.png"
        plt.tight_layout()
        plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        self.frame_paths.append(frame_path)
        return frame_path

    def create_gif(
        self,
        output_path: Path,
        duration: float = 0.5,
        loop: int = 0,
    ) -> Path:
        """
        Stitch all saved frames into an animated GIF.

        Args:
            output_path: Path for output GIF
            duration: Duration of each frame in seconds
            loop: Number of loops (0 = infinite)

        Returns:
            Path to created GIF
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for GIF generation. Install with:\n"
                "  pip install imageio\n"
                "or:\n"
                "  conda install imageio"
            )

        if not self.frame_paths:
            raise RuntimeError("No frames saved. Call save_epoch_frame() first.")

        # Sort frames by epoch number (in case they were saved out of order)
        sorted_frames = sorted(self.frame_paths, key=lambda p: int(p.stem.split('_')[1]))

        # Read all frames
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            images = [imageio.imread(str(frame)) for frame in sorted_frames]

        # Write GIF
        output_path = Path(output_path)
        imageio.mimsave(
            str(output_path),
            images,
            duration=duration,
            loop=loop,
        )

        print(f"\n{'='*60}")
        print(f"GIF created: {output_path}")
        print(f"  Frames: {len(sorted_frames)}")
        print(f"  Duration per frame: {duration}s")
        print(f"  Total duration: {len(sorted_frames) * duration:.1f}s")
        print(f"{'='*60}\n")

        return output_path


def create_training_dynamics_gif(
    frame_dir: Path,
    output_path: Path,
    duration: float = 0.5,
) -> Path:
    """
    Convenience function to create a GIF from a directory of frames.

    This is useful if frames were saved separately and you want to
    create the GIF after the fact.

    Args:
        frame_dir: Directory containing epoch frames
        output_path: Path for output GIF
        duration: Duration of each frame in seconds

    Returns:
        Path to created GIF
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required. Install with: pip install imageio")

    frame_dir = Path(frame_dir)
    frame_paths = sorted(frame_dir.glob("epoch_*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"No epoch frames found in {frame_dir}")

    images = [imageio.imread(str(frame)) for frame in frame_paths]

    output_path = Path(output_path)
    imageio.mimsave(str(output_path), images, duration=duration, loop=0)

    print(f"GIF created: {output_path} ({len(frame_paths)} frames)")
    return output_path
