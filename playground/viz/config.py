"""
Visualization Configuration

Defines all visualization-related settings in a structured, validated dataclass.
All settings are explicit, documented, and logged for reproducibility.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class VizMode(str, Enum):
    """Visualization output mode."""
    NONE = "none"        # No visualizations
    STATIC = "static"    # Final PCA/t-SNE only
    GIF = "gif"          # Epoch-by-epoch animation only
    BOTH = "both"        # GIF + final static plots


class GifMethod(str, Enum):
    """Projection method for GIF generation."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


class GifOverlay(str, Enum):
    """Text overlay on GIF frames."""
    NONE = "none"
    LOSS = "loss"
    EPOCH = "epoch"


@dataclass
class VizConfig:
    """
    Configuration for all visualization settings.

    This dataclass provides explicit, validated settings for visualization output.
    All flags are discoverable via --help and logged in experiment_manifest.json.

    Attributes:
        viz_mode: Output mode - none, static, gif, or both
        gif_every: Save a frame every N epochs (default: 1)
        gif_method: Projection method for GIF (pca, tsne, umap)
        gif_fps: Frames per second for GIF playback
        gif_max_points: Subsample embeddings for performance
        gif_overlay: Text overlay on frames (none, loss, epoch)
        save_frames: Whether to save individual frame PNGs
        gif_dpi: DPI for saved frames
        gif_figsize: Figure size (width, height) in inches

    Example:
        >>> config = VizConfig(
        ...     viz_mode=VizMode.BOTH,
        ...     gif_every=5,
        ...     gif_method=GifMethod.PCA,
        ...     gif_fps=2.0
        ... )
        >>> print(config.describe())
    """

    viz_mode: VizMode = VizMode.BOTH
    gif_every: int = 1
    gif_method: GifMethod = GifMethod.PCA
    gif_fps: float = 2.0
    gif_max_points: int = 5000
    gif_overlay: GifOverlay = GifOverlay.EPOCH
    save_frames: bool = True
    gif_dpi: int = 100
    gif_figsize: tuple = (10, 8)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string inputs to enums if needed
        if isinstance(self.viz_mode, str):
            self.viz_mode = VizMode(self.viz_mode.lower())
        if isinstance(self.gif_method, str):
            self.gif_method = GifMethod(self.gif_method.lower())
        if isinstance(self.gif_overlay, str):
            self.gif_overlay = GifOverlay(self.gif_overlay.lower())

        # Validate ranges
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.gif_every < 1:
            raise ValueError(f"gif_every must be >= 1, got {self.gif_every}")

        if self.gif_fps <= 0:
            raise ValueError(f"gif_fps must be > 0, got {self.gif_fps}")

        if self.gif_max_points < 100:
            raise ValueError(f"gif_max_points must be >= 100, got {self.gif_max_points}")

        if self.gif_dpi < 50:
            raise ValueError(f"gif_dpi must be >= 50, got {self.gif_dpi}")

        if len(self.gif_figsize) != 2:
            raise ValueError(f"gif_figsize must be (width, height), got {self.gif_figsize}")

        if self.gif_figsize[0] <= 0 or self.gif_figsize[1] <= 0:
            raise ValueError(f"gif_figsize dimensions must be > 0, got {self.gif_figsize}")

        # Warn about expensive operations
        if self.gif_method == GifMethod.TSNE and self.gif_every == 1:
            import warnings
            warnings.warn(
                "t-SNE with gif_every=1 will be slow. Consider gif_every=5 or higher.",
                UserWarning
            )

        if self.gif_method == GifMethod.UMAP and self.gif_every == 1:
            import warnings
            warnings.warn(
                "UMAP with gif_every=1 will be slow. Consider gif_every=5 or higher.",
                UserWarning
            )

    @property
    def should_generate_gif(self) -> bool:
        """Check if GIF generation is enabled."""
        return self.viz_mode in (VizMode.GIF, VizMode.BOTH)

    @property
    def should_generate_static(self) -> bool:
        """Check if static visualization is enabled."""
        return self.viz_mode in (VizMode.STATIC, VizMode.BOTH)

    @property
    def frame_duration(self) -> float:
        """Get frame duration in seconds (inverse of fps)."""
        return 1.0 / self.gif_fps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "viz_mode": self.viz_mode.value,
            "gif_every": self.gif_every,
            "gif_method": self.gif_method.value,
            "gif_fps": self.gif_fps,
            "gif_max_points": self.gif_max_points,
            "gif_overlay": self.gif_overlay.value,
            "save_frames": self.save_frames,
            "gif_dpi": self.gif_dpi,
            "gif_figsize": list(self.gif_figsize),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VizConfig':
        """Create VizConfig from dictionary."""
        # Handle figsize conversion from list
        if 'gif_figsize' in d and isinstance(d['gif_figsize'], list):
            d = d.copy()
            d['gif_figsize'] = tuple(d['gif_figsize'])
        return cls(**d)

    def describe(self) -> str:
        """Return a human-readable description of visualization settings."""
        lines = [
            "Visualization Configuration",
            "=" * 40,
            f"  Mode:        {self.viz_mode.value}",
        ]

        if self.should_generate_gif:
            lines.extend([
                f"  GIF every:   {self.gif_every} epoch(s)",
                f"  GIF method:  {self.gif_method.value.upper()}",
                f"  GIF fps:     {self.gif_fps}",
                f"  Max points:  {self.gif_max_points}",
                f"  Overlay:     {self.gif_overlay.value}",
                f"  Save frames: {self.save_frames}",
            ])

        return "\n".join(lines)


def create_viz_config_from_args(
    viz_mode: str = "both",
    gif_every: int = 1,
    gif_method: str = "pca",
    gif_fps: float = 2.0,
    gif_max_points: int = 5000,
    gif_overlay: str = "epoch",
    save_frames: bool = True,
) -> VizConfig:
    """
    Create VizConfig from CLI arguments.

    This is a convenience function for the CLI to create a VizConfig
    from individual argument values.

    Args:
        viz_mode: One of 'none', 'static', 'gif', 'both'
        gif_every: Save frame every N epochs
        gif_method: One of 'pca', 'tsne', 'umap'
        gif_fps: Frames per second
        gif_max_points: Maximum points to plot
        gif_overlay: One of 'none', 'loss', 'epoch'
        save_frames: Whether to keep individual frame PNGs

    Returns:
        Configured VizConfig instance
    """
    return VizConfig(
        viz_mode=VizMode(viz_mode.lower()),
        gif_every=gif_every,
        gif_method=GifMethod(gif_method.lower()),
        gif_fps=gif_fps,
        gif_max_points=gif_max_points,
        gif_overlay=GifOverlay(gif_overlay.lower()),
        save_frames=save_frames,
    )
