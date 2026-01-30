"""
I-Con Playground Visualization Module

A dedicated module for visualization, GIF generation, and rendering of training dynamics.
Designed for PhD researchers who need configurable, reproducible visualizations.

Key components:
- VizConfig: Configuration dataclass for all visualization settings
- generate_training_gif(): Standalone GIF generation from embeddings
- EpochEmbeddingCollector: Collects embeddings during training
- Projector: Unified interface for PCA/t-SNE/UMAP projections
"""

from .config import VizConfig, VizMode, GifMethod, GifOverlay
from .gif_generator import generate_training_gif, regenerate_gif_from_run
from .embedding_collector import EpochEmbeddingCollector
from .projector import Projector

__all__ = [
    # Configuration
    "VizConfig",
    "VizMode",
    "GifMethod",
    "GifOverlay",
    # GIF Generation
    "generate_training_gif",
    "regenerate_gif_from_run",
    # Embedding Collection
    "EpochEmbeddingCollector",
    # Projection
    "Projector",
]
