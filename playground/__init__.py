"""
I-Con Playground: A student-friendly layer for experimenting with I-Con configurations.

This module provides simplified interfaces for running small I-Con experiments,
visualizing learned representations, and evaluating embeddings with simple probes.

Visualization Module (playground.viz):
    - VizConfig: Configuration for visualization settings
    - generate_training_gif: Standalone GIF generation from embeddings
    - regenerate_gif_from_run: Regenerate GIFs without retraining
    - EpochEmbeddingCollector: Collect embeddings during training
    - Projector: PCA/t-SNE/UMAP projection utilities
"""

from .playground_config import PlaygroundConfig
from .playground_runner import run_playground_experiment, run_playground_experiment_pure_pytorch
from .playground_viz import plot_training_curves, plot_embeddings_2d
from .playground_probes import run_linear_probe

# Visualization module exports (for convenience)
from .viz import (
    VizConfig,
    VizMode,
    GifMethod,
    GifOverlay,
    generate_training_gif,
    regenerate_gif_from_run,
    EpochEmbeddingCollector,
    Projector,
)

__all__ = [
    # Core
    "PlaygroundConfig",
    "run_playground_experiment",
    "run_playground_experiment_pure_pytorch",
    "plot_training_curves",
    "plot_embeddings_2d",
    "run_linear_probe",
    # Visualization
    "VizConfig",
    "VizMode",
    "GifMethod",
    "GifOverlay",
    "generate_training_gif",
    "regenerate_gif_from_run",
    "EpochEmbeddingCollector",
    "Projector",
]
