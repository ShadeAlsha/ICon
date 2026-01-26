"""
I-Con Playground: A student-friendly layer for experimenting with I-Con configurations.

This module provides simplified interfaces for running small I-Con experiments,
visualizing learned representations, and evaluating embeddings with simple probes.
"""

from .playground_config import PlaygroundConfig
from .playground_runner import run_playground_experiment
from .playground_viz import plot_training_curves, plot_embeddings_2d
from .playground_probes import run_linear_probe

__all__ = [
    "PlaygroundConfig",
    "run_playground_experiment",
    "plot_training_curves",
    "plot_embeddings_2d",
    "run_linear_probe",
]
