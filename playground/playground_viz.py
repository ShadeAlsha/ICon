"""
Playground Visualization Module

Provides simple visualization utilities for analyzing I-Con experiments:
- Training curves (loss, accuracy over epochs)
- 2D embedding visualizations (PCA, t-SNE)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Literal
from pathlib import Path


# ---------------------------------------------------------------------------
# Training curve visualization
# ---------------------------------------------------------------------------

def plot_training_curves(
    logs: Dict[str, List[float]],
    out_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    show: bool = True,
) -> plt.Figure:
    """
    Plot training and validation curves (loss, accuracy) from experiment logs.

    Args:
        logs: Dictionary containing training logs with keys like:
              - "train_losses": list of training losses per epoch
              - "val_losses": list of validation losses per epoch
              - "train_accuracies": (optional) list of training accuracies
              - "val_accuracies": (optional) list of validation accuracies
        out_path: Optional path to save the figure as PNG
        figsize: Figure size as (width, height) tuple
        show: Whether to display the plot (set False for headless runs)

    Returns:
        matplotlib Figure object

    Example:
        >>> logs = {"train_losses": [1.0, 0.8, 0.6], "val_losses": [1.1, 0.9, 0.7]}
        >>> fig = plot_training_curves(logs, out_path="curves.png")
    """
    # Determine which metrics we have
    has_loss = "train_losses" in logs and len(logs["train_losses"]) > 0
    has_accuracy = "train_accuracies" in logs and len(logs["train_accuracies"]) > 0

    num_plots = sum([has_loss, has_accuracy])
    if num_plots == 0:
        print("Warning: No metrics found in logs to plot.")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_loss:
        ax = axes[plot_idx]
        epochs = range(1, len(logs["train_losses"]) + 1)

        ax.plot(epochs, logs["train_losses"], "b-", label="Train Loss", linewidth=2)

        if "val_losses" in logs and len(logs["val_losses"]) > 0:
            val_epochs = range(1, len(logs["val_losses"]) + 1)
            ax.plot(val_epochs, logs["val_losses"], "r-", label="Val Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training & Validation Loss", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        all_losses = logs["train_losses"] + logs.get("val_losses", [])
        if all_losses:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            margin = (max_loss - min_loss) * 0.1 or 0.1
            ax.set_ylim(max(0, min_loss - margin), max_loss + margin)

        plot_idx += 1

    if has_accuracy:
        ax = axes[plot_idx]
        epochs = range(1, len(logs["train_accuracies"]) + 1)

        ax.plot(epochs, logs["train_accuracies"], "b-", label="Train Acc", linewidth=2)

        if "val_accuracies" in logs and len(logs["val_accuracies"]) > 0:
            val_epochs = range(1, len(logs["val_accuracies"]) + 1)
            ax.plot(val_epochs, logs["val_accuracies"], "r-", label="Val Acc", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Training & Validation Accuracy", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved training curves to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 2D embedding visualization
# ---------------------------------------------------------------------------

def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    out_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    show: bool = True,
    title: Optional[str] = None,
    max_samples: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
    colormap: str = "tab10",
    point_size: int = 10,
    alpha: float = 0.7,
) -> plt.Figure:
    """
    Visualize embeddings in 2D using PCA, t-SNE, or UMAP.

    This function reduces high-dimensional embeddings to 2D for visualization,
    colored by class labels. It helps students see how different I-Con modes
    affect the geometry of the learned representation space.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of shape (n_samples,) with class labels
        method: Dimensionality reduction method - "pca" (fast), "tsne", or "umap" (requires umap-learn)
        out_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        show: Whether to display the plot
        title: Optional custom title for the plot
        max_samples: Maximum samples to plot (subsampled if exceeded)
        perplexity: Perplexity parameter for t-SNE (ignored for PCA/UMAP)
        random_state: Random seed for reproducibility
        colormap: Matplotlib colormap name for class colors
        point_size: Size of scatter points
        alpha: Transparency of points (0-1)

    Returns:
        matplotlib Figure object

    Example:
        >>> embeddings = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> fig = plot_embeddings_2d(embeddings, labels, method="pca")
    """
    # Validate inputs
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings ({len(embeddings)}) and labels ({len(labels)}) must have same length")

    # Subsample if too many points
    n_samples = len(embeddings)
    if n_samples > max_samples:
        print(f"Subsampling from {n_samples} to {max_samples} points for visualization...")
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Perform dimensionality reduction
    print(f"Computing 2D projection using {method.upper()}...")

    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_.sum() * 100
        method_info = f"PCA (explained variance: {explained_var:.1f}%)"

    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE
        # Use PCA pre-reduction if embedding dim is very high
        if embeddings.shape[1] > 50:
            from sklearn.decomposition import PCA
            print("  Pre-reducing with PCA to 50 dimensions...")
            pca = PCA(n_components=50, random_state=random_state)
            embeddings = pca.fit_transform(embeddings)

        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        method_info = f"t-SNE (perplexity={perplexity})"

    elif method.lower() == "umap":
        try:
            import umap
        except ImportError:
            print("\nWarning: umap-learn not installed. Falling back to PCA.")
            print("To use UMAP, install with: pip install umap-learn\n")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=random_state)
            embeddings_2d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_.sum() * 100
            method_info = f"PCA (explained variance: {explained_var:.1f}%) [UMAP unavailable]"
        else:
            # UMAP with sensible defaults
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=random_state,
            )
            embeddings_2d = reducer.fit_transform(embeddings)
            method_info = "UMAP"

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'pca', 'tsne', or 'umap'.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get unique labels and colormap
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap(colormap, n_classes)

    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(i)],
            label=f"Class {label}",
            s=point_size,
            alpha=alpha,
        )

    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Embedding Visualization ({method_info})", fontsize=14)

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)

    # Add legend (only if not too many classes)
    if n_classes <= 15:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            markerscale=2,
        )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved embedding plot to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_embeddings(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a saved .npz file.

    Args:
        npz_path: Path to the .npz file

    Returns:
        Dictionary with 'embeddings', 'labels', and optionally 'indices'

    Example:
        >>> data = load_embeddings("playground_runs/run_001/embeddings.npz")
        >>> print(data["embeddings"].shape)
    """
    data = np.load(npz_path)
    result = {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
    }
    if "indices" in data:
        result["indices"] = data["indices"]
    return result


def plot_embedding_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 100,
    out_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize the distribution of pairwise distances within and between classes.

    This helps students understand how well the embedding space separates classes.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of shape (n_samples,) with class labels
        n_samples: Number of samples to use for distance computation
        out_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics.pairwise import cosine_distances

    # Subsample for efficiency
    if len(embeddings) > n_samples:
        np.random.seed(42)
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Compute pairwise distances
    distances = cosine_distances(embeddings)

    # Separate within-class and between-class distances
    within_class_dists = []
    between_class_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if labels[i] == labels[j]:
                within_class_dists.append(distances[i, j])
            else:
                between_class_dists.append(distances[i, j])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(
        within_class_dists,
        bins=50,
        alpha=0.6,
        label="Same class",
        color="green",
        density=True,
    )
    ax.hist(
        between_class_dists,
        bins=50,
        alpha=0.6,
        label="Different class",
        color="red",
        density=True,
    )

    ax.set_xlabel("Cosine Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Pairwise Distances", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add summary statistics
    within_mean = np.mean(within_class_dists)
    between_mean = np.mean(between_class_dists)
    ax.axvline(within_mean, color="green", linestyle="--", linewidth=2, label=f"Within mean: {within_mean:.3f}")
    ax.axvline(between_mean, color="red", linestyle="--", linewidth=2, label=f"Between mean: {between_mean:.3f}")

    # Update legend
    ax.legend(fontsize=10)

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved distance plot to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_experiment_summary(
    results: Dict,
    out_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a summary figure with training curves and embedding visualization.

    Args:
        results: Results dictionary from run_playground_experiment
        out_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 6))

    # Training curves
    ax1 = fig.add_subplot(1, 3, 1)
    logs = results.get("logs", {})
    if "train_losses" in logs and len(logs["train_losses"]) > 0:
        epochs = range(1, len(logs["train_losses"]) + 1)
        ax1.plot(epochs, logs["train_losses"], "b-", label="Train", linewidth=2)
        if "val_losses" in logs and len(logs["val_losses"]) > 0:
            ax1.plot(range(1, len(logs["val_losses"]) + 1), logs["val_losses"], "r-", label="Val", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No loss data", ha="center", va="center")

    # PCA embedding
    ax2 = fig.add_subplot(1, 3, 2)
    embeddings = results.get("embeddings", np.array([]))
    labels = results.get("labels", np.array([]))
    if len(embeddings) > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(embeddings[:2000])  # Limit samples
        for i in np.unique(labels[:2000]):
            mask = labels[:2000] == i
            ax2.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=5, alpha=0.5, label=f"{i}")
        ax2.set_title(f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No embeddings", ha="center", va="center")

    # Config info
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    config = results.get("config", {})
    info_text = "\n".join([
        "Experiment Configuration",
        "=" * 25,
        f"Dataset: {config.get('dataset', 'N/A')}",
        f"Backbone: {config.get('backbone', 'N/A')}",
        f"I-Con Mode: {config.get('icon_mode', 'N/A')}",
        f"Epochs: {config.get('epochs', 'N/A')}",
        f"Batch Size: {config.get('batch_size', 'N/A')}",
        f"Learning Rate: {config.get('learning_rate', 'N/A')}",
        f"Temperature: {config.get('temperature', 'N/A')}",
        "",
        "Results",
        "=" * 25,
        f"Final Val Loss: {logs['val_losses'][-1]:.4f}" if logs.get('val_losses') else "N/A",
        f"Final Val Acc: {logs['val_accuracies'][-1]:.4f}" if logs.get('val_accuracies') else "N/A",
        "",
        "Observed Limitations",
        "=" * 25,
        "• Small datasets: nuisance",
        "  factors may dominate",
        "• Unsupervised objectives:",
        "  no guarantee of semantic",
        "  alignment with labels",
        "• PCA separation ≠ linear",
        "  separability",
    ])
    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved summary to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Distance Histograms
# ---------------------------------------------------------------------------

def plot_distance_histograms(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    show: bool = True,
    max_samples: int = 1000,
) -> plt.Figure:
    """
    Plot histograms of intra-class and inter-class distances.

    This visualization helps understand how well classes are separated
    in the embedding space. Good representations should have:
    - Small intra-class distances (tight clusters)
    - Large inter-class distances (well-separated classes)

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of shape (n_samples,) with class labels
        out_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        show: Whether to display the plot
        max_samples: Maximum samples to use (subsampled for efficiency)

    Returns:
        matplotlib Figure object

    Example:
        >>> embeddings = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> fig = plot_distance_histograms(embeddings, labels)
    """
    from sklearn.metrics.pairwise import cosine_distances

    # Subsample for efficiency
    if len(embeddings) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Compute pairwise distances
    distances = cosine_distances(embeddings)

    # Separate within-class and between-class distances
    within_dists = []
    between_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if labels[i] == labels[j]:
                within_dists.append(distances[i, j])
            else:
                between_dists.append(distances[i, j])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Within-class histogram
    ax1.hist(within_dists, bins=50, alpha=0.7, color="green", edgecolor="black")
    ax1.axvline(np.mean(within_dists), color="darkgreen", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(within_dists):.3f}")
    ax1.set_xlabel("Cosine Distance", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Within-Class Distances", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Between-class histogram
    ax2.hist(between_dists, bins=50, alpha=0.7, color="red", edgecolor="black")
    ax2.axvline(np.mean(between_dists), color="darkred", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(between_dists):.3f}")
    ax2.set_xlabel("Cosine Distance", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Between-Class Distances", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add summary text
    sep_ratio = np.mean(between_dists) / (np.mean(within_dists) + 1e-8)
    fig.suptitle(f"Distance Distributions (Separability Ratio: {sep_ratio:.2f})", fontsize=15, y=1.02)

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved distance histograms to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Compare Embedding Spaces
# ---------------------------------------------------------------------------

def compare_embedding_spaces(
    runs: List[Dict],
    method: Literal["pca", "tsne"] = "pca",
    out_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    show: bool = True,
    max_samples: int = 2000,
) -> plt.Figure:
    """
    Compare embedding spaces from multiple runs side-by-side.

    This is useful for comparing different I-Con configurations
    (e.g., SimCLR-like vs SNE-like vs Supervised) on the same dataset.

    Args:
        runs: List of results dictionaries from run_playground_experiment,
              each containing 'embeddings', 'labels', and 'config' keys
        method: Dimensionality reduction method - "pca" or "tsne"
        out_path: Optional path to save the figure
        figsize: Figure size as (width, height). Auto-calculated if None.
        show: Whether to display the plot
        max_samples: Maximum samples to plot per run

    Returns:
        matplotlib Figure object

    Example:
        >>> config1 = PlaygroundConfig(icon_mode="simclr_like", epochs=10)
        >>> results1 = run_playground_experiment(config1)
        >>> config2 = PlaygroundConfig(icon_mode="sne_like", epochs=10)
        >>> results2 = run_playground_experiment(config2)
        >>> fig = compare_embedding_spaces([results1, results2], method="pca")
    """
    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("No runs provided")

    # Auto-calculate figure size
    if figsize is None:
        cols = min(n_runs, 3)
        rows = (n_runs + cols - 1) // cols
        figsize = (6 * cols, 5 * rows)

    # Create subplot grid
    cols = min(n_runs, 3)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Ensure axes is always a list
    if n_runs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, run in enumerate(runs):
        ax = axes[idx]

        embeddings = run.get("embeddings", np.array([]))
        labels = run.get("labels", np.array([]))
        config = run.get("config", {})

        if len(embeddings) == 0:
            ax.text(0.5, 0.5, "No embeddings", ha="center", va="center")
            ax.set_title(f"Run {idx + 1}")
            continue

        # Subsample if needed
        if len(embeddings) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Reduce to 2D
        if method.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            var_explained = reducer.explained_variance_ratio_.sum() * 100
            method_label = f"PCA ({var_explained:.1f}% var)"
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE
            if embeddings.shape[1] > 50:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50, random_state=42)
                embeddings = pca.fit_transform(embeddings)
            reducer = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
            embeddings_2d = reducer.fit_transform(embeddings)
            method_label = "t-SNE"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        cmap = plt.cm.get_cmap("tab10", n_classes)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(i)],
                s=10,
                alpha=0.6,
                label=f"Class {label}" if n_classes <= 10 else None
            )

        # Set title from config
        icon_mode = config.get("icon_mode", "unknown")
        run_name = config.get("run_name", f"Run {idx + 1}")
        ax.set_title(f"{run_name}\n({method_label})", fontsize=12)
        ax.set_xlabel("Dimension 1", fontsize=10)
        ax.set_ylabel("Dimension 2", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add legend only if few classes
        if n_classes <= 10 and idx == 0:
            ax.legend(fontsize=8, markerscale=2, loc="upper left")

    # Hide unused subplots
    for idx in range(n_runs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved comparison plot to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
