"""
GIF Generator for Training Dynamics Visualization

Standalone GIF generation from saved embeddings.
Designed to be called independently from training for post-hoc visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

from .config import VizConfig, GifMethod, GifOverlay
from .projector import Projector
from .embedding_collector import EpochEmbeddingCollector


def generate_training_gif(
    embeddings_by_epoch: Dict[int, Dict[str, np.ndarray]],
    labels: np.ndarray,
    config: VizConfig,
    output_dir: Path,
    epoch_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generate a training dynamics GIF from embeddings.

    This function is the main entry point for GIF generation. It:
    1. Selects a fixed anchor index set from epoch 1 for consistent point tracking
    2. Fits a projection model ONCE on the final epoch embeddings
    3. Projects all epochs using the same projection (consistent axes)
    4. Generates frame images for each epoch using the SAME anchor points
    5. Stitches frames into an animated GIF

    The anchor-based strategy ensures that the same points are tracked across
    all epochs, which is critical for custom datasets where per-epoch embedding
    counts may differ.

    Args:
        embeddings_by_epoch: Dict mapping epoch -> {'embeddings': array, 'labels': array}
        labels: Reference labels (used if epoch data doesn't include labels)
        config: VizConfig with GIF settings
        output_dir: Directory to save frames and GIF
        epoch_metadata: Optional per-epoch metadata (loss, accuracy, etc.)
        random_state: Seed for reproducibility (same seed → same GIF)

    Returns:
        Dictionary with:
        - gif_path: Path to generated GIF
        - frame_paths: List of frame image paths
        - projector_info: Information about the projection
        - sanity_check: Results of embedding change check
        - anchor_strategy: Information about anchor point selection

    Example:
        >>> from playground.viz import generate_training_gif, VizConfig
        >>> config = VizConfig(gif_method='pca', gif_fps=2.0)
        >>> result = generate_training_gif(
        ...     embeddings_by_epoch=collector.get_all_epochs(),
        ...     labels=final_labels,
        ...     config=config,
        ...     output_dir=run_dir
        ... )
        >>> print(f"GIF saved to: {result['gif_path']}")
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "epoch_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Set random state for reproducibility (deterministic GIFs)
    np.random.seed(random_state)

    # Sort epochs
    epochs = sorted(embeddings_by_epoch.keys())
    if not epochs:
        raise ValueError("No epochs provided for GIF generation")

    print(f"\n{'='*60}")
    print("GENERATING TRAINING DYNAMICS GIF")
    print(f"{'='*60}")
    print(f"  Method:     {config.gif_method.value.upper()}")
    print(f"  Epochs:     {min(epochs)} to {max(epochs)} ({len(epochs)} frames)")
    print(f"  FPS:        {config.gif_fps}")
    print(f"  Max points: {config.gif_max_points}")
    print(f"  Output:     {output_dir}")

    # ================================================================
    # ANCHOR-BASED POINT TRACKING STRATEGY
    # ================================================================
    # Select fixed anchor indices from the first epoch to track across
    # all epochs. This ensures consistent visualization even when
    # per-epoch embedding counts differ (common with custom datasets).
    # ================================================================
    first_epoch = min(epochs)
    first_epoch_data = embeddings_by_epoch[first_epoch]
    first_epoch_count = len(first_epoch_data["embeddings"])

    # Compute minimum sample count across all epochs
    min_epoch_count = min(
        len(embeddings_by_epoch[e]["embeddings"]) for e in epochs
    )
    max_epoch_count = max(
        len(embeddings_by_epoch[e]["embeddings"]) for e in epochs
    )

    # Determine anchor count: min of (gif_max_points, min_epoch_count)
    anchor_count = min(config.gif_max_points, min_epoch_count)

    # Warn if dataset is too small or unstable
    if anchor_count < 50:
        warnings.warn(
            f"Dataset has only {anchor_count} trackable points across all epochs. "
            f"GIF visualization may be sparse. Consider collecting more samples.",
            UserWarning
        )
        print(f"\n  WARNING: Small dataset - only {anchor_count} points can be tracked")

    if max_epoch_count != min_epoch_count:
        print(f"\n  NOTE: Variable embedding counts detected across epochs:")
        print(f"    Min: {min_epoch_count}, Max: {max_epoch_count}")
        print(f"    Using ANCHOR-BASED TRACKING to ensure consistent visualization")
        print(f"    Strategy: Track {anchor_count} fixed points across all epochs")
    else:
        print(f"\n  Tracking strategy: Fixed anchor points ({anchor_count} samples)")

    # Select anchor indices from first epoch (seeded for determinism)
    if first_epoch_count > anchor_count:
        anchor_indices = np.sort(np.random.choice(
            first_epoch_count,
            anchor_count,
            replace=False
        ))
    else:
        anchor_indices = np.arange(first_epoch_count)

    anchor_strategy = {
        "strategy": "fixed_anchor_tracking",
        "anchor_count": int(anchor_count),
        "first_epoch_count": int(first_epoch_count),
        "min_epoch_count": int(min_epoch_count),
        "max_epoch_count": int(max_epoch_count),
        "variable_sizes": max_epoch_count != min_epoch_count,
        "random_state": random_state,
    }

    # Fit projector on final epoch embeddings (using anchor subset)
    final_epoch = max(epochs)
    final_embeddings = embeddings_by_epoch[final_epoch]["embeddings"]

    # Use anchor indices for fitting (clipped to final epoch size)
    fit_indices = anchor_indices[anchor_indices < len(final_embeddings)]
    if len(fit_indices) < len(anchor_indices):
        warnings.warn(
            f"Final epoch has fewer samples ({len(final_embeddings)}) than anchor set "
            f"({len(anchor_indices)}). Using {len(fit_indices)} points for projection fit.",
            UserWarning
        )
    fit_embeddings = final_embeddings[fit_indices]

    print(f"\n  Fitting {config.gif_method.value.upper()} projector on epoch {final_epoch}...")
    print(f"    Using {len(fit_embeddings)} anchor points for projection")
    projector = Projector(
        method=config.gif_method,
        random_state=random_state,
        max_samples_for_fit=len(fit_embeddings),  # Use all anchor points
    )
    projector.fit(fit_embeddings)

    # Print projection info
    proj_info = projector.get_info()
    if config.gif_method == GifMethod.PCA and projector.total_variance_explained is not None:
        print(f"  Variance explained: {projector.total_variance_explained:.1%}")
        if projector.explained_variance_ratio is not None:
            print(f"    PC1: {projector.explained_variance_ratio[0]:.1%}")
            print(f"    PC2: {projector.explained_variance_ratio[1]:.1%}")

    # Sanity check: verify embeddings change across epochs (using anchors)
    sanity_check = _check_embedding_changes_anchored(
        embeddings_by_epoch, epochs, anchor_indices
    )
    if sanity_check.get("warning"):
        print(f"\n  WARNING: {sanity_check['warning']}")

    # Generate frames using anchor-based tracking
    print(f"\n  Generating {len(epochs)} frames (tracking {anchor_count} anchor points)...")
    frame_paths = []
    skipped_epochs = []

    # Pre-compute all 2D projections to determine global axis limits
    # This ensures consistent frame sizes for GIF animation
    projected_data = []
    for epoch in epochs:
        epoch_data = embeddings_by_epoch[epoch]
        embeddings = epoch_data["embeddings"]
        epoch_labels = epoch_data.get("labels", labels)

        valid_indices = anchor_indices[anchor_indices < len(embeddings)]

        if len(valid_indices) < anchor_count * 0.5:
            projected_data.append(None)  # Will be skipped
            continue

        embeddings_anchored = embeddings[valid_indices]
        labels_anchored = epoch_labels[valid_indices] if len(epoch_labels) > max(valid_indices) else epoch_labels[:len(valid_indices)]
        embeddings_2d = projector.transform(embeddings_anchored)
        projected_data.append((embeddings_2d, labels_anchored, epoch))

    # Compute global axis limits from all projected data
    all_2d = np.concatenate([d[0] for d in projected_data if d is not None], axis=0)
    x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    # Add 5% padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    axis_limits = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

    # Generate frames with consistent axis limits
    for i, (epoch, data) in enumerate(zip(epochs, projected_data)):
        if data is None:
            warnings.warn(
                f"Epoch {epoch} has insufficient anchor points. "
                f"Skipping frame to maintain visualization consistency.",
                UserWarning
            )
            skipped_epochs.append(epoch)
            continue

        embeddings_2d, labels_anchored, _ = data

        # Get metadata for overlay
        metadata = epoch_metadata.get(epoch, {}) if epoch_metadata else {}

        # Generate frame with global axis limits
        frame_path = _generate_frame(
            embeddings_2d=embeddings_2d,
            labels=labels_anchored,
            epoch=epoch,
            output_path=frames_dir / f"epoch_{epoch:04d}.png",
            config=config,
            metadata=metadata,
            proj_info=proj_info,
            axis_limits=axis_limits,
        )
        frame_paths.append(frame_path)

        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(epochs) - 1:
            print(f"    Frame {i+1}/{len(epochs)} complete")

    if skipped_epochs:
        print(f"\n  NOTE: Skipped {len(skipped_epochs)} epochs due to insufficient anchor coverage: {skipped_epochs}")

    # Stitch frames into GIF
    gif_path = output_dir / "training_dynamics.gif"
    print(f"\n  Creating GIF: {gif_path}")

    _create_gif(
        frame_paths=frame_paths,
        output_path=gif_path,
        fps=config.gif_fps,
    )

    # Verify the GIF was created and is valid
    if not gif_path.exists():
        raise RuntimeError(f"GIF generation failed: file not created at {gif_path}")

    gif_size_kb = gif_path.stat().st_size / 1024
    total_duration = len(epochs) / config.gif_fps

    # Summary with clear success message
    print(f"\n{'='*60}")
    print("ANIMATED GIF CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"")
    print(f"  Output file: {gif_path}")
    print(f"  File size:   {gif_size_kb:.1f} KB")
    print(f"  Frames:      {len(epochs)} (epochs {min(epochs)}-{max(epochs)})")
    print(f"  Duration:    {total_duration:.1f}s at {config.gif_fps} fps")

    if config.gif_method == GifMethod.PCA and projector.total_variance_explained:
        print(f"  Projection:  PCA ({projector.total_variance_explained:.1%} variance explained)")
    else:
        print(f"  Projection:  {config.gif_method.value.upper()}")

    print(f"")
    print(f"  To view the animation:")
    print(f"    - Open in browser: file://{gif_path.absolute()}")
    print(f"    - macOS: open \"{gif_path}\"")
    print(f"    - Linux: xdg-open \"{gif_path}\"")
    print(f"    - Or drag into Slack/Discord/Google Slides")

    if not config.save_frames:
        print(f"\n  Cleaning up intermediate frame files...")
        for frame_path in frame_paths:
            frame_path.unlink(missing_ok=True)
        if frames_dir.exists() and not any(frames_dir.iterdir()):
            frames_dir.rmdir()
        frame_paths = []
    else:
        print(f"\n  Frame PNGs saved in: {frames_dir}")

    print(f"{'='*60}\n")

    return {
        "gif_path": gif_path,
        "frame_paths": frame_paths,
        "projector_info": proj_info,
        "sanity_check": sanity_check,
        "anchor_strategy": anchor_strategy,
        "n_epochs": len(epochs),
        "config": config.to_dict(),
    }


def _generate_frame(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    output_path: Path,
    config: VizConfig,
    metadata: Dict[str, Any],
    proj_info: Dict[str, Any],
    axis_limits: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    """
    Generate a single frame image.

    Args:
        embeddings_2d: 2D projected embeddings (N, 2)
        labels: Class labels (N,)
        epoch: Epoch number
        output_path: Path to save frame
        config: Visualization config
        metadata: Per-epoch metadata (loss, accuracy, etc.)
        proj_info: Projection information
        axis_limits: Optional (x_min, x_max, y_min, y_max) for consistent frame sizes

    Returns:
        Path to saved frame
    """
    fig, ax = plt.subplots(figsize=config.gif_figsize, dpi=config.gif_dpi)

    # Get unique labels and colors
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Use tab10 for <=10 classes, tab20 for more
    if n_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_classes]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    # Scatter plot colored by class
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f"Class {int(label)}",
            alpha=0.6,
            s=15,
            edgecolors='none',
        )

    # Axis labels with projection info
    method_name = proj_info.get('method', 'projection').upper()
    ax.set_xlabel(f"{method_name} 1", fontsize=11)
    ax.set_ylabel(f"{method_name} 2", fontsize=11)

    # Apply consistent axis limits for animated GIF (ensures same frame size)
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

    # Title with epoch
    title = f"Epoch {epoch}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Overlay text
    overlay_text = _get_overlay_text(config, epoch, metadata, proj_info)
    if overlay_text:
        ax.text(
            0.02, 0.98, overlay_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # Legend
    if n_classes <= 15:
        ax.legend(
            loc='upper right',
            fontsize=7,
            ncol=2 if n_classes > 5 else 1,
            framealpha=0.8,
        )

    ax.grid(True, alpha=0.3)

    # Save frame
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.gif_dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def _get_overlay_text(
    config: VizConfig,
    epoch: int,
    metadata: Dict[str, Any],
    proj_info: Dict[str, Any],
) -> str:
    """Generate overlay text for a frame."""
    if config.gif_overlay == GifOverlay.NONE:
        return ""

    lines = []

    if config.gif_overlay == GifOverlay.EPOCH:
        lines.append(f"Epoch: {epoch}")

    if config.gif_overlay == GifOverlay.LOSS:
        if "train_loss" in metadata:
            lines.append(f"Train Loss: {metadata['train_loss']:.4f}")
        if "val_loss" in metadata:
            lines.append(f"Val Loss: {metadata['val_loss']:.4f}")

    # Add variance info for PCA
    if config.gif_method == GifMethod.PCA:
        var_explained = proj_info.get('total_variance_explained')
        if var_explained:
            lines.append(f"Var: {var_explained:.1%}")

    return "\n".join(lines)


def _create_gif(
    frame_paths: List[Path],
    output_path: Path,
    fps: float,
    loop: int = 0,
) -> None:
    """
    Stitch frames into an animated GIF.

    Args:
        frame_paths: List of frame image paths (in order)
        output_path: Path for output GIF
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for GIF generation. Install with:\n"
            "  pip install imageio"
        )

    if not frame_paths:
        raise ValueError("No frames provided for GIF creation")

    # Read frames
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        images = [imageio.imread(str(p)) for p in frame_paths]

    # Calculate duration per frame
    duration = 1.0 / fps

    # Write GIF
    imageio.mimsave(
        str(output_path),
        images,
        duration=duration,
        loop=loop,
    )

    # Verify the GIF is animated (has multiple frames)
    _verify_gif_animated(output_path, len(images))


def _verify_gif_animated(output_path: Path, expected_frames: int) -> None:
    """
    Verify the generated GIF is truly animated.

    Args:
        output_path: Path to the GIF file
        expected_frames: Expected number of frames

    Raises:
        RuntimeError: If GIF is not animated or frame count is wrong
    """
    try:
        from PIL import Image
        with Image.open(output_path) as img:
            actual_frames = 0
            try:
                while True:
                    actual_frames += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if actual_frames < 2:
                raise RuntimeError(
                    f"GIF appears to be static (only {actual_frames} frame). "
                    f"Expected {expected_frames} frames for animation."
                )

            if actual_frames != expected_frames:
                warnings.warn(
                    f"GIF has {actual_frames} frames but expected {expected_frames}. "
                    f"Some frames may have been skipped.",
                    UserWarning
                )
    except ImportError:
        # PIL not available for verification, skip
        pass


def _check_embedding_changes(
    embeddings_by_epoch: Dict[int, Dict[str, np.ndarray]],
    epochs: List[int],
) -> Dict[str, Any]:
    """
    Sanity check: verify embeddings change across epochs.

    Returns analysis of how much embeddings changed from first to last epoch.
    """
    if len(epochs) < 2:
        return {"changed": True, "message": "Insufficient epochs for comparison"}

    first_epoch = min(epochs)
    last_epoch = max(epochs)

    first_emb = embeddings_by_epoch[first_epoch]["embeddings"]
    last_emb = embeddings_by_epoch[last_epoch]["embeddings"]

    # Compare embeddings (use smaller of the two sizes)
    n = min(len(first_emb), len(last_emb))
    first_emb = first_emb[:n]
    last_emb = last_emb[:n]

    # Compute change metrics
    diff = np.abs(first_emb - last_emb)
    mean_change = np.mean(diff)
    relative_change = mean_change / (np.mean(np.abs(first_emb)) + 1e-10)

    # Check if embeddings are essentially unchanged
    is_unchanged = relative_change < 0.01  # Less than 1% change

    return {
        "changed": not is_unchanged,
        "mean_absolute_change": float(mean_change),
        "relative_change": float(relative_change),
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "warning": (
            "Embeddings appear unchanged across epochs! "
            "This may indicate a training issue."
            if is_unchanged else None
        ),
    }


def _check_embedding_changes_anchored(
    embeddings_by_epoch: Dict[int, Dict[str, np.ndarray]],
    epochs: List[int],
    anchor_indices: np.ndarray,
) -> Dict[str, Any]:
    """
    Sanity check: verify embeddings change across epochs using anchor points.

    This version uses the same anchor indices for consistent comparison,
    which is essential when per-epoch embedding counts differ.

    Args:
        embeddings_by_epoch: Dict mapping epoch -> {'embeddings': array, 'labels': array}
        epochs: Sorted list of epoch numbers
        anchor_indices: Fixed indices to track across epochs

    Returns:
        Dict with change analysis:
        - changed: bool indicating if embeddings changed significantly
        - mean_absolute_change: average absolute difference
        - relative_change: change relative to embedding magnitude
        - first_epoch, last_epoch: epochs compared
        - anchor_count: number of anchor points used
        - warning: optional warning message
    """
    if len(epochs) < 2:
        return {"changed": True, "message": "Insufficient epochs for comparison"}

    first_epoch = min(epochs)
    last_epoch = max(epochs)

    first_emb = embeddings_by_epoch[first_epoch]["embeddings"]
    last_emb = embeddings_by_epoch[last_epoch]["embeddings"]

    # Use anchor indices that are valid for BOTH epochs
    valid_first = anchor_indices[anchor_indices < len(first_emb)]
    valid_last = anchor_indices[anchor_indices < len(last_emb)]
    valid_both = np.intersect1d(valid_first, valid_last)

    if len(valid_both) == 0:
        return {
            "changed": True,
            "warning": "No common anchor points between first and last epoch",
            "anchor_count": 0,
            "first_epoch": first_epoch,
            "last_epoch": last_epoch,
        }

    # Extract anchor embeddings
    first_anchored = first_emb[valid_both]
    last_anchored = last_emb[valid_both]

    # Compute change metrics
    diff = np.abs(first_anchored - last_anchored)
    mean_change = np.mean(diff)
    relative_change = mean_change / (np.mean(np.abs(first_anchored)) + 1e-10)

    # Check if embeddings are essentially unchanged
    is_unchanged = relative_change < 0.01  # Less than 1% change

    return {
        "changed": not is_unchanged,
        "mean_absolute_change": float(mean_change),
        "relative_change": float(relative_change),
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "anchor_count": len(valid_both),
        "warning": (
            "Embeddings appear unchanged across epochs! "
            "This may indicate a training issue."
            if is_unchanged else None
        ),
    }


def regenerate_gif_from_run(
    run_dir: Path,
    config: Optional[VizConfig] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Regenerate a GIF from a previous experiment run.

    This function enables re-visualization without retraining:
    1. Loads saved epoch embeddings from run_dir
    2. Generates new GIF with current config settings

    Args:
        run_dir: Path to experiment run directory
        config: VizConfig (uses defaults if None)
        output_path: Custom output path (defaults to run_dir/training_dynamics.gif)

    Returns:
        Same as generate_training_gif()

    Example:
        >>> # Regenerate GIF with different settings
        >>> result = regenerate_gif_from_run(
        ...     run_dir="playground_runs/my_experiment",
        ...     config=VizConfig(gif_method='tsne', gif_fps=1.0)
        ... )
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"\nRegenerating GIF from: {run_dir}")

    # Load epoch embeddings
    try:
        collector = EpochEmbeddingCollector.load(run_dir)
    except FileNotFoundError as e:
        # Try old format (epoch_frames directory with PNGs only)
        epoch_frames_dir = run_dir / "epoch_frames"
        if epoch_frames_dir.exists():
            raise FileNotFoundError(
                f"Found old-format epoch_frames but no saved embeddings.\n"
                f"Cannot regenerate GIF without saved embeddings.\n"
                f"Please re-run the experiment with the new visualization system."
            )
        raise e

    embeddings_by_epoch = collector.get_all_epochs()

    if not embeddings_by_epoch:
        raise ValueError(f"No epoch embeddings found in {run_dir}")

    # Use provided config or defaults
    if config is None:
        config = VizConfig()

    # Determine output directory
    if output_path is None:
        output_dir = run_dir
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference labels from any epoch
    first_epoch = min(embeddings_by_epoch.keys())
    labels = embeddings_by_epoch[first_epoch]["labels"]

    # Load metadata if available
    epoch_metadata = {}
    logs_path = run_dir / "logs.json"
    if logs_path.exists():
        import json
        with open(logs_path) as f:
            logs = json.load(f)
        # Convert logs to per-epoch metadata
        train_losses = logs.get("train_losses", [])
        val_losses = logs.get("val_losses", [])
        for i, epoch in enumerate(sorted(embeddings_by_epoch.keys())):
            epoch_metadata[epoch] = {}
            if i < len(train_losses):
                epoch_metadata[epoch]["train_loss"] = train_losses[i]
            if i < len(val_losses):
                epoch_metadata[epoch]["val_loss"] = val_losses[i]

    # Load random state from config if available
    config_path = run_dir / "config.json"
    random_state = 42
    if config_path.exists():
        import json
        with open(config_path) as f:
            run_config = json.load(f)
        random_state = run_config.get("seed", 42)

    # Generate GIF
    result = generate_training_gif(
        embeddings_by_epoch=embeddings_by_epoch,
        labels=labels,
        config=config,
        output_dir=output_dir,
        epoch_metadata=epoch_metadata,
        random_state=random_state,
    )

    # Move GIF to custom path if specified
    if output_path is not None and output_path != result["gif_path"]:
        import shutil
        shutil.move(result["gif_path"], output_path)
        result["gif_path"] = output_path

    return result
