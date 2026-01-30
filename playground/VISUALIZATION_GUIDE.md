# I-Con Playground Visualization Guide

A PhD researcher's guide to training dynamics visualization and GIF generation.

## Overview

The I-Con Playground provides a configurable visualization system for understanding how learned representations evolve during training. This guide covers:

1. **Visualization modes** - What to output and when
2. **GIF configuration** - Fine-grained control over animations
3. **Projection methods** - PCA, t-SNE, and UMAP options
4. **Regenerating GIFs** - Post-hoc visualization without retraining
5. **Reproducibility** - Deterministic outputs for papers

---

## Quick Start

### Default behavior (recommended for first runs)

```bash
# GIF + static plots generated automatically
python -m playground.playground_cli --dataset cifar10 --epochs 10
```

### Disable visualization (for headless servers)

```bash
python -m playground.playground_cli --dataset cifar10 --epochs 100 --viz_mode none
```

### GIF only (skip static plots)

```bash
python -m playground.playground_cli --dataset cifar10 --epochs 50 --viz_mode gif
```

---

## Visualization Modes

Control what outputs are generated with `--viz_mode`:

| Mode | GIF | Static Plots | Use Case |
|------|-----|--------------|----------|
| `none` | No | No | Headless runs, CI/CD |
| `static` | No | Yes | Quick final snapshot |
| `gif` | Yes | No | Focus on dynamics |
| `both` | Yes | Yes | **Default** - full analysis |

---

## GIF Configuration

### Frame Frequency

Save frames every N epochs with `--gif_every`:

```bash
# Frame every 5 epochs (faster, smaller GIF)
python -m playground.playground_cli --gif_every 5 --epochs 50

# Frame every epoch (detailed but larger)
python -m playground.playground_cli --gif_every 1 --epochs 20
```

**PhD tip:** For long runs (>100 epochs), use `--gif_every 10` to keep GIFs manageable.

### Projection Method

Choose how embeddings are projected to 2D with `--gif_method`:

```bash
# PCA (default) - fast, linear, shows variance
python -m playground.playground_cli --gif_method pca

# t-SNE - slower, non-linear, better cluster separation
python -m playground.playground_cli --gif_method tsne --gif_every 10

# UMAP - requires umap-learn, good balance of speed/quality
python -m playground.playground_cli --gif_method umap --gif_every 5
```

| Method | Speed | Consistency | Best For |
|--------|-------|-------------|----------|
| PCA | Fast | Deterministic | Quick iteration, paper figures |
| t-SNE | Slow | Variable | Final presentations |
| UMAP | Medium | Deterministic | Balanced quality/speed |

**Important:** PCA is fitted ONCE on final epoch embeddings and reused for all frames.
This ensures consistent axes across the animation.

### Playback Speed

Control GIF speed with `--gif_fps`:

```bash
# Slower playback (0.5 fps = 2 seconds per frame)
python -m playground.playground_cli --gif_fps 0.5

# Faster playback (4 fps = 0.25 seconds per frame)
python -m playground.playground_cli --gif_fps 4
```

### Performance Tuning

For large datasets, subsample points with `--gif_max_points`:

```bash
# Subsample to 2000 points (faster rendering)
python -m playground.playground_cli --gif_max_points 2000

# Full dataset (slower but complete)
python -m playground.playground_cli --gif_max_points 50000
```

### Frame Overlay

Add text to frames with `--gif_overlay`:

```bash
# Show epoch number (default)
python -m playground.playground_cli --gif_overlay epoch

# Show loss values
python -m playground.playground_cli --gif_overlay loss

# No overlay
python -m playground.playground_cli --gif_overlay none
```

### Keep or Discard Frames

By default, individual frame PNGs are saved. To discard them:

```bash
# Only keep the final GIF
python -m playground.playground_cli --no_save_frames
```

---

## Regenerating GIFs

**Key feature for PhD workflows:** Re-visualize without retraining.

### Basic regeneration

```bash
# Regenerate GIF from a previous run
python -m playground.playground_cli --regen_gif --load_dir playground_runs/my_experiment
```

### Change visualization settings

```bash
# Re-render with t-SNE instead of PCA
python -m playground.playground_cli \
  --regen_gif \
  --load_dir playground_runs/my_experiment \
  --gif_method tsne

# Slow down playback
python -m playground.playground_cli \
  --regen_gif \
  --load_dir playground_runs/my_experiment \
  --gif_fps 0.5
```

### Use case: Paper figures

```bash
# Run experiment once
python -m playground.playground_cli --dataset cifar10 --epochs 100

# Try different visualizations without retraining
python -m playground.playground_cli --regen_gif --load_dir playground_runs/run_001 --gif_method pca --gif_fps 1
python -m playground.playground_cli --regen_gif --load_dir playground_runs/run_001 --gif_method tsne --gif_fps 1
python -m playground.playground_cli --regen_gif --load_dir playground_runs/run_001 --gif_method umap --gif_fps 1
```

---

## Reproducibility

### Deterministic outputs

GIF generation is deterministic when:
1. Same random seed (`--seed`)
2. Same projection method
3. Same embeddings

```bash
# Reproducible run
python -m playground.playground_cli --seed 42 --gif_method pca
```

### What's logged

All visualization settings are saved in `experiment_manifest.json`:

```json
{
  "visualization": {
    "viz_mode": "both",
    "gif_every": 1,
    "gif_method": "pca",
    "gif_fps": 2.0,
    "gif_max_points": 5000,
    "gif_overlay": "epoch"
  }
}
```

---

## Output Structure

```
playground_runs/my_experiment/
├── config.json                  # Full config
├── experiment_manifest.json     # Includes viz settings
├── embeddings.npz               # Final embeddings
├── logs.json                    # Training metrics
├── final_model.pt               # Model checkpoint
├── training_curves.png          # Loss/accuracy plots
├── embeddings_pca.png           # Static embedding plot
├── training_dynamics.gif        # Animated visualization
└── epoch_embeddings/            # Per-epoch data (for regen)
    ├── metadata.json
    ├── epoch_0001.npz
    ├── epoch_0005.npz
    └── ...
└── epoch_frames/                # Frame images (if save_frames=True)
    ├── epoch_0001.png
    ├── epoch_0005.png
    └── ...
```

---

## Typical PhD Workflows

### 1. Quick exploration

```bash
# Fast iteration with minimal viz
python -m playground.playground_cli \
  --dataset mnist \
  --epochs 5 \
  --viz_mode static
```

### 2. Full experiment with dynamics

```bash
# Complete analysis
python -m playground.playground_cli \
  --dataset cifar10 \
  --backbone resnet18 \
  --icon_mode simclr_like \
  --epochs 50 \
  --viz_mode both \
  --gif_every 5 \
  --gif_method pca
```

### 3. Long training, visualize later

```bash
# Training (headless server)
python -m playground.playground_cli \
  --dataset cifar10 \
  --epochs 200 \
  --viz_mode gif \
  --gif_every 10

# Later: regenerate with different settings
python -m playground.playground_cli \
  --regen_gif \
  --load_dir playground_runs/experiment_001 \
  --gif_fps 1 \
  --gif_method tsne
```

### 4. Comparing runs

```bash
# Run multiple experiments
for mode in simclr_like sne_like tsne_like; do
  python -m playground.playground_cli \
    --icon_mode $mode \
    --epochs 30 \
    --gif_every 5
done

# GIFs are in each run directory
ls playground_runs/*/training_dynamics.gif
```

---

## Troubleshooting

### GIF not generated

1. Check `--viz_mode` is `gif` or `both`
2. Ensure enough epochs ran (need at least 1 frame)
3. Check for errors in console output

### GIF too large

- Increase `--gif_every` (fewer frames)
- Decrease `--gif_max_points` (faster rendering)
- Use PCA instead of t-SNE/UMAP

### Embeddings look identical across epochs

This is detected automatically. You'll see:

```
WARNING: Embeddings appear unchanged across epochs!
This may indicate a training issue.
```

Possible causes:
- Frozen model
- Zero gradients
- Learning rate too low

### UMAP not available

```bash
pip install umap-learn
```

### t-SNE is slow

- Use PCA for quick iterations
- Increase `--gif_every` to reduce frames
- Use t-SNE only for final figures

---

## API Usage (Python)

```python
from playground.viz import generate_training_gif, VizConfig, VizMode, GifMethod
from playground.viz.embedding_collector import EpochEmbeddingCollector
from pathlib import Path

# Load saved embeddings
collector = EpochEmbeddingCollector.load(Path("playground_runs/my_run"))
embeddings_by_epoch = collector.get_all_epochs()

# Configure visualization
config = VizConfig(
    viz_mode=VizMode.GIF,
    gif_method=GifMethod.PCA,
    gif_fps=2.0,
    gif_max_points=5000,
)

# Generate GIF
result = generate_training_gif(
    embeddings_by_epoch=embeddings_by_epoch,
    labels=embeddings_by_epoch[1]['labels'],
    config=config,
    output_dir=Path("custom_output"),
    random_state=42,
)

print(f"GIF saved to: {result['gif_path']}")
print(f"Variance explained: {result['projector_info']['total_variance_explained']:.1%}")
```

---

## CLI Reference

```
Visualization controls:
  --viz_mode {none,static,gif,both}
                        Visualization output mode (default: both)
  --gif_every N         Save frame every N epochs (default: 1)
  --gif_method {pca,tsne,umap}
                        Projection method for GIF (default: pca)
  --gif_fps FLOAT       Frames per second (default: 2.0)
  --gif_max_points INT  Max points to plot (default: 5000)
  --gif_overlay {none,loss,epoch}
                        Text overlay on frames (default: epoch)
  --no_save_frames      Don't save individual frame PNGs

Regeneration:
  --regen_gif           Regenerate GIF from saved embeddings
  --load_dir PATH       Directory to load experiment from
```

---

## Best Practices

1. **Start with PCA** - Fast, deterministic, good for iteration
2. **Use `--gif_every 5-10`** for long runs (>50 epochs)
3. **Save embeddings** (default) - Enables post-hoc regeneration
4. **Check sanity warnings** - Unchanged embeddings indicate training issues
5. **Use `--regen_gif`** for paper figures - Try different methods without retraining
