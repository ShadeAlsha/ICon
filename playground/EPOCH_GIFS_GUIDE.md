# Epoch-by-Epoch GIF Visualization Guide

## Why Learning Dynamics Matter

Static PCA plots show the **final state** of your embeddings, but they hide the **learning journey**. Different objectives (SimCLR vs SNE vs supervised) may converge to similar final configurations but take dramatically different paths to get there.

**Epoch-by-epoch GIFs reveal:**
- **Convergence speed**: How quickly does the embedding space stabilize?
- **Learning patterns**: Does the model gradually refine clusters, or does it reorganize suddenly?
- **Objective differences**: Do SimCLR and SNE really learn differently, or just converge to the same solution at different rates?
- **Failure modes**: Does the model get stuck in local minima? Does it oscillate?

## Quick Start

### Basic Usage

Enable GIF generation with the `--save_epoch_gifs` flag:

```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --epochs 10 \
  --save_epoch_gifs
```

This creates:
```
playground_runs/mnist_mlp_simclr_like_<timestamp>/
├── epoch_frames/
│   ├── epoch_001.png
│   ├── epoch_002.png
│   ├── ...
│   └── epoch_010.png
└── training_dynamics.gif  # ← Animated visualization!
```

### Comparing Objectives

Compare learning dynamics across different I-Con modes:

```bash
python -m playground.playground_cli \
  --recipe simclr_vs_sne \
  --dataset mnist \
  --epochs 10 \
  --save_epoch_gifs
```

This creates two GIFs showing how SimCLR and SNE embeddings evolve differently.

### With Custom Hyperparameters

Test how temperature affects learning dynamics:

```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --epochs 10 \
  --temperature 0.1 \
  --divergence js \
  --save_epoch_gifs
```

## Technical Details

### How It Works

1. **After each epoch**, the system:
   - Runs validation to collect embeddings
   - Projects embeddings to 2D using PCA
   - Saves a frame as PNG

2. **PCA consistency**: The PCA projection is fit **once** (on the first epoch) and reused for all subsequent epochs. This ensures the axes remain consistent across frames, making the animation meaningful.

3. **After training completes**:
   - All frames are stitched into an animated GIF using `imageio`
   - Each frame lasts 0.5 seconds (configurable)
   - GIF loops infinitely

### Performance Impact

- **Training time**: ~10-15% slower (extra validation pass per epoch)
- **Disk space**: ~200KB per frame + ~200KB for final GIF
  - For 10 epochs: ~2.2MB total
  - For 100 epochs: ~22MB total

### Subsampling

To keep visualization fast, the system automatically subsamples to 2000 points per frame if you have more samples. This is transparent and doesn't affect training.

## Advanced Usage

### Programmatic API

You can use the epoch GIF manager directly in custom training scripts:

```python
from playground.epoch_gif_utils import EpochGIFManager
from pathlib import Path

# Create manager
manager = EpochGIFManager(
    output_dir=Path("my_experiment/frames"),
    max_samples=2000,  # Subsample if more
    figsize=(10, 8),
    dpi=100,
)

# During training loop
for epoch in range(epochs):
    # ... train ...

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, val_loader)

    # Save frame
    manager.save_epoch_frame(
        embeddings=embeddings,
        labels=labels,
        epoch=epoch + 1,
    )

# After training
manager.create_gif("training_dynamics.gif", duration=0.5)
```

### Consistent PCA Across Experiments

If you want to compare multiple experiments with the same PCA projection:

```python
# Fit PCA on final embeddings from reference experiment
manager = EpochGIFManager(output_dir=Path("exp1/frames"))
manager.fit_pca(reference_embeddings)

# Now all experiments use the same PCA axes
for experiment_data in experiments:
    manager.save_epoch_frame(
        embeddings=experiment_data["embeddings"],
        labels=experiment_data["labels"],
        epoch=experiment_data["epoch"],
    )
```

## Interpretation Guide

### What to Look For

1. **Cluster formation**
   - Do clusters emerge gradually or suddenly?
   - Do they drift or remain stable once formed?

2. **Inter-cluster relationships**
   - Do similar classes (e.g., "3" and "8") start together and separate?
   - Are cluster positions stable or do they reorganize?

3. **Convergence behavior**
   - Does learning plateau early?
   - Are there distinct phases (e.g., rapid clustering then fine-tuning)?

4. **Objective-specific patterns**
   - **SimCLR**: Often shows rapid clustering with strong separation
   - **SNE**: May show more gradual, smooth cluster refinement
   - **t-SNE**: Often creates tighter, more compact clusters
   - **Supervised**: Typically shows fastest convergence to class-separated structure

### Example Observations

**Fast convergence** (good):
- Clusters form by epoch 2-3
- Positions stabilize quickly
- Suggests efficient optimization

**Slow convergence** (may need tuning):
- Clusters still moving at epoch 8-10
- Try: higher learning rate, better initialization, different backbone

**Oscillation** (bad):
- Clusters drift back and forth
- Try: lower learning rate, different optimizer, gradient clipping

**Collapse** (very bad):
- All points converge to a single cluster
- Try: different temperature, regularization, check for bugs

## Requirements

The GIF feature requires `imageio`:

```bash
pip install imageio
```

Or if using conda:

```bash
conda install imageio
```

This is automatically checked when you use `--save_epoch_gifs`, and you'll get a helpful error message if it's missing.

## Troubleshooting

### "imageio is required for GIF generation"

Install imageio:
```bash
pip install imageio
```

### GIF is too fast/slow

Modify the duration in `playground_runner.py`:
```python
manager.create_gif(gif_path, duration=1.0)  # 1 second per frame
```

### Too many frames (long training)

The system saves every epoch. For very long training (100+ epochs), consider:
- Saving frames only every N epochs (requires code modification)
- Using a lower frame rate (longer duration per frame)

### PCA axes inconsistent

Make sure `auto_fit_pca=True` (default) in the first epoch. The system automatically handles this.

## Research Use Cases

1. **Method comparison**: Compare SimCLR, SNE, supervised learning dynamics
2. **Hyperparameter tuning**: Visualize effect of temperature, learning rate
3. **Architecture study**: Does ResNet18 vs ResNet34 learn differently?
4. **Dataset analysis**: Do some datasets have smoother learning curves?
5. **Debugging**: Identify when/why training fails

## Examples

### Example 1: Temperature Sweep

```bash
for temp in 0.1 0.5 1.0; do
    python -m playground.playground_cli \
      --dataset mnist \
      --temperature $temp \
      --epochs 10 \
      --save_epoch_gifs
done
```

Compare the three GIFs to see how temperature affects learning dynamics.

### Example 2: Recipe with GIFs

```bash
python -m playground.playground_cli \
  --recipe kernel_comparison \
  --dataset cifar10 \
  --epochs 20 \
  --save_epoch_gifs
```

Creates GIFs for Gaussian, Student-t, and Barlow Twins kernels.

### Example 3: Custom Experiment

```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone simplecnn \
  --icon_mode tsne_like \
  --epochs 15 \
  --temperature 1.0 \
  --divergence rkl \
  --save_epoch_gifs
```

See how t-SNE with reverse KL divergence evolves.

## Citation

If you use the epoch GIF visualization in your research, consider citing:

```
@software{icon_playground_gifs,
  title={I-Con Playground: Epoch-by-Epoch Visualization},
  author={[Your Name]},
  year={2026},
  note={Animated PCA visualizations for representation learning dynamics}
}
```

## Future Improvements

Potential enhancements (contributions welcome):
- [ ] Support for t-SNE projection (slower but often better for visualization)
- [ ] Support for UMAP projection
- [ ] Side-by-side comparison GIFs (multiple experiments in one animation)
- [ ] Overlay training loss/accuracy curves on frames
- [ ] Interactive HTML5 visualizations
- [ ] 3D rotating projections

## Feedback

Have suggestions? Found a bug? Open an issue on GitHub or contact the I-Con team.
