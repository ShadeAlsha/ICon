# I-Con Playground

**A student-friendly layer for experimenting with I-Con configurations**

---

## What is I-Con Playground?

I-Con Playground is a small educational layer on top of the official I-Con implementation. It lets you run small I-Con experiments on datasets like CIFAR-10 and MNIST, switch between preset I-Con modes (e.g., SimCLR-like, SNE-like, t-SNE-like), and visualize the resulting representation space with simple probes.

This tool is designed for:
- **Students** at workshops like MLSS who want to understand representation learning
- **Researchers** who want to quickly prototype I-Con configurations
- **Educators** who want to demonstrate how different objectives affect learned representations

**Note**: This playground is for teaching and exploration, not for reproducing the full paper's ImageNet-scale experiments.

---

## Quickstart

### Installation

The playground uses the same dependencies as the main I-Con repository, plus:

```bash
pip install matplotlib scikit-learn
```

### CLI Usage

Run an experiment from the command line:

```bash
# SimCLR-like contrastive learning on CIFAR-10
python -m playground.playground_cli \
    --dataset cifar10 \
    --backbone resnet18 \
    --icon_mode simclr_like \
    --epochs 10 \
    --batch_size 256 \
    --out_dir runs/cifar10_simclr

# Use a preset for quick start
python -m playground.playground_cli \
    --preset cifar_contrastive \
    --epochs 5

# t-SNE-like embedding on MNIST
python -m playground.playground_cli \
    --dataset mnist \
    --backbone simplecnn \
    --icon_mode tsne_like \
    --epochs 20
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset to use | `cifar10` |
| `--backbone` | Encoder architecture | `resnet18` |
| `--icon_mode` | I-Con objective preset | `simclr_like` |
| `--epochs` | Number of training epochs | `10` |
| `--batch_size` | Training batch size | `256` |
| `--lr` | Learning rate | `1e-3` |
| `--temperature` | Softmax temperature | `0.5` |
| `--out_dir` | Output directory | `playground_runs` |
| `--no_viz` | Skip visualization | `False` |
| `--tsne` | Use t-SNE instead of PCA | `False` |
| `--probe_only` | Only run probes on existing embeddings | `False` |

### Notebook Usage

Launch the interactive notebook:

```bash
jupyter notebook playground/notebooks/ICon_Playground.ipynb
```

Or open in Google Colab (upload the notebook and the playground directory).

### Python API

```python
from playground import PlaygroundConfig, run_playground_experiment
from playground import plot_training_curves, plot_embeddings_2d, run_linear_probe

# Configure experiment
config = PlaygroundConfig(
    dataset="cifar10",
    backbone="resnet18",
    icon_mode="simclr_like",
    epochs=10,
    batch_size=256,
)

# Run training
results = run_playground_experiment(config)

# Visualize results
plot_training_curves(results["logs"])
plot_embeddings_2d(results["embeddings"], results["labels"])

# Evaluate embeddings
probe_results = run_linear_probe(results["embeddings"], results["labels"])
print(f"Linear probe accuracy: {probe_results['test_accuracy']:.2%}")
```

---

## I-Con Modes

The playground provides preset configurations that recover different representation learning objectives:

| Mode | Description | Supervisory Distribution | Learned Distribution |
|------|-------------|-------------------------|---------------------|
| `simclr_like` | Contrastive learning (InfoNCE-style) | Augmentation (same sample = neighbors) | Gaussian kernel with cosine distance |
| `sne_like` | SNE-style dimensionality reduction | Augmentation | Gaussian kernel with Euclidean distance |
| `tsne_like` | t-SNE-style embedding | Augmentation | Student-t kernel (heavier tails) |
| `supervised` | Supervised learning | Label (same class = neighbors) | Gaussian kernel with cosine distance |
| `cluster_like` | Clustering-oriented | Augmentation | Student-t with learnable temperature |

---

## Output Structure

Each experiment creates a directory with:

```
playground_runs/
└── cifar10_resnet18_simclr_like_20240101_120000/
    ├── config.json           # Experiment configuration
    ├── logs.json             # Training logs (loss, accuracy per epoch)
    ├── embeddings.npz        # Learned embeddings and labels
    ├── final_model.pt        # Trained model weights
    ├── training_curves.png   # Loss/accuracy plots
    ├── embeddings_pca.png    # PCA visualization
    └── experiment_summary.png # Combined summary figure
```

---

## Intended Use

This playground is aimed at:

- **MLSS-level students** learning about representation learning
- **Quick prototyping** of I-Con configurations
- **Educational demonstrations** of how objectives affect embeddings

It is NOT designed for:
- Reproducing full paper experiments (ImageNet-scale training)
- Production use
- Rigorous benchmarking

---

## Architecture

```
playground/
├── __init__.py              # Package exports
├── playground_config.py     # Configuration dataclass
├── playground_runner.py     # Experiment execution
├── playground_viz.py        # Visualization utilities
├── playground_probes.py     # Evaluation probes
├── playground_cli.py        # Command-line interface
├── README_PLAYGROUND.md     # This file
└── notebooks/
    └── ICon_Playground.ipynb # Interactive notebook
```

The playground hooks into the existing I-Con code:
- Uses `dataloaders.get_dataloaders()` for data loading
- Uses `mappers.ResNet`, `mappers.SimpleCNN` for encoders
- Uses `distributions.Gaussian`, `distributions.StudentT`, `distributions.Augmentation` for kernels
- Uses `model.Model` (PyTorch Lightning module) for training

---

## Acknowledgment

All core I-Con logic comes from the original authors of "I-Con: A Unifying Framework for Representation Learning". This playground is an additive convenience layer for educational purposes.

---

## Troubleshooting

### "No module named 'playground'"

Make sure you're running from the I-Con repository root:

```bash
cd /path/to/ICon
python -m playground.playground_cli --help
```

### Out of memory

Reduce batch size:

```bash
python -m playground.playground_cli --batch_size 64
```

### Slow training

Use a smaller backbone or fewer epochs:

```bash
python -m playground.playground_cli --backbone resnet18 --epochs 5
```

### No GPU available

The playground will automatically fall back to CPU, or force it:

```bash
python -m playground.playground_cli --cpu
```

---

## Preset Gallery

The playground provides 8 I-Con presets that recover different representation learning methods:

### Original Presets

- **`simclr_like`**: InfoNCE-style contrastive learning with augmentation pairs and cosine Gaussian kernel. Learns by pulling augmented views of the same image together.

- **`sne_like`**: SNE-style neighborhood preservation with Euclidean Gaussian kernel. Focuses on preserving local neighborhood structure in the embedding space.

- **`tsne_like`**: t-SNE-style embedding with heavy-tailed Student-t kernel. Better for visualization due to reduced crowding, handles both local and global structure.

- **`supervised`**: Supervised learning with label-based supervisory distribution. Directly uses class labels as the supervisory signal.

- **`cluster_like`**: Clustering-oriented with learnable Student-t temperature. Adapts the kernel sharpness during training for better cluster formation.

### New Presets (Extended)

- **`barlow_twins_like`**: Decorrelation and redundancy reduction style. Uses sharper Gaussian kernel (temp × 0.3) to encourage independent, decorrelated features.

- **`vicreg_like`**: Variance-invariance-covariance regularization style. Uses learnable Student-t kernel with heavy tails to encourage spread and adaptive covariance structure.

- **`debiasing_like`**: Balanced similarity structure to avoid amplifying biases. Uses moderate temperature (min 0.7) to prevent overly sharp distributions that might reinforce spurious correlations.

---

## Example Recipes

Recipes provide pre-configured multi-run experiments for comparative analysis:

### CLI Usage

```bash
# Compare SimCLR vs SNE
python -m playground.playground_cli --recipe simclr_vs_sne --dataset cifar10 --epochs 10

# Compare different backbones
python -m playground.playground_cli --recipe backbone_comparison --epochs 5

# Supervised vs unsupervised
python -m playground.playground_cli --recipe supervised_vs_unsupervised --epochs 10

# Temperature sweep
python -m playground.playground_cli --recipe temperature_sweep --epochs 10

# Full comparison of all major modes
python -m playground.playground_cli --recipe full_comparison --epochs 10
```

### Python API

```python
from playground.playground_recipes import get_recipe
from playground.playground_runner import run_playground_experiment
from playground.playground_viz import compare_embedding_spaces

# Get recipe configs
configs = get_recipe("supervised_vs_unsupervised", dataset="cifar10", epochs=10)

# Run experiments
results_list = [run_playground_experiment(cfg) for cfg in configs]

# Compare visually
compare_embedding_spaces(results_list, method="pca")
```

### Available Recipes

| Recipe | Description | # Configs |
|--------|-------------|-----------|
| `simclr_vs_sne` | Compare contrastive vs SNE-style | 2 |
| `backbone_comparison` | Test different architectures | 2-3 |
| `supervised_vs_unsupervised` | Labels vs augmentations | 2 |
| `temperature_sweep` | Effect of temperature (0.1, 0.5, 1.0) | 3 |
| `kernel_comparison` | Gaussian vs Student-t vs Barlow | 3 |
| `full_comparison` | All major modes | 5 |

---

## Suggested MLSS Exercises

Hands-on exercises for students to explore representation learning:

### Exercise 1: Compare I-Con Modes (30 min)
Run the `simclr_vs_sne` recipe with 10 epochs:
- Which mode produces better linear probe accuracy?
- How do the 2D embeddings differ visually (PCA/t-SNE)?
- Compare the separability ratios - what does this tell you?
- Plot distance histograms for both - how do intra/inter-class distances differ?

**Learning Goal**: Understand how different similarity kernels affect embedding geometry.

### Exercise 2: Temperature Exploration (20 min)
Run `temperature_sweep` with `simclr_like` mode:
- How does temperature=0.1 vs 1.0 affect training stability?
- Which temperature gives the best test accuracy?
- Plot distance histograms for each - what changes?
- What's the trade-off between sharp and soft distributions?

**Learning Goal**: Explore the role of temperature in contrastive learning.

### Exercise 3: Supervised vs Unsupervised (25 min)
Run `supervised_vs_unsupervised` recipe:
- Does supervised always win on linear probe accuracy?
- How many epochs until supervised catches up to unsupervised (if at all)?
- Which produces more visually separated clusters in 2D plots?
- Try limiting labels (supervised) vs augmentation quality (unsupervised) - which is more robust?

**Learning Goal**: Compare label-based vs augmentation-based supervision.

### Exercise 4: Backbone Capacity (20 min)
Run `backbone_comparison` recipe:
- Does ResNet34 always outperform ResNet18?
- What's the accuracy vs training time trade-off?
- Try on MNIST with MLP vs SimpleCNN - which is more efficient?
- Does embedding dimension matter more than backbone depth?

**Learning Goal**: Understand the role of architecture in representation quality.

### Exercise 5: Custom Hypothesis Testing (45 min)
Design your own experiment:
1. Pick a hypothesis (e.g., "Barlow Twins learns more decorrelated features")
2. Choose 2-3 configs to test it
3. Run experiments and create comparison plots
4. Analyze separability metrics and probe accuracies
5. Write a 1-paragraph conclusion

**Learning Goal**: Practice scientific experimentation in ML.

---

## Self-Check & Tests

Before using the playground (or after making changes), verify everything works:

### Quick Self-Check

Run the built-in self-check script:

```bash
python -m playground.self_check
```

This runs a test suite:
- ✓ Imports all modules
- ✓ Creates a minimal config
- ✓ Runs a 1-epoch experiment on MNIST (CPU)
- ✓ Extracts embeddings and validates shapes
- ✓ Runs linear probe and kNN probe
- ✓ Generates visualizations (in-memory)
- ✓ Tests recipe system
- ✓ Validates config error handling

**Expected runtime**: 1-3 minutes (CPU-only)

### Automated Tests

If you have pytest installed, run the full test suite:

```bash
# Install pytest (if needed)
pip install pytest

# Run all tests
pytest tests/test_playground_smoke.py -v

# Run only fast tests (skip slow training tests)
pytest tests/test_playground_smoke.py -v -m "not slow"
```

The test suite includes:
- Configuration validation tests
- Import/module tests
- Minimal training integration tests
- Probe and visualization tests
- Recipe system tests

### Troubleshooting

If self-check fails:

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **PyTorch issues**: Verify PyTorch is working
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. **Memory issues**: Reduce batch size or use smaller backbones
   ```python
   config = PlaygroundConfig(batch_size=32, backbone="mlp")
   ```

4. **GPU issues**: Force CPU mode
   ```bash
   python -m playground.playground_cli --cpu
   ```

For more help, see the Troubleshooting section above or open an issue on GitHub.

---

## Contributing

This playground is designed to be extended. To add new features:

1. **New I-Con modes**: Add to `SUPPORTED_ICON_MODES` in `playground_config.py`
2. **New visualizations**: Add functions to `playground_viz.py`
3. **New probes**: Add evaluation methods to `playground_probes.py`

Keep code simple and well-commented for student readability.
