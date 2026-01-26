# I-Con Playground Quick Reference

**Essential commands for using the playground**

---

## 🚀 Getting Started

### 1. Verify Installation
```bash
python -m playground.self_check
```
Expected: All tests pass in 1-3 minutes

### 2. Your First Experiment
```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --icon_mode simclr_like \
  --epochs 5 \
  --cpu
```

---

## 📋 Common Commands

### Single Experiments

```bash
# CIFAR-10 with ResNet-18 (CPU)
python -m playground.playground_cli \
  --dataset cifar10 \
  --backbone resnet18 \
  --icon_mode simclr_like \
  --epochs 10 \
  --cpu

# MNIST with SimpleCNN (fast)
python -m playground.playground_cli \
  --dataset mnist \
  --backbone simplecnn \
  --icon_mode tsne_like \
  --epochs 20 \
  --cpu

# CIFAR-10 supervised (if you have GPU)
python -m playground.playground_cli \
  --dataset cifar10 \
  --backbone resnet18 \
  --icon_mode supervised \
  --epochs 10 \
  --gpu
```

### Recipes (Multi-Experiment)

```bash
# Compare SimCLR vs SNE
python -m playground.playground_cli \
  --recipe simclr_vs_sne \
  --dataset cifar10 \
  --epochs 10

# Compare backbones
python -m playground.playground_cli \
  --recipe backbone_comparison \
  --dataset cifar10 \
  --epochs 5

# Full comparison (5 experiments)
python -m playground.playground_cli \
  --recipe full_comparison \
  --dataset mnist \
  --epochs 5
```

### Visualization Only

```bash
# Re-run probes and viz on existing results
python -m playground.playground_cli \
  --probe_only \
  --load_dir playground_runs/your_experiment_name
```

---

## 🐍 Python API

### Basic Usage

```python
from playground import PlaygroundConfig, run_playground_experiment

# Configure
config = PlaygroundConfig(
    dataset="cifar10",
    backbone="resnet18",
    icon_mode="simclr_like",
    epochs=10,
    batch_size=256,
)

# Run
results = run_playground_experiment(config, gpu=False)

# Access results
embeddings = results["embeddings"]  # numpy array (N, embedding_dim)
labels = results["labels"]          # numpy array (N,)
logs = results["logs"]              # dict with train/val losses
```

### With Probes and Viz

```python
from playground import PlaygroundConfig, run_playground_experiment
from playground.playground_probes import run_linear_probe, run_knn_probe
from playground.playground_viz import plot_embeddings_2d, plot_distance_histograms

# Run experiment
config = PlaygroundConfig(dataset="mnist", backbone="mlp", epochs=5)
results = run_playground_experiment(config, gpu=False)

# Evaluate
linear_acc = run_linear_probe(results["embeddings"], results["labels"])
knn_acc = run_knn_probe(results["embeddings"], results["labels"], k=5)

print(f"Linear probe: {linear_acc['test_accuracy']:.2%}")
print(f"kNN probe: {knn_acc['test_accuracy']:.2%}")

# Visualize
plot_embeddings_2d(results["embeddings"], results["labels"], method="pca")
plot_distance_histograms(results["embeddings"], results["labels"])
```

### Using Recipes

```python
from playground.playground_recipes import get_recipe
from playground.playground_runner import run_playground_experiment
from playground.playground_viz import compare_embedding_spaces

# Get recipe
configs = get_recipe("simclr_vs_sne", dataset="cifar10", epochs=10)

# Run all configs
results_list = []
for config in configs:
    results = run_playground_experiment(config, gpu=False)
    results_list.append(results)

# Compare
compare_embedding_spaces(results_list, method="pca")
```

---

## 🧪 Testing

### Quick Self-Check
```bash
python -m playground.self_check
```

### Full Test Suite
```bash
pytest tests/test_playground_smoke.py -v
```

### Fast Tests Only (Skip Training)
```bash
pytest tests/test_playground_smoke.py -v -m "not slow"
```

---

## 🎓 For Teaching/Demos

### Preparation (Before Class)
```bash
# 1. Verify setup
python -m playground.self_check

# 2. Test a quick experiment
python -m playground.playground_cli \
  --dataset mnist \
  --icon_mode simclr_like \
  --epochs 1 \
  --cpu
```

### Live Demo Flow
```bash
# 1. Show single experiment
python -m playground.playground_cli \
  --dataset cifar10 \
  --icon_mode simclr_like \
  --epochs 5 \
  --cpu

# 2. Show recipe comparison
python -m playground.playground_cli \
  --recipe simclr_vs_sne \
  --dataset mnist \
  --epochs 5

# 3. Students explore in notebook
jupyter notebook playground/notebooks/ICon_Playground.ipynb
```

---

## ⚙️ Configuration Options

### I-Con Modes (--icon_mode)
- `simclr_like` - Contrastive (InfoNCE)
- `sne_like` - SNE-style
- `tsne_like` - t-SNE-style
- `supervised` - Label-based
- `cluster_like` - Clustering
- `barlow_twins_like` - Decorrelation
- `vicreg_like` - Variance-invariance
- `debiasing_like` - Bias mitigation

### Datasets (--dataset)
- `cifar10` - CIFAR-10 (10 classes, 32x32 RGB)
- `cifar100` - CIFAR-100 (100 classes, 32x32 RGB)
- `mnist` - MNIST (10 classes, 28x28 grayscale)
- `stl10` - STL-10 (10 classes, 96x96 RGB)

### Backbones (--backbone)
- `resnet18`, `resnet34`, `resnet50` - For RGB images
- `simplecnn` - For MNIST
- `mlp` - For MNIST (fastest)

### Recipes (--recipe)
- `simclr_vs_sne` - 2 configs
- `backbone_comparison` - 2-3 configs
- `supervised_vs_unsupervised` - 2 configs
- `temperature_sweep` - 3 configs
- `kernel_comparison` - 3 configs
- `full_comparison` - 5 configs

---

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 64

# Use smaller backbone
--backbone mlp  # for MNIST
--backbone resnet18  # for RGB images

# Force CPU
--cpu
```

### Slow Training
```bash
# Reduce epochs
--epochs 5

# Use smaller dataset
--dataset mnist

# Use GPU if available
--gpu
```

### Import Errors
```bash
pip install -r requirements.txt
```

### Config Validation Failed
Check error message - it will tell you exactly what's wrong:
```
ValueError: epochs must be positive, got -1
ValueError: Unsupported dataset: invalid_name. Choose from: ['cifar10', 'cifar100', 'mnist', 'stl10']
```

---

## 📊 Expected Performance

### Training Time (approximate, CPU)

| Dataset | Backbone | Epochs | Time |
|---------|----------|--------|------|
| MNIST | MLP | 5 | ~1 min |
| MNIST | SimpleCNN | 10 | ~3 min |
| CIFAR-10 | ResNet18 | 10 | ~30 min |
| CIFAR-10 | ResNet34 | 10 | ~50 min |

### Typical Accuracies (Linear Probe)

| Dataset | Mode | Epochs | Accuracy |
|---------|------|--------|----------|
| MNIST | SimCLR-like | 10 | 85-90% |
| MNIST | Supervised | 10 | 95-98% |
| CIFAR-10 | SimCLR-like | 50 | 60-70% |
| CIFAR-10 | Supervised | 50 | 75-85% |

*Note: These are ballpark figures with minimal tuning*

---

## 📝 Tips

1. **Start small**: Use MNIST with MLP for quick iterations
2. **Use CPU by default**: Add `--cpu` to all commands unless you know GPU is available
3. **Check self-check first**: Run `python -m playground.self_check` before demos
4. **Save outputs**: Results are automatically saved to `playground_runs/`
5. **Compare carefully**: Use recipes for fair comparisons (same hyperparameters)
6. **Visualize early**: Even 1-epoch experiments show interesting patterns

---

## 🆘 Getting Help

1. **Self-check**: `python -m playground.self_check`
2. **README**: See `playground/README_PLAYGROUND.md`
3. **Tests**: Run `pytest tests/test_playground_smoke.py -v`
4. **CLI help**: `python -m playground.playground_cli --help`

---

**Quick Links**:
- Full README: `playground/README_PLAYGROUND.md`
- Test Suite: `tests/test_playground_smoke.py`
- Self-Check: `python -m playground.self_check`
- Main Notebook: `playground/notebooks/ICon_Playground.ipynb`
