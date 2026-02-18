# I-Con Reproducibility Guide

This guide provides step-by-step instructions to reproduce I-Con experiments and verify functionality.

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel/AMD/Apple Silicon)
- **RAM**: 8 GB
- **Storage**: 5 GB free space (for datasets and outputs)
- **GPU**: Not required (CPU mode available)

### Recommended for Full Reproduction
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **GPU**:
  - NVIDIA GPU with 6+ GB VRAM (CUDA 11.8+)
  - OR Apple Silicon M1/M2/M3/M4 (MPS support)
- **Storage**: 10 GB free space

### Software Requirements
- Python 3.8+
- PyTorch 2.0.0+
- PyTorch Lightning 2.0.0+
- See `requirements.txt` for complete dependencies

## Quick Validation (5 minutes)

Verify installation and basic functionality:

```bash
# 1. Clone and install
git clone https://github.com/ShadeAlsha/ICon.git
cd ICon
pip install -r requirements.txt

# 2. Run quick test (30 seconds)
python -m playground.playground_cli \
  --dataset mnist \
  --method simclr \
  --epochs 1 \
  --batch-size 128 \
  --device cpu \
  --no-viz

# 3. Check output directory created
ls playground_runs/
```

**Expected Output**:
- Training completes without errors
- Final loss displayed (typically 4.0-6.0 for 1 epoch)
- Directory created in `playground_runs/` with timestamp

## Running Tests

### Unit Tests (3 seconds)

Test core components:

```bash
pytest playground/tests/test_divergences.py -v
pytest playground/tests/test_config_integration.py -v
pytest playground/tests/test_device_handling.py -v
```

**Expected Output**:
- 44 tests pass
- 2 tests skipped (GPU-specific on CPU-only machines)
- Runtime: 2-3 seconds total

**What's Tested**:
- KL, Reverse KL, and Jensen-Shannon divergences
- Configuration validation and defaults
- Device selection logic (CPU/CUDA/MPS)
- Optimizer initialization
- Weight decay handling

### Integration Test (20 seconds)

Test end-to-end training pipeline:

```bash
pytest playground/tests/test_integration_pure_pytorch.py::TestIntegrationPurePyTorch::test_minimal_training_cpu -v
```

**Expected Output**:
- 1 test passes
- Runtime: 15-20 seconds
- Loss decreases over 5 training steps

**What's Tested**:
- Complete training loop
- Batch processing
- Loss computation
- Gradient updates
- Model state persistence

### Full Test Suite (30 seconds)

Run all tests:

```bash
pytest playground/tests/ -v
```

**Expected Output**:
- 45+ tests pass
- 2 skipped (GPU tests on CPU machines)
- Runtime: 25-30 seconds

## Reproducing Core Experiments

### Experiment 1: SimCLR on MNIST

**Purpose**: Verify contrastive learning implementation

**Command**:
```bash
python -m playground.playground_cli \
  --dataset mnist \
  --method simclr \
  --epochs 20 \
  --batch-size 256 \
  --embedding-dim 128 \
  --lr 1e-3 \
  --device auto
```

**Runtime**:
- CPU: ~15 minutes
- GPU (RTX 3090): ~2 minutes
- Apple Silicon (M1): ~5 minutes

**Expected Output**:
- Final loss: 2.0-3.0
- Linear probe accuracy: 85-92%
- Embedding visualization saved
- Files created:
  - `config.json` (configuration)
  - `embeddings.npz` (final embeddings)
  - `logs.json` (training history)
  - `final_model.pt` (model checkpoint)

**Output Location**: `playground_runs/mnist_simclr_YYYYMMDD_HHMMSS/`

### Experiment 2: Parametric t-SNE on CIFAR-10

**Purpose**: Verify dimensionality reduction implementation

**Command**:
```bash
python -m playground.playground_cli \
  --dataset cifar10 \
  --method tsne \
  --epochs 30 \
  --batch-size 256 \
  --embedding-dim 2 \
  --lr 1e-3 \
  --device auto
```

**Runtime**:
- CPU: ~45 minutes
- GPU (RTX 3090): ~5 minutes
- Apple Silicon (M1): ~15 minutes

**Expected Output**:
- Final loss: 1.5-2.5
- Clear cluster separation in embedding plot
- 10 distinct clusters visible (one per class)
- Linear probe accuracy: 40-55% (expected for 2D embeddings)

**Output Location**: `playground_runs/cifar10_tsne_YYYYMMDD_HHMMSS/`

### Experiment 3: Supervised Contrastive Learning

**Purpose**: Verify label-based learning

**Command**:
```bash
python -m playground.playground_cli \
  --dataset mnist \
  --method supervised \
  --epochs 20 \
  --batch-size 256 \
  --embedding-dim 128 \
  --lr 1e-3 \
  --device auto
```

**Runtime**:
- CPU: ~15 minutes
- GPU (RTX 3090): ~2 minutes
- Apple Silicon (M1): ~5 minutes

**Expected Output**:
- Final loss: 1.5-2.5
- Linear probe accuracy: 92-96%
- Better separation than SimCLR
- Class-based clustering visible in embeddings

### Experiment 4: Method Comparison Recipe

**Purpose**: Compare SimCLR vs t-SNE side-by-side

**Command**:
```bash
python -m playground.playground_cli \
  --recipe simclr_vs_sne \
  --dataset mnist \
  --epochs 10 \
  --batch-size 128 \
  --device cpu
```

**Runtime**:
- CPU: ~10 minutes
- GPU: ~2 minutes

**Expected Output**:
- Two experiments complete
- Comparison plot generated: `recipe_simclr_vs_sne_comparison.png`
- Summary JSON: `recipe_simclr_vs_sne_summary.json`
- Shows embedding differences between methods

## Testing Different Configurations

### Divergence Functions

Test all divergence implementations:

```bash
# KL Divergence (default)
python -m playground.playground_cli --dataset mnist --method simclr --divergence kl --epochs 5 --device cpu --no-viz

# Reverse KL
python -m playground.playground_cli --dataset mnist --method simclr --divergence rkl --epochs 5 --device cpu --no-viz

# Jensen-Shannon
python -m playground.playground_cli --dataset mnist --method simclr --divergence js --epochs 5 --device cpu --no-viz
```

**Runtime per Test**: 2-3 minutes (CPU)

**Expected Outputs**:
- All complete successfully
- Different final loss values:
  - KL: 3.5-5.0
  - RKL: 3.0-4.5
  - JS: 3.8-5.2
- Similar probe accuracy (±3%)

### Optimizers

Test optimizer implementations:

```bash
# AdamW (default, recommended)
python -m playground.playground_cli --dataset mnist --method simclr --optimizer adamw --epochs 5 --device cpu --no-viz

# Adam
python -m playground.playground_cli --dataset mnist --method simclr --optimizer adam --epochs 5 --device cpu --no-viz

# SGD
python -m playground.playground_cli --dataset mnist --method simclr --optimizer sgd --lr 0.1 --epochs 5 --device cpu --no-viz
```

**Runtime per Test**: 2-3 minutes (CPU)

**Expected Outputs**:
- All converge successfully
- AdamW typically achieves best final loss
- SGD requires higher learning rate (0.1 vs 1e-3)

### Device Selection

Test hardware acceleration:

```bash
# CPU (always works)
python -m playground.playground_cli --dataset mnist --method simclr --epochs 1 --device cpu

# Auto (uses best available: CUDA > MPS > CPU)
python -m playground.playground_cli --dataset mnist --method simclr --epochs 1 --device auto

# GPU (requires CUDA or MPS)
python -m playground.playground_cli --dataset mnist --method simclr --epochs 1 --device cuda

# Apple Silicon
python -m playground.playground_cli --dataset mnist --method simclr --epochs 1 --device mps
```

**Runtime**: 30-60 seconds each

**Expected Outputs**:
- Device clearly logged at start: `Using device: [cpu/cuda/mps]`
- GPU devices show model name
- Error if unavailable device requested

## Expected File Structure

After running experiments, verify output structure:

```
playground_runs/
├── mnist_simclr_20260218_153045/
│   ├── config.json              # Full configuration
│   ├── embeddings.npz           # Final embeddings (shape: [N, embedding_dim])
│   ├── logs.json               # Training metrics per epoch
│   ├── final_model.pt          # Model checkpoint
│   ├── embedding_plot.png      # t-SNE visualization (if --viz)
│   └── probe_results.json      # Linear probe accuracy (if --probe)
│
└── recipe_simclr_vs_sne_comparison.png  # Recipe outputs
```

### File Sizes (Approximate)

| File | MNIST | CIFAR-10 |
|------|-------|----------|
| config.json | 2 KB | 2 KB |
| embeddings.npz | 5 MB | 50 MB |
| logs.json | 10 KB | 20 KB |
| final_model.pt | 2-10 MB | 5-50 MB |
| embedding_plot.png | 500 KB | 1 MB |

## Verification Checklist

Use this checklist to verify full reproducibility:

**Installation**:
- [ ] Repository cloned successfully
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] PyTorch CUDA/MPS available (if using GPU)

**Unit Tests**:
- [ ] Divergence tests pass (3 divergences × multiple tests)
- [ ] Config tests pass (validation, defaults, integration)
- [ ] Device tests pass (auto-selection, error handling)
- [ ] Total: 44 passed, 2 skipped (if CPU-only)

**Integration Tests**:
- [ ] End-to-end training completes
- [ ] Loss decreases over epochs
- [ ] Model checkpoints save correctly

**CLI Functionality**:
- [ ] Basic training works (`--dataset mnist --method simclr`)
- [ ] All divergences work (`--divergence kl/rkl/js`)
- [ ] All optimizers work (`--optimizer adam/adamw/sgd`)
- [ ] Device selection works (`--device cpu/cuda/mps/auto`)
- [ ] Visualization works (`--viz` flag)
- [ ] Linear probe works (`--probe` flag)

**Core Experiments**:
- [ ] SimCLR on MNIST (accuracy 85-92%)
- [ ] t-SNE on CIFAR-10 (clear clusters)
- [ ] Supervised contrastive (accuracy 92-96%)
- [ ] Recipe comparison (two runs complete)

**Output Files**:
- [ ] `config.json` created and readable
- [ ] `embeddings.npz` created with correct shape
- [ ] `logs.json` contains per-epoch metrics
- [ ] `final_model.pt` can be loaded

## Troubleshooting

### Tests Fail to Import

**Issue**: `ModuleNotFoundError: No module named 'pytest'`

**Solution**:
```bash
pip install pytest
```

### CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python -m playground.playground_cli --batch-size 64 --device cuda

# Or use CPU
python -m playground.playground_cli --device cpu
```

### Slow Training on Mac

**Issue**: Training is slow despite having Apple Silicon

**Solution**:
```bash
# Ensure MPS is being used
python -m playground.playground_cli --device mps

# Verify in output: "Using device: mps"
```

### Different Results Than Expected

**Issue**: Accuracy or loss values differ from guide

**Causes**:
- Random seed variation (expected ±5% accuracy)
- Different PyTorch version
- Different hardware (GPU vs CPU training can differ slightly)

**Acceptable Variation**:
- Accuracy: ±5%
- Loss: ±20%
- Training time: ±50%

### Recipe Comparison Plot Not Generated

**Issue**: Recipe runs but no comparison plot

**Solution**:
```bash
# Ensure matplotlib is installed
pip install matplotlib

# Check output directory
ls playground_runs/recipe_*.png
```

## Performance Benchmarks

Expected runtimes for full experiments:

### MNIST (60,000 samples, 20 epochs)

| Hardware | Batch Size | Time per Epoch | Total Time |
|----------|-----------|----------------|------------|
| CPU (8 cores) | 256 | ~45 sec | ~15 min |
| RTX 3090 | 512 | ~5 sec | ~2 min |
| A100 | 1024 | ~3 sec | ~1 min |
| M1 Max | 256 | ~15 sec | ~5 min |

### CIFAR-10 (50,000 samples, 30 epochs)

| Hardware | Batch Size | Time per Epoch | Total Time |
|----------|-----------|----------------|------------|
| CPU (8 cores) | 256 | ~90 sec | ~45 min |
| RTX 3090 | 512 | ~8 sec | ~4 min |
| A100 | 1024 | ~5 sec | ~2.5 min |
| M1 Max | 256 | ~30 sec | ~15 min |

## Success Criteria

Your reproduction is successful if:

✅ All unit tests pass (44/44, 2 skipped on CPU)
✅ Integration test completes without errors
✅ SimCLR achieves 85%+ probe accuracy on MNIST
✅ t-SNE shows clear visual clusters
✅ All divergences and optimizers work
✅ Output files are created with expected structure
✅ Training completes in expected timeframe (±50%)
✅ Results match expected ranges (±5% accuracy, ±20% loss)

## Citation

If you use I-Con in your research, please cite:

```bibtex
@inproceedings{alshammariunifying,
  title={I-Con: A Unifying Framework for Representation Learning},
  author={Alshammari, Shaden Naif and Hershey, John R and Feldmann, Axel and Freeman, William T and Hamilton, Mark},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
