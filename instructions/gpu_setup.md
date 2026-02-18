# GPU Setup Guide

This guide covers GPU setup for I-Con training and experimentation.

## CUDA Version Requirements

### Minimum Requirements
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.7 or higher (typically bundled with PyTorch)
- **NVIDIA Driver**:
  - Linux: 450.80.02 or higher
  - Windows: 452.39 or higher

### Recommended Versions
- **CUDA**: 12.1 or higher
- **PyTorch**: 2.0.0 or higher (included in requirements.txt)

### Check Your CUDA Version

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Check if CUDA is available in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## PyTorch Installation

### CUDA 11.8

```bash
# Using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Using conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### CUDA 12.1

```bash
# Using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Using conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### CUDA 12.4+ (Latest)

```bash
# Using pip
pip install torch torchvision torchaudio

# Using conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
GPU: NVIDIA GeForce RTX 3090 (or your GPU model)
```

## VRAM Recommendations

### Minimum Requirements by Dataset

| Dataset | Batch Size | Embedding Dim | Min VRAM | Recommended VRAM |
|---------|-----------|---------------|----------|------------------|
| MNIST | 256 | 128 | 2 GB | 4 GB |
| CIFAR-10 | 256 | 128 | 4 GB | 6 GB |
| CIFAR-10 | 512 | 256 | 6 GB | 8 GB |
| Custom Data (small images) | 256 | 128 | 4 GB | 8 GB |
| Custom Data (large images) | 128 | 512 | 8 GB | 12 GB |

### Memory Optimization Tips

If you encounter CUDA out-of-memory errors:

1. **Reduce batch size**:
   ```bash
   python -m playground.playground_cli --batch-size 128
   ```

2. **Reduce embedding dimension**:
   ```bash
   python -m playground.playground_cli --embedding-dim 64
   ```

3. **Use gradient accumulation** (in code):
   ```python
   trainer = pl.Trainer(accumulate_grad_batches=2)  # Effective batch size = batch_size * 2
   ```

4. **Use mixed precision training**:
   ```python
   trainer = pl.Trainer(precision=16)  # Or precision="16-mixed"
   ```

5. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi  # Real-time monitoring
   ```

## Multi-GPU Setup

I-Con supports multi-GPU training through PyTorch Lightning's distributed training strategies.

### Single Machine, Multiple GPUs

```bash
# Automatic: Use all available GPUs
python -m playground.playground_cli --devices -1

# Specify number of GPUs
python -m playground.playground_cli --devices 2

# Specify GPU IDs
CUDA_VISIBLE_DEVICES=0,1 python -m playground.playground_cli --devices 2
```

### Distributed Training Strategies

For large-scale training:

```python
from pytorch_lightning import Trainer

# Data Parallel (DDP) - Recommended
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="ddp"
)

# DDP with find_unused_parameters (if needed)
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true"
)
```

### Multi-Node Training

For cluster environments:

```bash
# Node 0 (master)
python -m playground.playground_cli \
    --devices 4 \
    --num-nodes 2 \
    --strategy ddp

# Node 1 (worker)
python -m playground.playground_cli \
    --devices 4 \
    --num-nodes 2 \
    --strategy ddp
```

### Verify Multi-GPU Usage

```bash
# Terminal 1: Start training
python -m playground.playground_cli --devices 2

# Terminal 2: Monitor GPU usage
watch -n 1 nvidia-smi
```

You should see multiple python processes using different GPUs.

## Apple Silicon (MPS) Setup

For M1/M2/M3/M4 Macs:

### Requirements
- macOS 12.3 or higher
- PyTorch 2.0.0 or higher

### Installation

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Or with conda
conda install pytorch torchvision torchaudio -c pytorch
```

### Usage

```bash
# Automatic device selection (will choose MPS on Apple Silicon)
python -m playground.playground_cli --device auto

# Force MPS
python -m playground.playground_cli --device mps
```

### Verify MPS

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Performance Notes
- MPS provides 2-5x speedup over CPU on Apple Silicon
- Performance may vary by model size and operation types
- Not all PyTorch operations are MPS-optimized yet

## Troubleshooting

### CUDA Out of Memory

```bash
# Error: RuntimeError: CUDA out of memory

# Solutions:
1. Reduce batch size: --batch-size 64
2. Reduce embedding dimension: --embedding-dim 64
3. Clear GPU cache before training:
   python -c "import torch; torch.cuda.empty_cache()"
```

### CUDA Not Available

```bash
# Error: torch.cuda.is_available() returns False

# Check:
1. nvidia-smi works
2. PyTorch CUDA version matches system CUDA:
   python -c "import torch; print(torch.version.cuda)"
3. Reinstall PyTorch with correct CUDA version (see above)
```

### cuDNN Errors

```bash
# Error: cuDNN version mismatch

# Solution: Reinstall PyTorch (includes bundled cuDNN)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Multi-GPU Hanging

```bash
# Training hangs with multiple GPUs

# Solutions:
1. Set NCCL environment variables:
   export NCCL_DEBUG=INFO
   export NCCL_P2P_DISABLE=1

2. Use different strategy:
   --strategy ddp_spawn  # Instead of ddp
```

## Example Installation Commands

### Complete Setup (CUDA 12.1)

```bash
# 1. Create environment
conda create -n icon python=3.10
conda activate icon

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install I-Con dependencies
cd ICon
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# 5. Test training
python -m playground.playground_cli --dataset mnist --method simclr --epochs 5
```

### Google Colab Setup

```python
# Colab comes with PyTorch and CUDA pre-installed
!git clone https://github.com/ShadeAlsha/ICon.git
%cd ICon
!pip install -r requirements.txt

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Run training
!python -m playground.playground_cli --dataset cifar10 --method simclr --epochs 20
```

### Cluster Setup (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=icon_training
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G

module load cuda/12.1
module load python/3.10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate icon

cd $SLURM_SUBMIT_DIR
python -m playground.playground_cli \
    --dataset cifar10 \
    --method simclr \
    --epochs 100 \
    --devices 4 \
    --batch-size 512
```

## Performance Benchmarks

Typical training times on different hardware (CIFAR-10, 100 epochs):

| Hardware | Batch Size | Time per Epoch | Total Time |
|----------|-----------|----------------|------------|
| CPU (16 cores) | 128 | ~15 min | ~25 hours |
| RTX 3090 (24GB) | 512 | ~45 sec | ~1.25 hours |
| A100 (40GB) | 1024 | ~30 sec | ~50 min |
| 4x A100 (DDP) | 4096 | ~15 sec | ~25 min |
| M1 Max (MPS) | 256 | ~2 min | ~3.5 hours |

## Additional Resources

- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Lightning Multi-GPU Training](https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
