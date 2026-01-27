# Using Custom Data with I-Con Playground

A guide for PhD researchers bringing their own datasets to I-Con experiments.

## Overview

The I-Con Playground supports three ways to use custom data:

| Method | Best For | Labels Required? |
|--------|----------|------------------|
| **Image Folders** | Raw images organized by class | Yes (folder names) |
| **Pre-computed Embeddings** | Features from pre-trained models | Optional |
| **Custom PyTorch Dataset** | Complex data pipelines | You decide |

---

## Method 1: Image Folders (Simplest)

Organize your images into folders by class, then point I-Con at the root directory.

### Expected Structure

```
my_dataset/
├── class_a/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── class_b/
│   ├── img001.jpg
│   └── ...
└── class_c/
    └── ...
```

Or with separate train/val splits:

```
my_dataset/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

### Running the Experiment

```bash
python -m playground.playground_cli \
  --custom_dataset "folder:/path/to/my_dataset" \
  --backbone resnet18 \
  --icon_mode simclr_like \
  --epochs 50
```

### Options

- Images are automatically resized to 224x224
- Standard ImageNet normalization is applied
- Contrastive augmentations are applied for contrastive modes

### What if I have no class labels?

Put all images in a single folder called `unlabeled`:

```
my_dataset/
└── unlabeled/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

The system will treat all images as a single class. This works for:
- Unsupervised contrastive learning (simclr_like, sne_like, etc.)
- Clustering experiments (cluster_like)

---

## Method 2: Pre-computed Embeddings

If you already have features from a pre-trained model (ResNet, CLIP, etc.), you can run I-Con objectives directly on those embeddings.

### Creating the Embedding File

Save your embeddings as a `.npz` file:

```python
import numpy as np

# Your pre-computed embeddings
embeddings = ...  # Shape: (N, D)
labels = ...      # Shape: (N,) - class indices, can be all zeros if unlabeled

# Save
np.savez(
    "my_embeddings.npz",
    embeddings=embeddings,
    labels=labels,
)
```

For separate train/val splits:

```python
np.savez(
    "my_embeddings.npz",
    train_embeddings=train_emb,
    train_labels=train_labels,
    val_embeddings=val_emb,
    val_labels=val_labels,
)
```

### Running the Experiment

```bash
python -m playground.playground_cli \
  --custom_dataset "embeddings:/path/to/my_embeddings.npz" \
  --icon_mode tsne_like \
  --epochs 100
```

### Notes

- Embedding dimension is auto-detected
- Contrastive modes work by treating each sample as its own "augmentation"
- This is fast since no forward pass is needed

---

## Method 3: Custom PyTorch Dataset

For complex data pipelines, write your own `torch.utils.data.Dataset` class.

### Creating Your Dataset Class

Create a Python file (e.g., `my_dataset.py`):

```python
import torch
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    """
    Your custom dataset.

    Must return a dict with keys:
    - 'image': tensor of shape (C, H, W) or (D,) for embeddings
    - 'label': integer class label
    - 'index': integer sample index

    For contrastive learning, also return:
    - 'image1': second augmented view of the same sample
    """

    def __init__(self, root_path, transform=None):
        self.data = ...  # Load your data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply transform if provided
        if self.transform:
            image = self.transform(sample['image'])
            image1 = self.transform(sample['image'])  # Different augmentation
        else:
            image = sample['image']
            image1 = sample['image']

        return {
            'image': image,
            'image1': image1,  # For contrastive learning
            'label': sample.get('label', 0),
            'index': idx,
        }
```

### Running the Experiment

```bash
python -m playground.playground_cli \
  --custom_dataset "custom:/path/to/my_dataset.py:MyCustomDataset" \
  --backbone resnet18 \
  --icon_mode simclr_like \
  --epochs 50
```

Format: `custom:<path_to_file>:<ClassName>`

---

## Choosing the Right I-Con Mode

| Mode | Supervisory Signal | Use Case |
|------|-------------------|----------|
| `simclr_like` | Augmentation pairs | Self-supervised learning |
| `sne_like` | Augmentation pairs | Visualization-focused |
| `tsne_like` | Augmentation pairs | Heavy-tailed, better for clusters |
| `supervised` | Class labels | When you have labels |
| `cluster_like` | Augmentation pairs | Discovering clusters |

**For unlabeled data:** Use `simclr_like`, `sne_like`, `tsne_like`, or `cluster_like`.

**For labeled data:** Use `supervised` or any contrastive mode.

---

## Visualizing Your Results

### During Training

```bash
# Generate animated GIF of learning dynamics
python -m playground.playground_cli \
  --custom_dataset "folder:/path/to/my_data" \
  --viz_mode both \
  --gif_every 5 \
  --epochs 100
```

### After Training

Regenerate visualizations without retraining:

```bash
# Change projection method
python -m playground.playground_cli \
  --regen_gif \
  --load_dir playground_runs/my_experiment \
  --gif_method tsne

# Slower playback
python -m playground.playground_cli \
  --regen_gif \
  --load_dir playground_runs/my_experiment \
  --gif_fps 0.5
```

---

## Complete Examples

### Example 1: Medical Images

```bash
# Assuming images organized by diagnosis
python -m playground.playground_cli \
  --custom_dataset "folder:/data/xrays" \
  --backbone resnet50 \
  --icon_mode supervised \
  --epochs 100 \
  --batch_size 64 \
  --viz_mode both
```

### Example 2: CLIP Embeddings

```python
# First, extract CLIP embeddings
import clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path

model, preprocess = clip.load("ViT-B/32")

embeddings = []
labels = []

for class_idx, class_dir in enumerate(Path("my_images").iterdir()):
    for img_path in class_dir.glob("*.jpg"):
        img = preprocess(Image.open(img_path)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_image(img)
        embeddings.append(emb.cpu().numpy())
        labels.append(class_idx)

np.savez("clip_embeddings.npz",
         embeddings=np.vstack(embeddings),
         labels=np.array(labels))
```

```bash
# Then run I-Con on the embeddings
python -m playground.playground_cli \
  --custom_dataset "embeddings:clip_embeddings.npz" \
  --icon_mode tsne_like \
  --epochs 200 \
  --gif_every 10
```

### Example 3: Unlabeled Images for Clustering

```bash
# Put all images in one "unlabeled" folder
python -m playground.playground_cli \
  --custom_dataset "folder:/data/unlabeled_images" \
  --icon_mode cluster_like \
  --epochs 50 \
  --viz_mode both
```

---

## Troubleshooting

### "No images found"

Check your folder structure matches the expected format. Class folders should contain images directly.

### "KeyError: 'image'"

Your custom dataset must return a dict with `'image'`, `'label'`, and `'index'` keys.

### "Embeddings appear unchanged"

This warning means your embeddings aren't changing during training. Possible causes:
- Using pre-computed embeddings with the wrong backbone
- Learning rate too low
- Model frozen

### Memory Issues

- Reduce `--batch_size`
- Reduce `--gif_max_points`
- Use `--num_workers 0` on macOS

---

## Output Structure

After running with custom data:

```
playground_runs/custom_dataset_resnet18_simclr_like_TIMESTAMP/
├── config.json                  # Full configuration
├── experiment_manifest.json     # Reproducibility info
├── embeddings.npz               # Final learned embeddings
├── logs.json                    # Training metrics
├── final_model.pt               # Model weights
├── training_dynamics.gif        # Animated visualization
├── training_curves.png          # Loss plots
└── embeddings_pca.png           # Static embedding plot
```

The learned embeddings in `embeddings.npz` can be used for:
- Downstream classification
- Clustering analysis
- Nearest-neighbor retrieval
- Transfer to other tasks
