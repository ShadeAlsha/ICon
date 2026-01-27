"""
Custom Dataset Support for I-Con Playground

Provides support for:
1. Folder-based image datasets (ImageFolder-style)
2. Pre-computed embeddings (.npz/.pt files)
3. Custom PyTorch Dataset injection

This enables PhD-style experiments with custom/novel data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import importlib.util
import os


class FolderImageDataset(Dataset):
    """
    Dataset for images organized in folders by class.

    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg

    Similar to ImageFolder but with additional features for I-Con.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        contrastive: bool = True,
        num_views: int = 2,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.gif'),
    ):
        """
        Args:
            root: Root directory containing class folders
            transform: Transform to apply to images
            contrastive: Whether to return multiple views for contrastive learning
            num_views: Number of views to return (default 2)
            extensions: Valid image file extensions
        """
        self.root = Path(root)
        self.transform = transform or self._default_transform()
        self.contrastive = contrastive
        self.num_views = num_views
        self.extensions = extensions

        # Discover classes and images
        self.classes = []
        self.class_to_idx = {}
        self.samples = []  # List of (image_path, class_idx)

        self._scan_directory()

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {root}.\n"
                f"Expected folder structure:\n"
                f"  {root}/\n"
                f"    class1/\n"
                f"      image1.jpg\n"
                f"      image2.jpg\n"
                f"    class2/\n"
                f"      image1.jpg\n"
                f"      ..."
            )

        print(f"FolderImageDataset: Found {len(self.samples)} images in {len(self.classes)} classes")

    def _scan_directory(self):
        """Scan directory for class folders and images."""
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, idx))

    def _default_transform(self):
        """Default transform for images."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        data = {
            'image': self.transform(img),
            'label': label,
            'index': idx,
        }

        if self.contrastive:
            for i in range(self.num_views - 1):
                data[f'image{i+1}'] = self.transform(img)

        return data


class EmbeddingDataset(Dataset):
    """
    Dataset for pre-computed embeddings.

    Supports loading from:
    - .npz files (numpy arrays)
    - .pt files (PyTorch tensors)

    Expected format:
        - embeddings: (N, D) array of embeddings
        - labels: (N,) array of class labels
        - indices (optional): (N,) array of sample indices

    This is useful for:
    - Transfer learning experiments
    - Using embeddings from pre-trained models
    - Fast iteration without recomputing features
    """

    def __init__(
        self,
        path: str,
        split: str = 'train',
        contrastive: bool = False,
    ):
        """
        Args:
            path: Path to .npz or .pt file containing embeddings
            split: 'train' or 'val' (for files with multiple splits)
            contrastive: Whether this is for contrastive learning
                        (Note: pre-computed embeddings usually don't support
                        contrastive mode since augmentations are already applied)
        """
        self.path = Path(path)
        self.split = split
        self.contrastive = contrastive

        if not self.path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        # Load embeddings
        if self.path.suffix == '.npz':
            self._load_npz()
        elif self.path.suffix == '.pt':
            self._load_pt()
        else:
            raise ValueError(f"Unsupported file format: {self.path.suffix}. Use .npz or .pt")

        print(f"EmbeddingDataset: Loaded {len(self.embeddings)} embeddings "
              f"with dimension {self.embeddings.shape[1]}")

    def _load_npz(self):
        """Load from numpy .npz file."""
        data = np.load(self.path, allow_pickle=True)

        # Try different key formats
        emb_key = f'{self.split}_embeddings' if f'{self.split}_embeddings' in data else 'embeddings'
        label_key = f'{self.split}_labels' if f'{self.split}_labels' in data else 'labels'

        self.embeddings = torch.from_numpy(data[emb_key]).float()
        self.labels = torch.from_numpy(data[label_key]).long()

        if 'indices' in data:
            self.indices = torch.from_numpy(data['indices']).long()
        else:
            self.indices = torch.arange(len(self.embeddings))

    def _load_pt(self):
        """Load from PyTorch .pt file."""
        data = torch.load(self.path, weights_only=True)

        if isinstance(data, dict):
            emb_key = f'{self.split}_embeddings' if f'{self.split}_embeddings' in data else 'embeddings'
            label_key = f'{self.split}_labels' if f'{self.split}_labels' in data else 'labels'

            self.embeddings = data[emb_key].float()
            self.labels = data[label_key].long()
            self.indices = data.get('indices', torch.arange(len(self.embeddings))).long()
        else:
            # Assume tuple format (embeddings, labels)
            self.embeddings = data[0].float()
            self.labels = data[1].long()
            self.indices = torch.arange(len(self.embeddings))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'label': self.labels[idx].item(),
            'index': self.indices[idx].item(),
        }


def load_custom_dataset(
    path: str,
    module_name: str = None,
    class_name: str = None,
) -> Dataset:
    """
    Load a custom PyTorch Dataset from a Python file.

    Usage:
        dataset = load_custom_dataset(
            path="my_dataset.py",
            class_name="MyCustomDataset"
        )

    The custom dataset class should:
    1. Inherit from torch.utils.data.Dataset
    2. Return dict with keys: 'image', 'label', 'index'
    3. Optionally support 'image1' for contrastive learning

    Args:
        path: Path to Python file containing the dataset class
        module_name: Optional module name (defaults to filename)
        class_name: Name of the dataset class to load

    Returns:
        Instance of the custom dataset
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {path}")

    if class_name is None:
        raise ValueError("class_name is required to load custom dataset")

    # Load the module
    module_name = module_name or path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the dataset class
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in {path}.\n"
            f"Available classes: {[name for name in dir(module) if not name.startswith('_')]}"
        )

    dataset_class = getattr(module, class_name)

    # Verify it's a Dataset subclass
    if not issubclass(dataset_class, Dataset):
        raise TypeError(
            f"{class_name} must be a subclass of torch.utils.data.Dataset"
        )

    return dataset_class


def get_custom_dataloaders(
    dataset_type: str,
    path: str,
    batch_size: int = 256,
    num_workers: int = 4,
    contrastive: bool = True,
    num_views: int = 2,
    train_split: float = 0.8,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders from custom datasets.

    Args:
        dataset_type: One of 'folder', 'embeddings', 'custom'
        path: Path to dataset (folder, .npz/.pt file, or .py file)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        contrastive: Whether to use contrastive mode
        num_views: Number of views for contrastive learning
        train_split: Fraction for training (if dataset has no predefined split)
        **kwargs: Additional arguments passed to dataset class

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract class_name from kwargs - only used for 'custom' type
    class_name = kwargs.pop('class_name', None)

    if dataset_type == 'folder':
        # Folder-based image dataset
        # Classes are discovered from folder names, not from class_name parameter
        train_path = Path(path) / 'train'
        val_path = Path(path) / 'val'

        if not train_path.exists():
            # Single folder - do random split
            full_dataset = FolderImageDataset(
                path, contrastive=contrastive, num_views=num_views, **kwargs
            )
            train_size = int(train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        else:
            # Separate train/val folders
            train_dataset = FolderImageDataset(
                train_path, contrastive=contrastive, num_views=num_views, **kwargs
            )
            val_dataset = FolderImageDataset(
                val_path, contrastive=False, num_views=1, **kwargs
            )

    elif dataset_type == 'embeddings':
        # Pre-computed embeddings
        train_dataset = EmbeddingDataset(path, split='train', contrastive=contrastive)
        val_dataset = EmbeddingDataset(path, split='val', contrastive=False)

    elif dataset_type == 'custom':
        # Custom PyTorch Dataset - requires class_name
        if class_name is None:
            raise ValueError("class_name is required for custom datasets")

        dataset_class = load_custom_dataset(path, class_name=class_name)

        # Instantiate dataset
        full_dataset = dataset_class(**kwargs)

        # Split into train/val
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'folder', 'embeddings', or 'custom'")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // num_views if contrastive else batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
