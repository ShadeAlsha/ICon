import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from os.path import join
import random
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image


class FlattenTransform:
    """Picklable transform that optionally flattens tensors."""
    def __init__(self, flatten=False):
        self.flatten = flatten

    def __call__(self, x):
        return x.view(-1) if self.flatten else x


class ContrastiveDatasetFromImages(Dataset):
    def __init__(self, dataset, num_views=2, transform=None, contrastive=True, distinct_views=True):
        self.dataset = dataset  
        self.num_views = num_views
        self.transform = transform  
        self.distinct_views = distinct_views
        self.contrastive = contrastive

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Ensure the image is in PIL format if it's a tensor
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
            
        data = {"image":self.transform(img), "label": label, "index": idx}
        if self.contrastive:
            for i in range(self.num_views-1):
                data[f"image{i+1}"]= self.transform(img)
        return data



def get_dataloaders(
    batch_size=256,
    num_views=2,
    dataset_name='cifar10',
    num_workers=4,
    size=224,
    root='./data',
    with_augmentation=True,
    contrastive = True,
    unlabeled = True,
    shuffle_train=True,
    shuffle_test=True,
    non_parametric = False,
    return_datasets=False,
    max_train_samples=None,
    ):
    dataset_name = dataset_name.lower()

    # Ensure dataset root directory exists
    import os
    os.makedirs(root, exist_ok=True)

    # Define normalization parameters
    normalization_params = {
        "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "cifar100": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "tinyimagenet": {"mean": (0.4802, 0.4481, 0.3975), "std": (0.2302, 0.2265, 0.2262)},
        "imagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        "stl10": {"mean":(0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        "mnist": {"mean": (0.1307,), "std": (0.3081,)},
        "oxfordpets": {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713)},
    }

    if dataset_name not in normalization_params:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'cifar10', 'cifar100', 'tinyimagenet', 'stl10', or 'mnist'.")

    # Select normalization
    mean, std = normalization_params[dataset_name]["mean"], normalization_params[dataset_name]["std"]
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define transformation pipelines
    flatten_transform = FlattenTransform(flatten=(dataset_name == 'mnist'))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip() if dataset_name != 'mnist' else transforms.RandomAffine(30, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) if dataset_name != 'mnist' else None,
        transforms.RandomGrayscale(p=0.2) if dataset_name != 'mnist' else None,
        transforms.ToTensor(),
        normalize,
        flatten_transform,
    ]) if with_augmentation else transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
        flatten_transform,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
        flatten_transform,
    ])

    # Remove None transforms (applicable for MNIST)
    train_transform.transforms = [t for t in train_transform.transforms if t is not None]

    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True)
    elif dataset_name.lower() == "cifar100":
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
    elif dataset_name.lower() == "tinyimagenet":
        data_dir = f"{root}/tiny-imagenet-200"
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train")
        test_dataset = datasets.ImageFolder(root=f"{data_dir}/val")
    elif dataset_name.lower() == "stl10":
        train_dataset = datasets.STL10(root=root, split='train+unlabeled' if unlabeled else 'train', download=True)
        test_dataset = datasets.STL10(root=root, split='test', download=True)
    elif dataset_name.lower() == "mnist":
        train_dataset = datasets.MNIST(root=root, train=True, download=True)
        test_dataset = datasets.MNIST(root=root, train=False, download=True)
    elif dataset_name.lower() == "oxfordpets":
        train_dataset = datasets.OxfordIIITPet(root=root, split='trainval', target_types='category', download=True, transform=None)
        test_dataset = datasets.OxfordIIITPet(root=root, split='test', target_types='category', download=True, transform=None)
    elif dataset_name.lower() == "imagenet":
        data_dir = join(root, "imagenet2/ILSVRC/Data/CLS-LOC")
        train_dir = join(data_dir, "train")
        test_dir = join(data_dir, "val2")
        train_dataset = datasets.ImageFolder(root=train_dir)
        test_dataset = datasets.ImageFolder(root=test_dir)
    
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        indices = random.sample(range(len(train_dataset)), max_train_samples)
        train_dataset = Subset(train_dataset, indices)
        
    train_dataset_wrapped = ContrastiveDatasetFromImages(train_dataset, num_views=num_views, transform=train_transform, contrastive=contrastive)
    test_dataset_wrapped = ContrastiveDatasetFromImages(test_dataset, num_views=num_views, transform=test_transform, contrastive=False)
            
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=batch_size // num_views,
        shuffle=shuffle_train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=batch_size // num_views,
        shuffle=shuffle_test,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader