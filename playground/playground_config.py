"""
Playground Configuration Module

Provides a simple, student-friendly configuration abstraction for I-Con experiments.
The PlaygroundConfig translates high-level settings into the internal I-Con Config.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_config import Config
from mappers.base import ResNet, SimpleCNN, MLPMapper
from distributions.distance import Gaussian, StudentT
from distributions.graph import Augmentation, Label
from playground.divergences import SUPPORTED_DIVERGENCES

# Note: UniformCluster is available for future use but not currently used in presets
# from distributions.clustering import UniformCluster


# ---------------------------------------------------------------------------
# Supported options (for reference and validation)
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["cifar10", "cifar100", "mnist", "stl10"]
SUPPORTED_BACKBONES = ["resnet18", "resnet34", "resnet50", "simplecnn", "mlp"]
SUPPORTED_ICON_MODES = [
    "simclr_like",
    "sne_like",
    "tsne_like",
    "supervised",
    "cluster_like",
    "barlow_twins_like",
    "vicreg_like",
    "debiasing_like",
]


# ---------------------------------------------------------------------------
# PlaygroundConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlaygroundConfig:
    """
    A simplified configuration for I-Con playground experiments.

    This config provides a student-friendly interface to set up I-Con experiments
    without needing to understand the full Config system. It translates these
    high-level settings into the internal I-Con configuration.

    Attributes:
        dataset: Which dataset to use. Options: "cifar10", "cifar100", "mnist", "stl10"
        backbone: Which encoder architecture. Options: "resnet18", "resnet34",
                  "resnet50", "simplecnn" (for MNIST), "mlp" (for MNIST)
        icon_mode: Which I-Con preset to use:
            - "simclr_like": Contrastive learning with augmentation pairs (InfoNCE-style)
            - "sne_like": SNE-style neighborhood preservation
            - "tsne_like": t-SNE-style variant with heavy-tailed kernel
            - "supervised": Supervised learning with label-based supervisory distribution
            - "cluster_like": Clustering-oriented configuration
            - "barlow_twins_like": Decorrelation/redundancy-reduction style
            - "vicreg_like": Variance-invariance-covariance regularization style
            - "debiasing_like": Configuration emphasizing debiasing behavior
        epochs: Number of training epochs (default: 10)
        batch_size: Training batch size (default: 256)
        learning_rate: Optimizer learning rate (default: 1e-3)
        temperature: Temperature/sigma for the learned distribution (default: 0.5)
        embedding_dim: Dimension of the learned embedding space (default: 128)
        output_dir: Directory to save logs, checkpoints, and plots (default: "playground_runs")
        run_name: Optional custom name for this run (default: auto-generated from config)
        num_workers: Number of dataloader workers (default: 4)
        image_size: Input image size (default: 32 for CIFAR, 28 for MNIST)
        use_augmentation: Whether to use data augmentation (default: True)
        seed: Random seed for reproducibility (default: 42)
        save_checkpoints: Whether to save model checkpoints during training (default: False)

    Example:
        >>> config = PlaygroundConfig(
        ...     dataset="cifar10",
        ...     backbone="resnet18",
        ...     icon_mode="simclr_like",
        ...     epochs=10,
        ...     batch_size=256
        ... )
        >>> icon_config = config.to_icon_config()
    """

    dataset: str = "cifar10"
    backbone: str = "resnet18"
    icon_mode: str = "simclr_like"

    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    temperature: float = 0.5
    embedding_dim: int = 128

    output_dir: str = "playground_runs"
    run_name: Optional[str] = None
    num_workers: int = 4
    image_size: Optional[int] = None
    use_augmentation: bool = True
    seed: int = 42
    save_checkpoints: bool = False

    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    divergence: str = "kl"
    use_linear_probe: bool = True

    # Custom dataset options
    custom_dataset_type: Optional[str] = None  # 'folder', 'embeddings', 'custom'
    custom_dataset_path: Optional[str] = None  # Path to custom data
    custom_dataset_class: Optional[str] = None  # Class name for 'custom' type

    def __post_init__(self):
        """Normalize and set defaults. Validation deferred to to_icon_config()."""
        self.dataset = self.dataset.lower()
        self.backbone = self.backbone.lower()
        self.icon_mode = self.icon_mode.lower()
        self.divergence = self.divergence.lower()
        self.optimizer = self.optimizer.lower()

        if self.image_size is None:
            self.image_size = {"mnist": 28, "cifar10": 32, "cifar100": 32, "stl10": 96}.get(self.dataset, 32)

        if self.run_name is None:
            self.run_name = f"{self.dataset}_{self.backbone}_{self.icon_mode}"

    def validate(self) -> None:
        """
        Validate configuration. Called only from to_icon_config().

        Raises:
            ValueError: If any parameter is invalid
        """
        # Allow custom datasets to bypass dataset validation
        if self.custom_dataset_type is None and self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}. Choose from: {SUPPORTED_DATASETS}")

        if self.custom_dataset_type is not None:
            if self.custom_dataset_type not in ['folder', 'embeddings', 'custom']:
                raise ValueError(f"Unsupported custom_dataset_type: {self.custom_dataset_type}. "
                                 f"Choose from: folder, embeddings, custom")
            if self.custom_dataset_path is None:
                raise ValueError("custom_dataset_path is required when using custom_dataset_type")
            if self.custom_dataset_type == 'custom' and self.custom_dataset_class is None:
                raise ValueError("custom_dataset_class is required when custom_dataset_type='custom'")

        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.backbone}. Choose from: {SUPPORTED_BACKBONES}")

        if self.icon_mode not in SUPPORTED_ICON_MODES:
            raise ValueError(f"Unsupported icon_mode: {self.icon_mode}. Choose from: {SUPPORTED_ICON_MODES}")

        if self.divergence not in SUPPORTED_DIVERGENCES:
            raise ValueError(f"Unsupported divergence: {self.divergence}. Choose from: {SUPPORTED_DIVERGENCES}")

        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Choose from: adam, adamw, sgd")

        if self.dataset != "mnist" and self.backbone in ["simplecnn", "mlp"]:
            raise ValueError(f"Backbone '{self.backbone}' is designed for MNIST only. Use ResNet for {self.dataset}.")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        if self.dataset == "mnist" and self.backbone.startswith("resnet"):
            warnings.warn(f"{self.backbone} may not be optimal for MNIST. Consider 'simplecnn' or 'mlp'.", UserWarning)

        if self.batch_size < 16:
            warnings.warn(f"Very small batch_size ({self.batch_size}). Training may be unstable.", UserWarning)

        if self.epochs > 200:
            warnings.warn(f"Large number of epochs ({self.epochs}). Training may take long.", UserWarning)

    def _create_mapper(self):
        """Create the encoder/mapper based on backbone selection."""
        if self.backbone == "simplecnn":
            # SimpleCNN is designed for 28x28 grayscale images (MNIST)
            return SimpleCNN(
                output_dim=self.embedding_dim,
                input_key="image",
                output_key="embedding"
            )
        elif self.backbone == "mlp":
            return MLPMapper(
                input_dim=28 * 28,
                hidden_dims=[512, 256],
                output_dim=self.embedding_dim,
                input_key="image",
                output_key="embedding"
            )
        else:
            return ResNet(
                model_type=self.backbone,
                small_image=(self.image_size <= 64),
                feat_dim=self.embedding_dim,
                input_key="image",
                output_key="embedding"
            )

    def _create_supervisory_distribution(self):
        """
        Create the supervisory distribution based on icon_mode.

        The supervisory distribution defines the "target" neighborhood structure
        that we want our learned embeddings to match.
        """
        if self.icon_mode == "simclr_like":
            # Augmentation-based: samples from the same image should be neighbors
            # This recovers InfoNCE/SimCLR-style contrastive learning
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "supervised":
            # Label-based: samples with the same class label should be neighbors
            return Label(
                input_key="label",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode in ["sne_like", "tsne_like", "cluster_like"]:
            # For SNE/t-SNE/clustering modes, we also use augmentation
            # but the difference is in the learned distribution kernel
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "barlow_twins_like":
            # Barlow Twins-style: augmentation-based (like SimCLR)
            # The key difference is in how we structure the learned distribution
            # to emphasize decorrelation/redundancy reduction
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "vicreg_like":
            # VICReg-style: augmentation-based
            # Uses variance-invariance-covariance principles
            # The learned distribution will emphasize these properties
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "debiasing_like":
            # Debiasing configuration: uses augmentation but with
            # modified parameters to reduce biases in the learned space
            # Key: we want to avoid spurious correlations
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

        else:
            return Augmentation(
                input_key="index",
                mask_diagonal=True,
                normalize=True
            )

    def _create_learned_distribution(self):
        """
        Create the learned distribution based on icon_mode.

        The learned distribution defines how we measure similarity in the
        learned embedding space.
        """
        if self.icon_mode == "simclr_like":
            # Gaussian kernel with cosine similarity (InfoNCE-like)
            return Gaussian(
                sigma=self.temperature,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "sne_like":
            # Gaussian kernel with Euclidean distance (SNE-like)
            return Gaussian(
                sigma=self.temperature,
                metric="euclidean",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "tsne_like":
            # Student-t kernel (t-SNE-like)
            # Heavy tails help with visualization and avoiding crowding
            return StudentT(
                gamma=1.0,
                df=1.0,  # degrees of freedom = 1 gives heavy tails like t-SNE
                metric="euclidean",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "supervised":
            # For supervised mode, use Gaussian with cosine similarity
            return Gaussian(
                sigma=self.temperature,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "cluster_like":
            # Student-t with learnable gamma for clustering
            return StudentT(
                gamma=1.0,
                df=1.0,
                learnable_gamma=True,
                metric="euclidean",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "barlow_twins_like":
            # Barlow Twins emphasizes decorrelation and redundancy reduction
            # We use Gaussian kernel but with a sharper temperature to encourage
            # decorrelated features (lower temperature = sharper, more decorrelated)
            # Note: True Barlow Twins uses a cross-correlation loss, but we approximate
            # this in I-Con's framework with a sharper Gaussian distribution
            sharp_temp = self.temperature * 0.3  # Sharper than default
            return Gaussian(
                sigma=sharp_temp,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "vicreg_like":
            # VICReg emphasizes variance, invariance, and covariance
            # We approximate this with a Student-t kernel that has:
            # - Heavy tails (encourages variance/spread)
            # - Learnable gamma (adapts covariance structure)
            # This isn't a perfect VICReg implementation but captures the spirit
            return StudentT(
                gamma=1.0,
                df=1.0,
                learnable_gamma=True,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        elif self.icon_mode == "debiasing_like":
            # Debiasing configuration: uses a Gaussian kernel with
            # a moderate temperature to avoid overly sharp distributions
            # that might reinforce biases. The goal is balanced similarity
            # that doesn't amplify spurious correlations.
            balanced_temp = max(self.temperature, 0.7)  # Ensure not too sharp
            return Gaussian(
                sigma=balanced_temp,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

        else:
            return Gaussian(
                sigma=self.temperature,
                metric="cosine",
                input_key="embedding",
                mask_diagonal=True,
                normalize=True
            )

    def _get_num_classes(self) -> int:
        """Return the number of classes for the selected dataset."""
        num_classes_map = {
            "cifar10": 10,
            "cifar100": 100,
            "mnist": 10,
            "stl10": 10,
        }
        return num_classes_map.get(self.dataset, 10)

    def to_icon_config(self) -> Config:
        """
        Convert this PlaygroundConfig to the internal I-Con Config.

        This method creates all the necessary components (mapper, supervisory
        distribution, learned distribution) and wraps them in a Config object
        that can be passed to the I-Con Model.

        Returns:
            Config: The internal I-Con configuration object.
        """
        self.validate()

        mapper = self._create_mapper()
        supervisory_distribution = self._create_supervisory_distribution()
        learned_distribution = self._create_learned_distribution()

        return Config(
            mapper=mapper,
            supervisory_distribution=supervisory_distribution,
            learned_distribution=learned_distribution,
            num_classes=self._get_num_classes(),
            lr=self.learning_rate,
            optimizer=self.optimizer,
            weight_decay=self.weight_decay,
            divergence=self.divergence,
            linear_probe=self.use_linear_probe,
            accuracy_mode="unsupervised" if self.icon_mode != "supervised" else "regular",
            use_ema=False,
            loss_type="ce",  # Cross-entropy (I-Con default)
        )

    def describe(self) -> str:
        """Return a human-readable description of this configuration."""
        mode_descriptions = {
            "simclr_like": "Contrastive learning (InfoNCE-style) with augmentation pairs",
            "sne_like": "SNE-style dimensionality reduction with Gaussian kernel",
            "tsne_like": "t-SNE-style dimensionality reduction with Student-t kernel",
            "supervised": "Supervised learning with label-based neighborhood structure",
            "cluster_like": "Clustering-oriented learning with learnable Student-t kernel",
            "barlow_twins_like": "Barlow Twins-style decorrelation and redundancy reduction",
            "vicreg_like": "VICReg-style variance-invariance-covariance regularization",
            "debiasing_like": "Debiasing configuration with balanced similarity structure",
        }

        return f"""
I-Con Playground Configuration
==============================
Run Name:    {self.run_name}
Dataset:     {self.dataset.upper()} ({self._get_num_classes()} classes)
Backbone:    {self.backbone}
I-Con Mode:  {self.icon_mode}
             -> {mode_descriptions.get(self.icon_mode, '')}

Training:
  Epochs:         {self.epochs}
  Batch size:     {self.batch_size}
  Learning rate:  {self.learning_rate}
  Temperature:    {self.temperature}
  Embedding dim:  {self.embedding_dim}
  Save checkpoints: {self.save_checkpoints}

Output: {self.output_dir}
""".strip()

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary for logging."""
        result = {
            "run_name": self.run_name,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "icon_mode": self.icon_mode,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "embedding_dim": self.embedding_dim,
            "divergence": self.divergence,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "image_size": self.image_size,
            "use_augmentation": self.use_augmentation,
            "seed": self.seed,
            "use_linear_probe": self.use_linear_probe,
            "save_checkpoints": self.save_checkpoints,
        }
        # Add custom dataset info if present
        if self.custom_dataset_type is not None:
            result["custom_dataset_type"] = self.custom_dataset_type
            result["custom_dataset_path"] = self.custom_dataset_path
            if self.custom_dataset_class is not None:
                result["custom_dataset_class"] = self.custom_dataset_class
        return result


# ---------------------------------------------------------------------------
# Helper function to create common configurations
# ---------------------------------------------------------------------------

def quick_config(
    preset: Literal[
        "cifar_contrastive",
        "cifar_supervised",
        "mnist_tsne",
        "cifar_barlow",
        "cifar_vicreg",
    ] = "cifar_contrastive",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> PlaygroundConfig:
    """
    Create a PlaygroundConfig from a named preset for quick experimentation.

    Presets:
        - "cifar_contrastive": SimCLR-like contrastive learning on CIFAR-10
        - "cifar_supervised": Supervised I-Con on CIFAR-10
        - "mnist_tsne": t-SNE-style embedding on MNIST
        - "cifar_barlow": Barlow Twins-style on CIFAR-10
        - "cifar_vicreg": VICReg-style on CIFAR-10

    Args:
        preset: Name of the preset configuration
        epochs: Number of training epochs
        output_dir: Directory for outputs

    Returns:
        PlaygroundConfig: A configured PlaygroundConfig instance
    """
    if preset == "cifar_contrastive":
        return PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="simclr_like",
            epochs=epochs,
            batch_size=256,
            learning_rate=1e-3,
            temperature=0.5,
            output_dir=output_dir,
        )
    elif preset == "cifar_supervised":
        return PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="supervised",
            epochs=epochs,
            batch_size=256,
            learning_rate=1e-3,
            temperature=0.5,
            output_dir=output_dir,
        )
    elif preset == "mnist_tsne":
        return PlaygroundConfig(
            dataset="mnist",
            backbone="simplecnn",
            icon_mode="tsne_like",
            epochs=epochs,
            batch_size=256,
            learning_rate=1e-3,
            temperature=1.0,
            output_dir=output_dir,
        )
    elif preset == "cifar_barlow":
        return PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="barlow_twins_like",
            epochs=epochs,
            batch_size=256,
            learning_rate=1e-3,
            temperature=0.5,  # Will be sharpened in _create_learned_distribution
            output_dir=output_dir,
        )
    elif preset == "cifar_vicreg":
        return PlaygroundConfig(
            dataset="cifar10",
            backbone="resnet18",
            icon_mode="vicreg_like",
            epochs=epochs,
            batch_size=256,
            learning_rate=1e-3,
            temperature=0.7,
            output_dir=output_dir,
        )
    else:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from: "
            f"cifar_contrastive, cifar_supervised, mnist_tsne, cifar_barlow, cifar_vicreg"
        )
