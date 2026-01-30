"""
Playground Recipes Module

Pre-configured experiment recipes for teaching and comparative analysis.
Each recipe returns one or more PlaygroundConfig objects designed to
demonstrate specific concepts or compare different I-Con configurations.
"""

from typing import List
from .playground_config import PlaygroundConfig


# ---------------------------------------------------------------------------
# Recipe Functions
# ---------------------------------------------------------------------------

def get_recipe_simclr_vs_sne(
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Compare SimCLR-like contrastive learning vs SNE-like embedding.

    This recipe demonstrates how different learned distribution kernels
    affect the geometry of the embedding space:
    - SimCLR-like: Gaussian kernel with cosine similarity
    - SNE-like: Gaussian kernel with Euclidean distance

    Both use augmentation as the supervisory signal, but differ in how
    they measure similarity in the embedding space.

    Args:
        dataset: Which dataset to use
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 2 PlaygroundConfig objects
    """
    configs = [
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="simclr_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_simclr",
        ),
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="sne_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_sne",
        ),
    ]
    return configs


def get_recipe_backbone_comparison(
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Compare different backbone architectures on the same task.

    This recipe demonstrates how encoder capacity affects representation quality.
    Uses the same I-Con objective (SimCLR-like) but varies the backbone.

    Args:
        dataset: Which dataset to use (must support ResNets)
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 2-3 PlaygroundConfig objects with different backbones
    """
    if dataset == "mnist":
        # MNIST-specific backbones
        configs = [
            PlaygroundConfig(
                dataset=dataset,
                backbone="mlp",
                icon_mode="simclr_like",
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_mlp",
            ),
            PlaygroundConfig(
                dataset=dataset,
                backbone="simplecnn",
                icon_mode="simclr_like",
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_simplecnn",
            ),
        ]
    else:
        # RGB datasets - compare ResNet variants
        configs = [
            PlaygroundConfig(
                dataset=dataset,
                backbone="resnet18",
                icon_mode="simclr_like",
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_resnet18",
            ),
            PlaygroundConfig(
                dataset=dataset,
                backbone="resnet34",
                icon_mode="simclr_like",
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_resnet34",
            ),
        ]
    return configs


def get_recipe_supervised_vs_unsupervised(
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Compare supervised vs unsupervised (contrastive) learning.

    This recipe demonstrates the key difference between using labels
    vs augmentations as the supervisory signal:
    - Supervised: Uses class labels directly
    - SimCLR-like: Uses augmentation pairs (no labels during training)

    Both configurations can be evaluated with the same linear probe to
    compare how well they learn class-discriminative features.

    Args:
        dataset: Which dataset to use
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 2 PlaygroundConfig objects
    """
    configs = [
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="supervised",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_supervised",
        ),
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="simclr_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_unsupervised",
        ),
    ]
    return configs


def get_recipe_temperature_sweep(
    dataset: str = "cifar10",
    icon_mode: str = "simclr_like",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Sweep over different temperature values to see their effect.

    Temperature controls the sharpness of the similarity distribution:
    - Lower temperature (0.1): Sharper, more discriminative
    - Higher temperature (1.0): Softer, more tolerant

    Args:
        dataset: Which dataset to use
        icon_mode: Which I-Con mode to use
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 3 PlaygroundConfig objects with different temperatures
    """
    temperatures = [0.1, 0.5, 1.0]
    configs = []

    for temp in temperatures:
        configs.append(
            PlaygroundConfig(
                dataset=dataset,
                backbone="resnet18" if dataset != "mnist" else "simplecnn",
                icon_mode=icon_mode,
                temperature=temp,
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_{icon_mode}_temp{temp}",
            )
        )

    return configs


def get_recipe_kernel_comparison(
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Compare different learned distribution kernels.

    This recipe demonstrates how the choice of kernel affects the
    embedding space geometry:
    - Gaussian (SimCLR-like): Standard contrastive
    - Student-t (t-SNE-like): Heavy-tailed, good for visualization
    - Barlow Twins-like: Sharper, emphasizes decorrelation

    Args:
        dataset: Which dataset to use
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 3 PlaygroundConfig objects with different kernels
    """
    configs = [
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="simclr_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_gaussian",
        ),
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="tsne_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_studentt",
        ),
        PlaygroundConfig(
            dataset=dataset,
            backbone="resnet18" if dataset != "mnist" else "simplecnn",
            icon_mode="barlow_twins_like",
            epochs=epochs,
            output_dir=output_dir,
            run_name=f"{dataset}_barlow",
        ),
    ]
    return configs


def get_recipe_full_comparison(
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs"
) -> List[PlaygroundConfig]:
    """
    Comprehensive comparison of all major I-Con modes.

    This recipe runs experiments with all major presets to give
    a complete picture of how different configurations affect
    the learned representations.

    Note: This recipe takes longer to run (5+ experiments).

    Args:
        dataset: Which dataset to use
        epochs: Number of training epochs per config
        output_dir: Base output directory

    Returns:
        List of 5 PlaygroundConfig objects covering major modes
    """
    modes = ["simclr_like", "sne_like", "tsne_like", "supervised", "barlow_twins_like"]
    configs = []

    for mode in modes:
        configs.append(
            PlaygroundConfig(
                dataset=dataset,
                backbone="resnet18" if dataset != "mnist" else "simplecnn",
                icon_mode=mode,
                epochs=epochs,
                output_dir=output_dir,
                run_name=f"{dataset}_{mode}",
            )
        )

    return configs


# ---------------------------------------------------------------------------
# Recipe Registry
# ---------------------------------------------------------------------------

AVAILABLE_RECIPES = {
    "simclr_vs_sne": get_recipe_simclr_vs_sne,
    "backbone_comparison": get_recipe_backbone_comparison,
    "supervised_vs_unsupervised": get_recipe_supervised_vs_unsupervised,
    "temperature_sweep": get_recipe_temperature_sweep,
    "kernel_comparison": get_recipe_kernel_comparison,
    "full_comparison": get_recipe_full_comparison,
}


def get_recipe(
    recipe_name: str,
    dataset: str = "cifar10",
    epochs: int = 10,
    output_dir: str = "playground_runs",
    # CLI overrides - applied to ALL configs in recipe
    divergence: str = None,
    optimizer: str = None,
    weight_decay: float = None,
    num_workers: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    temperature: float = None,
    embedding_dim: int = None,
    **kwargs
) -> List[PlaygroundConfig]:
    """
    Get a recipe by name with optional CLI overrides.

    Args:
        recipe_name: Name of the recipe. Options:
            - "simclr_vs_sne": Compare SimCLR vs SNE
            - "backbone_comparison": Compare different backbones
            - "supervised_vs_unsupervised": Compare supervised vs unsupervised
            - "temperature_sweep": Sweep over temperatures
            - "kernel_comparison": Compare different kernels
            - "full_comparison": Comprehensive comparison of all modes
        dataset: Which dataset to use
        epochs: Number of training epochs per config
        output_dir: Base output directory
        divergence: Override divergence for all configs (None = use recipe default)
        optimizer: Override optimizer for all configs (None = use recipe default)
        weight_decay: Override weight_decay for all configs (None = use recipe default)
        num_workers: Override num_workers for all configs (None = use recipe default)
        batch_size: Override batch_size for all configs (None = use recipe default)
        learning_rate: Override learning_rate for all configs (None = use recipe default)
        temperature: Override temperature for all configs (None = use recipe default)
        embedding_dim: Override embedding_dim for all configs (None = use recipe default)
        **kwargs: Additional keyword arguments passed to the recipe function

    Returns:
        List of PlaygroundConfig objects with CLI overrides applied

    Example:
        >>> configs = get_recipe("simclr_vs_sne", dataset="cifar10", epochs=5, divergence="js")
        >>> for config in configs:
        ...     print(f"{config.run_name}: divergence={config.divergence}")
    """
    if recipe_name not in AVAILABLE_RECIPES:
        available = ", ".join(AVAILABLE_RECIPES.keys())
        raise ValueError(
            f"Unknown recipe: {recipe_name}. "
            f"Available recipes: {available}"
        )

    recipe_fn = AVAILABLE_RECIPES[recipe_name]
    configs = recipe_fn(dataset=dataset, epochs=epochs, output_dir=output_dir, **kwargs)

    # Apply CLI overrides to all configs
    overrides_applied = []
    for config in configs:
        if divergence is not None:
            config.divergence = divergence
            overrides_applied.append(f"divergence={divergence}")
        if optimizer is not None:
            config.optimizer = optimizer
            overrides_applied.append(f"optimizer={optimizer}")
        if weight_decay is not None:
            config.weight_decay = weight_decay
            overrides_applied.append(f"weight_decay={weight_decay}")
        if num_workers is not None:
            config.num_workers = num_workers
            overrides_applied.append(f"num_workers={num_workers}")
        if batch_size is not None:
            config.batch_size = batch_size
            overrides_applied.append(f"batch_size={batch_size}")
        if learning_rate is not None:
            config.learning_rate = learning_rate
            overrides_applied.append(f"learning_rate={learning_rate}")
        if temperature is not None:
            config.temperature = temperature
            overrides_applied.append(f"temperature={temperature}")
        if embedding_dim is not None:
            config.embedding_dim = embedding_dim
            overrides_applied.append(f"embedding_dim={embedding_dim}")

    return configs


def list_recipes() -> List[str]:
    """
    List all available recipe names.

    Returns:
        List of recipe names
    """
    return list(AVAILABLE_RECIPES.keys())
