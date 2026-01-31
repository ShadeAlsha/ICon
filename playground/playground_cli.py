#!/usr/bin/env python3
"""
I-Con Playground CLI

Command-line interface for running I-Con experiments from the terminal.
Provides an easy way to train models, visualize results, and run probes.

Usage:
    # Single experiment
    python -m playground.playground_cli --dataset cifar10 --backbone resnet18 --icon_mode simclr_like --epochs 10

    # Recipe (multi-experiment)
    python -m playground.playground_cli --recipe simclr_vs_sne --epochs 10

    # Or with the shorthand:
    python -m playground.playground_cli --preset cifar_contrastive --epochs 5
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="playground",
        description="I-Con Playground: Run small I-Con experiments with ease",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical Workflow:
  1. Prepare dataset (use built-in or --custom_dataset)
  2. Run training (specify --icon_mode and --epochs)
  3. Inspect embeddings (PCA/UMAP plots in output directory)
  4. Interpret probes (linear/k-NN accuracies measure quality)

Examples:

  Quick start (uses sensible defaults):
    python -m playground.playground_cli

  Standard datasets:
    python -m playground.playground_cli --dataset cifar10 --icon_mode simclr_like --epochs 10
    python -m playground.playground_cli --preset cifar_contrastive --epochs 5

  Embedding dimension ablation:
    python -m playground.playground_cli --ablate_embedding_dim --epochs 50

  Custom data (bring your own):
    # Image folder organized by class
    python -m playground.playground_cli --custom_dataset "folder:/path/to/images" --epochs 50

    # Pre-computed embeddings
    python -m playground.playground_cli --custom_dataset "embeddings:features.npz" --epochs 100

  Visualization:
    # Generate animated GIF of learning dynamics
    python -m playground.playground_cli --dataset cifar10 --viz_mode both --gif_every 5

    # Regenerate GIF with different settings (no retraining)
    python -m playground.playground_cli --regen_gif --load_dir playground_runs/my_run --gif_method tsne

  Multiple experiments:
    python -m playground.playground_cli --recipe simclr_vs_sne --epochs 10

For custom data documentation, see: playground/CUSTOM_DATA_GUIDE.md
For visualization options, see: playground/VISUALIZATION_GUIDE.md
        """,
    )

    # Preset/Recipe option (mutually exclusive groups)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preset",
        type=str,
        choices=["cifar_contrastive", "cifar_supervised", "mnist_tsne", "cifar_barlow", "cifar_vicreg"],
        help="Use a preset configuration instead of specifying each option",
    )
    mode_group.add_argument(
        "--recipe",
        type=str,
        help="Run a pre-configured recipe (multi-experiment). Available: simclr_vs_sne, backbone_comparison, supervised_vs_unsupervised, temperature_sweep, kernel_comparison, full_comparison",
    )

    # Dataset and model configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "mnist", "stl10"],
        help="Dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "simplecnn", "mlp"],
        help="Backbone encoder architecture (default: resnet18)",
    )
    parser.add_argument(
        "--icon_mode",
        type=str,
        default="simclr_like",
        choices=["simclr_like", "sne_like", "tsne_like", "supervised", "cluster_like",
                 "barlow_twins_like", "vicreg_like", "debiasing_like"],
        help="I-Con mode/objective. simclr_like: contrastive learning from augmentations "
             "(learns invariances to crops/color/blur). sne_like/tsne_like: neighbor-preserving embeddings. "
             "supervised: uses class labels directly. (default: simclr_like)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--lr", "--learning_rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature/sigma for learned distribution (default: 0.5)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--divergence",
        type=str,
        default="kl",
        choices=["kl", "rkl", "js"],
        help="Divergence measure: kl=KL(p||q), rkl=KL(q||p), js=Jensen-Shannon (default: kl)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer type (default: adamw)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer (default: 1e-4)",
    )

    # Ablation study
    parser.add_argument(
        "--ablate_embedding_dim",
        action="store_true",
        help="Run controlled ablation: train with embedding_dim=64 and 128, then compare. "
             "Generates comparison plot and JSON with probe accuracies.",
    )

    # Output and runtime
    parser.add_argument(
        "--out_dir", "--output_dir",
        type=str,
        default="playground_runs",
        dest="output_dir",
        help="Output directory for logs and results (default: playground_runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )

    # Visualization and probes
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip visualization (for headless runs) - equivalent to --viz_mode none",
    )
    parser.add_argument(
        "--no_probe",
        action="store_true",
        help="Skip evaluation probes. Probes measure representation quality by training "
             "a simple classifier (linear or k-NN) on frozen embeddings. Higher probe "
             "accuracy = better learned representations.",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Use t-SNE instead of PCA for static embedding visualization (slower but often nicer)",
    )

    # Visualization control section
    viz_group = parser.add_argument_group('Visualization controls',
        'Fine-grained control over epoch visualization and GIF generation')
    viz_group.add_argument(
        "--viz_mode",
        type=str,
        default="both",
        choices=["none", "static", "gif", "both"],
        help="Visualization output mode: none (no viz), static (final only), "
             "gif (animation only), both (default)",
    )
    viz_group.add_argument(
        "--gif_every",
        type=int,
        default=1,
        metavar="N",
        help="Save a GIF frame every N epochs (default: 1). Example: --gif_every 5",
    )
    viz_group.add_argument(
        "--gif_method",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="Projection method for GIF: pca (fast, default), tsne (slower), umap (requires umap-learn)",
    )
    viz_group.add_argument(
        "--gif_fps",
        type=float,
        default=2.0,
        help="Frames per second for GIF playback (default: 2.0)",
    )
    viz_group.add_argument(
        "--gif_max_points",
        type=int,
        default=5000,
        help="Maximum points to plot in GIF (subsample for performance, default: 5000)",
    )
    viz_group.add_argument(
        "--gif_overlay",
        type=str,
        default="epoch",
        choices=["none", "loss", "epoch"],
        help="Text overlay on GIF frames: none, loss (train/val loss), epoch (default)",
    )
    viz_group.add_argument(
        "--no_save_frames",
        action="store_true",
        help="Don't save individual frame PNGs (only keep the GIF)",
    )

    # Probe-only and regen modes
    parser.add_argument(
        "--probe_only",
        action="store_true",
        help="Only run probes and visualization on existing embeddings",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load existing experiment from (used with --probe_only or --regen_gif)",
    )
    parser.add_argument(
        "--regen_gif",
        action="store_true",
        help="Regenerate GIF from saved embeddings without retraining. "
             "Requires --load_dir. Allows changing GIF settings (--gif_method, --gif_fps, etc.)",
    )

    # Advanced options
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )

    # Device selection (primary interface)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection for training. "
             "'auto' automatically selects CUDA > MPS > CPU. "
             "'cuda' requires CUDA GPU (fails if unavailable). "
             "'mps' requires Apple Silicon GPU (fails if unavailable). "
             "'cpu' forces CPU execution. (default: auto)",
    )

    # Backward compatibility flags (deprecated)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="[DEPRECATED] Use --device cpu instead. Force CPU training.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="[DEPRECATED] Use --device auto instead. Enable GPU if available.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--debug_device",
        action="store_true",
        help="Print detailed device placement info for debugging GPU issues",
    )
    parser.add_argument(
        "--save_epoch_gifs",
        action="store_true",
        help="[DEPRECATED] Use --viz_mode gif or --viz_mode both instead. "
             "This flag is kept for backwards compatibility.",
    )

    # Custom dataset options
    custom_group = parser.add_argument_group('Custom datasets (bring your own data)',
        'Run I-Con on your own images or embeddings. See CUSTOM_DATA_GUIDE.md for details.')
    custom_group.add_argument(
        "--custom_dataset",
        type=str,
        metavar="TYPE:PATH",
        help="""Load custom data instead of CIFAR/MNIST. Formats:

  Image folders (organized by class):
    --custom_dataset "folder:/path/to/images"

  Pre-computed embeddings (.npz or .pt):
    --custom_dataset "embeddings:/path/to/features.npz"

  Custom PyTorch Dataset class:
    --custom_dataset "custom:/path/to/dataset.py:MyDatasetClass"

See playground/CUSTOM_DATA_GUIDE.md for complete examples.""",
    )
    custom_group.add_argument(
        "--custom_transform",
        type=str,
        metavar="SPEC",
        help="Transform for folder datasets: 'resize:224,normalize:imagenet' (default) "
             "or 'resize:28,grayscale' for smaller images",
    )

    return parser


def resolve_device_arg(args: argparse.Namespace) -> str:
    """
    Resolve device selection from CLI arguments, handling backward compatibility.

    Args:
        args: Parsed command-line arguments

    Returns:
        Device string: 'auto', 'cpu', 'cuda', or 'mps'
    """
    # Handle backward compatibility with --cpu and --gpu flags
    if args.cpu and args.gpu:
        print("Warning: Both --cpu and --gpu specified. Using --cpu.")
        return "cpu"

    if args.cpu:
        print("Note: --cpu is deprecated. Use --device cpu instead.")
        return "cpu"

    if args.gpu:
        print("Note: --gpu is deprecated. Use --device auto instead.")
        return "auto"  # --gpu meant "use GPU if available"

    # Use the explicit --device argument
    return args.device


def run_regen_gif(args: argparse.Namespace) -> None:
    """Regenerate GIF from saved embeddings without retraining."""
    from playground.viz import regenerate_gif_from_run, VizConfig, VizMode, GifMethod, GifOverlay

    if not args.load_dir:
        print("Error: --load_dir is required with --regen_gif")
        sys.exit(1)

    load_dir = Path(args.load_dir)
    if not load_dir.exists():
        print(f"Error: Directory not found: {load_dir}")
        sys.exit(1)

    print(f"\nRegenerating GIF from: {load_dir}")

    # Build VizConfig from CLI args
    config = VizConfig(
        viz_mode=VizMode.GIF,  # Force GIF mode for regen
        gif_every=args.gif_every,
        gif_method=GifMethod(args.gif_method.lower()),
        gif_fps=args.gif_fps,
        gif_max_points=args.gif_max_points,
        gif_overlay=GifOverlay(args.gif_overlay.lower()),
        save_frames=not args.no_save_frames,
    )

    try:
        result = regenerate_gif_from_run(
            run_dir=load_dir,
            config=config,
        )
        print(f"\nGIF regenerated successfully!")
        print(f"  Output: {result['gif_path']}")
        if result.get('sanity_check', {}).get('warning'):
            print(f"\n  WARNING: {result['sanity_check']['warning']}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure the experiment was run with GIF generation enabled (--viz_mode gif or --viz_mode both)")
        sys.exit(1)
    except Exception as e:
        print(f"\nError regenerating GIF: {e}")
        sys.exit(1)


def run_probe_only(args: argparse.Namespace) -> None:
    """Run probes and visualization on existing embeddings."""
    from playground.playground_runner import load_experiment_results
    from playground.playground_viz import (
        plot_training_curves,
        plot_embeddings_2d,
        create_experiment_summary,
    )
    from playground.playground_probes import run_linear_probe, run_knn_probe, analyze_class_separability

    if not args.load_dir:
        print("Error: --load_dir is required with --probe_only")
        sys.exit(1)

    load_dir = Path(args.load_dir)
    if not load_dir.exists():
        print(f"Error: Directory not found: {load_dir}")
        sys.exit(1)

    print(f"\nLoading experiment from: {load_dir}")
    results = load_experiment_results(str(load_dir))

    if "embeddings" not in results or len(results["embeddings"]) == 0:
        print("Error: No embeddings found in the specified directory")
        sys.exit(1)

    print(f"Loaded {len(results['embeddings'])} embeddings")

    # Run linear probe
    if not args.no_probe:
        print("\nRunning linear probe...")
        probe_results = run_linear_probe(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

        # Also run kNN probe
        print("\nRunning kNN probe...")
        knn_results = run_knn_probe(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

        # Also run separability analysis
        analyze_class_separability(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

    # Visualization
    if not args.no_viz:
        # Training curves
        if "logs" in results and results["logs"].get("train_losses"):
            print("\nPlotting training curves...")
            plot_training_curves(
                results["logs"],
                out_path=str(load_dir / "training_curves.png"),
                show=True,
            )

        # Embedding visualization
        print(f"\nPlotting embeddings ({'t-SNE' if args.tsne else 'PCA'})...")
        method = "tsne" if args.tsne else "pca"
        plot_embeddings_2d(
            results["embeddings"],
            results["labels"],
            method=method,
            out_path=str(load_dir / f"embeddings_{method}.png"),
            show=True,
        )

    print("\nDone!")


def run_recipe(args: argparse.Namespace) -> None:
    """Run a recipe with multiple configurations."""
    from playground.playground_recipes import get_recipe, list_recipes
    from playground.playground_runner import run_playground_experiment_pure_pytorch
    from playground.playground_viz import compare_embedding_spaces
    from playground.playground_probes import run_linear_probe, run_knn_probe
    import json

    # Validate recipe name
    available_recipes = list_recipes()
    if args.recipe not in available_recipes:
        print(f"Error: Unknown recipe '{args.recipe}'")
        print(f"Available recipes: {', '.join(available_recipes)}")
        sys.exit(1)

    print(f"\nRunning recipe: {args.recipe}")

    # Pass CLI overrides to recipe (these will be applied to ALL configs)
    configs = get_recipe(
        args.recipe,
        dataset=args.dataset,
        epochs=args.epochs,
        output_dir=args.output_dir,
        divergence=args.divergence if args.divergence != "kl" else None,  # Only override if non-default
        optimizer=args.optimizer if args.optimizer != "adamw" else None,
        weight_decay=args.weight_decay if args.weight_decay != 1e-4 else None,
        num_workers=args.num_workers if args.num_workers != 4 else None,
        batch_size=args.batch_size if args.batch_size != 256 else None,
        learning_rate=args.learning_rate if args.learning_rate != 1e-3 else None,
        temperature=args.temperature if args.temperature != 0.5 else None,
        embedding_dim=args.embedding_dim if args.embedding_dim != 128 else None,
    )

    # Print any overrides applied
    sample_cfg = configs[0]
    overrides = []
    if args.divergence != "kl":
        overrides.append(f"divergence={args.divergence}")
    if args.optimizer != "adamw":
        overrides.append(f"optimizer={args.optimizer}")
    if args.num_workers != 4:
        overrides.append(f"num_workers={args.num_workers}")
    if args.temperature != 0.5:
        overrides.append(f"temperature={args.temperature}")
    if args.embedding_dim != 128:
        overrides.append(f"embedding_dim={args.embedding_dim}")
    if args.batch_size != 256:
        overrides.append(f"batch_size={args.batch_size}")
    if args.learning_rate != 1e-3:
        overrides.append(f"learning_rate={args.learning_rate}")
    if args.weight_decay != 1e-4:
        overrides.append(f"weight_decay={args.weight_decay}")
    if overrides:
        print(f"CLI overrides applied to all configs: {', '.join(overrides)}")

    print(f"\nThis recipe includes {len(configs)} experiments:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.run_name}")

    results_list = []
    summary = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT {i}/{len(configs)}: {config.run_name}")
        print(f"{'='*60}")

        device = resolve_device_arg(args)

        try:
            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=not args.quiet,
                save_checkpoints=args.save_checkpoints,
                device=device,
                debug_device=args.debug_device,
                save_epoch_gifs=args.save_epoch_gifs,
            )
            results_list.append(results)

            # Run probes
            if not args.no_probe and len(results["embeddings"]) > 0:
                probe_results = run_linear_probe(results["embeddings"], results["labels"], verbose=False)
                knn_results = run_knn_probe(results["embeddings"], results["labels"], verbose=False)

                summary.append({
                    "name": config.run_name,
                    "linear_probe_acc": probe_results['test_accuracy'],
                    "knn_probe_acc": knn_results['test_accuracy'],
                })

                print(f"\n  Linear Probe: {probe_results['test_accuracy']:.2%}")
                print(f"  kNN Probe (k=5): {knn_results['test_accuracy']:.2%}")

        except Exception as e:
            print(f"\nError running experiment {config.run_name}: {e}")
            print("Continuing with remaining experiments...")
            continue

    # Comparison visualization
    if not args.no_viz and len(results_list) > 0:
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON PLOTS")
        print(f"{'='*60}")
        comp_path = Path(args.output_dir) / f"recipe_{args.recipe}_comparison.png"
        compare_embedding_spaces(results_list, method="pca", out_path=str(comp_path), show=True)

    # Print summary
    if summary:
        print(f"\n{'='*60}")
        print("RECIPE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Experiment':<30} {'Linear Probe':<15} {'kNN Probe':<15}")
        print("-" * 60)
        for item in summary:
            print(f"{item['name']:<30} {item['linear_probe_acc']:>13.2%} {item['knn_probe_acc']:>13.2%}")

        # Save summary
        summary_path = Path(args.output_dir) / f"recipe_{args.recipe}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


def run_embedding_dim_ablation(args: argparse.Namespace) -> None:
    """
    Run controlled embedding dimension ablation study.

    Trains models with embedding_dim = 64 and 128, keeping all other
    hyperparameters constant. Generates comparison plot and JSON.

    This is a minimal, clean ablation for understanding how embedding
    dimension affects representation quality.
    """
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    from playground.playground_config import PlaygroundConfig
    from playground.playground_runner import run_playground_experiment_pure_pytorch
    from playground.playground_probes import run_linear_probe, run_knn_probe, analyze_class_separability

    if not args.quiet:
        print("\n" + "="*60)
        print("RUNNING EMBEDDING DIMENSION ABLATION")
        print("="*60)
        print("\nThis ablation compares embedding_dim = 64 vs 128")
        print("All other hyperparameters are held constant.\n")

    # Handle custom dataset parsing
    custom_type, custom_path, custom_class = None, None, None
    dataset_to_use = args.dataset

    if args.custom_dataset:
        parts = args.custom_dataset.split(':')
        if len(parts) < 2:
            print("Error: --custom_dataset format should be 'type:path'")
            sys.exit(1)
        custom_type = parts[0]
        custom_path = parts[1]
        if custom_type == 'custom' and len(parts) >= 3:
            custom_class = parts[2]
        dataset_to_use = "custom"

    # Determine viz mode
    if args.no_viz:
        viz_mode = "none"
    elif args.save_epoch_gifs:
        viz_mode = "both"
    else:
        viz_mode = args.viz_mode

    # Create base output directory for ablation
    base_output = Path(args.output_dir)
    ablation_name = f"ablation_embedding_dim_{args.icon_mode}_{args.dataset}"
    ablation_dir = base_output / ablation_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Two embedding dimensions to test
    embedding_dims = [64, 128]
    results_by_dim = {}

    device = resolve_device_arg(args)

    # Run experiments for each dimension
    for dim in embedding_dims:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH EMBEDDING_DIM = {dim}")
        print(f"{'='*60}\n")

        # Create config for this dimension
        config = PlaygroundConfig(
            dataset=dataset_to_use,
            backbone=args.backbone,
            icon_mode=args.icon_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            embedding_dim=dim,  # ABLATION VARIABLE
            divergence=args.divergence,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            output_dir=str(ablation_dir / f"dim_{dim}"),  # Separate subdirectory
            seed=args.seed,  # Same seed for fair comparison
            num_workers=args.num_workers,
            save_checkpoints=False,  # Don't need checkpoints for ablation
            custom_dataset_type=custom_type,
            custom_dataset_path=custom_path,
            custom_dataset_class=custom_class,
            viz_mode=viz_mode,
            gif_every=args.gif_every,
            gif_method=args.gif_method,
            gif_fps=args.gif_fps,
            gif_max_points=args.gif_max_points,
            gif_overlay=args.gif_overlay,
            save_frames=not args.no_save_frames,
        )

        # Run experiment
        try:
            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=not args.quiet,
                save_checkpoints=False,
                device=device,
                debug_device=args.debug_device,
            )
        except Exception as e:
            print(f"\nExperiment failed for dim={dim}: {e}")
            sys.exit(1)

        # Run probes if embeddings exist
        if len(results["embeddings"]) > 0:
            if not args.quiet:
                print(f"\nRunning evaluation probes for dim={dim}...")

            # Linear probe
            probe_results = run_linear_probe(
                results["embeddings"],
                results["labels"],
                verbose=not args.quiet,
            )

            # kNN probe
            knn_results = run_knn_probe(
                results["embeddings"],
                results["labels"],
                verbose=not args.quiet,
            )

            # Separability
            sep_results = analyze_class_separability(
                results["embeddings"],
                results["labels"],
                verbose=not args.quiet,
            )

            # Store results
            results_by_dim[dim] = {
                "embedding_dim": dim,
                "linear_probe_test_acc": probe_results.get("test_accuracy", 0.0),
                "knn_test_acc": knn_results.get("test_accuracy", 0.0),
                "separability_ratio": sep_results.get("separability_ratio", 0.0),
                "run_dir": results["paths"]["run_dir"],
            }

    # Generate comparison outputs
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}\n")

    # Create comparison JSON
    comparison = {
        "ablation_type": "embedding_dimension",
        "results": results_by_dim,
        "config": {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "icon_mode": args.icon_mode,
            "epochs": args.epochs,
            "seed": args.seed,
        }
    }

    comparison_path = ablation_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison saved to: {comparison_path}")

    # Print comparison table
    print("\nResults Summary:")
    print(f"{'Embedding Dim':<15} {'Linear Probe':<15} {'k-NN Probe':<15} {'Separability':<15}")
    print("-" * 60)
    for dim in embedding_dims:
        r = results_by_dim[dim]
        print(f"{dim:<15} {r['linear_probe_test_acc']:<15.4f} {r['knn_test_acc']:<15.4f} {r['separability_ratio']:<15.4f}")

    # Generate comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = list(results_by_dim.keys())
    linear_accs = [results_by_dim[d]["linear_probe_test_acc"] for d in dims]
    knn_accs = [results_by_dim[d]["knn_test_acc"] for d in dims]

    ax.plot(dims, linear_accs, 'o-', label='Linear Probe', linewidth=2, markersize=10)
    ax.plot(dims, knn_accs, 's-', label='k-NN Probe', linewidth=2, markersize=10)

    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Embedding Dimension Ablation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = ablation_dir / "embedding_dim_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plot saved to: {plot_path}")
    print(f"\nAll ablation results saved in: {ablation_dir}")
    print("\nAblation complete!")


def run_experiment(args: argparse.Namespace) -> None:
    """Run a full I-Con experiment."""
    from playground.playground_config import PlaygroundConfig, quick_config
    from playground.playground_runner import run_playground_experiment_pure_pytorch
    from playground.playground_viz import (
        plot_training_curves,
        plot_embeddings_2d,
        plot_distance_histograms,
        create_experiment_summary,
    )
    from playground.playground_probes import run_linear_probe, run_knn_probe, analyze_class_separability

    if args.preset:
        if not args.quiet:
            print(f"\nUsing preset: {args.preset}")
        config = quick_config(
            preset=args.preset,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
    else:
        # Parse custom dataset argument if provided
        custom_type, custom_path, custom_class = None, None, None
        dataset_to_use = args.dataset  # Default to CLI arg

        if args.custom_dataset:
            parts = args.custom_dataset.split(':')
            if len(parts) < 2:
                print("Error: --custom_dataset format should be 'type:path' or 'custom:path:ClassName'")
                sys.exit(1)
            custom_type = parts[0]
            custom_path = parts[1]
            if custom_type == 'custom' and len(parts) >= 3:
                custom_class = parts[2]

            # CRITICAL: Override dataset to "custom" when using custom_dataset
            # This prevents CIFAR/MNIST validation logic from running
            dataset_to_use = "custom"

        # Determine visualization mode
        # Handle backwards compatibility: --no_viz and --save_epoch_gifs
        if args.no_viz:
            viz_mode = "none"
        elif args.save_epoch_gifs:
            # Deprecated flag, map to new system
            viz_mode = "both"
        else:
            viz_mode = args.viz_mode

        config = PlaygroundConfig(
            dataset=dataset_to_use,
            backbone=args.backbone,
            icon_mode=args.icon_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            embedding_dim=args.embedding_dim,
            divergence=args.divergence,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            output_dir=args.output_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            save_checkpoints=args.save_checkpoints,
            custom_dataset_type=custom_type,
            custom_dataset_path=custom_path,
            custom_dataset_class=custom_class,
            # Visualization settings
            viz_mode=viz_mode,
            gif_every=args.gif_every,
            gif_method=args.gif_method,
            gif_fps=args.gif_fps,
            gif_max_points=args.gif_max_points,
            gif_overlay=args.gif_overlay,
            save_frames=not args.no_save_frames,
        )

    device = resolve_device_arg(args)

    try:
        results = run_playground_experiment_pure_pytorch(
            config,
            verbose=not args.quiet,
            save_checkpoints=args.save_checkpoints,
            device=device,
            debug_device=args.debug_device,
        )
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        sys.exit(1)

    run_dir = Path(results["paths"]["run_dir"])

    # Run linear probe
    if not args.no_probe and len(results["embeddings"]) > 0:
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION PROBES")
        print("=" * 60)

        # Linear probe
        probe_results = run_linear_probe(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

        # kNN probe
        knn_results = run_knn_probe(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

        # Save probe results
        import json
        probe_path = run_dir / "probe_results.json"
        with open(probe_path, "w") as f:
            json.dump({
                "linear_probe": probe_results,
                "knn_probe": knn_results,
            }, f, indent=2)

        # Separability analysis
        sep_results = analyze_class_separability(
            results["embeddings"],
            results["labels"],
            verbose=not args.quiet,
        )

    # Visualization
    if not args.no_viz and len(results["embeddings"]) > 0:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Training curves
        if results["logs"].get("train_losses"):
            print("\nPlotting training curves...")
            plot_training_curves(
                results["logs"],
                out_path=str(run_dir / "training_curves.png"),
                show=False,
            )

        # Embedding visualization: Generate PCA + UMAP (nonlinear)
        # PCA is always generated (fast, linear baseline)
        print("\nPlotting embeddings using PCA...")
        plot_embeddings_2d(
            results["embeddings"],
            results["labels"],
            method="pca",
            out_path=str(run_dir / "embeddings_pca.png"),
            show=False,
        )

        # UMAP as nonlinear projection (falls back to PCA if unavailable)
        print("\nPlotting embeddings using UMAP (nonlinear)...")
        plot_embeddings_2d(
            results["embeddings"],
            results["labels"],
            method="umap",
            out_path=str(run_dir / "embeddings_umap.png"),
            show=False,
        )

        # Optional t-SNE if user explicitly requests it
        if args.tsne:
            print("\nPlotting embeddings using t-SNE (user requested)...")
            plot_embeddings_2d(
                results["embeddings"],
                results["labels"],
                method="tsne",
                out_path=str(run_dir / "embeddings_tsne.png"),
                show=False,
            )

        # Distance histograms
        print("\nPlotting distance histograms...")
        plot_distance_histograms(
            results["embeddings"],
            results["labels"],
            out_path=str(run_dir / "distance_histograms.png"),
            show=False,
        )

        # Summary figure
        print("\nCreating experiment summary...")
        create_experiment_summary(
            results,
            out_path=str(run_dir / "experiment_summary.png"),
            show=False,
        )

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {run_dir}")

    print(f"\nSaved artifacts:")
    print(f"  Data:")
    print(f"    embeddings.npz       - Learned representations ({len(results['embeddings'])} samples)")
    print(f"    logs.json            - Training metrics")
    print(f"    config.json          - Full configuration")
    print(f"    final_model.pt       - Model weights")

    if not args.no_viz:
        print(f"\n  Visualizations:")
        print(f"    training_curves.png  - Loss/accuracy over epochs")
        print(f"    embeddings_pca.png   - 2D embedding plot (linear)")
        print(f"    embeddings_umap.png  - 2D embedding plot (nonlinear)")
        if args.tsne:
            print(f"    embeddings_tsne.png  - 2D embedding plot (t-SNE)")
        print(f"    distance_histograms.png")

    # Check if GIF was generated
    gif_path = run_dir / "training_dynamics.gif"
    if gif_path.exists():
        gif_size = gif_path.stat().st_size / 1024
        print(f"\n  Animation:")
        print(f"    training_dynamics.gif - Animated visualization ({gif_size:.1f} KB)")
        print(f"\n  To view the GIF:")
        print(f"    open \"{gif_path}\"")

    print(f"\n  To regenerate visualizations later:")
    print(f"    python -m playground.playground_cli --regen_gif --load_dir \"{run_dir}\"")
    print("=" * 60)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Print header
    if not args.quiet:
        print("""
╔═══════════════════════════════════════════════════════════╗
║                   I-CON PLAYGROUND                        ║
║   A student-friendly tool for I-Con experiments           ║
╚═══════════════════════════════════════════════════════════╝
        """)

    # Handle different modes
    if args.regen_gif:
        run_regen_gif(args)
    elif args.probe_only:
        run_probe_only(args)
    elif args.recipe:
        run_recipe(args)
    elif args.ablate_embedding_dim:
        run_embedding_dim_ablation(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
