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
Examples:
  # Run a SimCLR-like experiment on CIFAR-10
  python -m playground.playground_cli --dataset cifar10 --backbone resnet18 --icon_mode simclr_like --epochs 10

  # Use a preset configuration
  python -m playground.playground_cli --preset cifar_contrastive --epochs 5

  # Run a recipe (multiple experiments)
  python -m playground.playground_cli --recipe simclr_vs_sne --dataset cifar10 --epochs 10

  # Run t-SNE-like embedding on MNIST
  python -m playground.playground_cli --dataset mnist --backbone simplecnn --icon_mode tsne_like --epochs 20

  # Only run probes on existing embeddings
  python -m playground.playground_cli --probe_only --load_dir playground_runs/my_experiment

  # Run without visualization (headless mode)
  python -m playground.playground_cli --dataset cifar10 --icon_mode simclr_like --no_viz
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
        help="I-Con mode/objective preset (default: simclr_like)",
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
        help="Skip visualization (for headless runs)",
    )
    parser.add_argument(
        "--no_probe",
        action="store_true",
        help="Skip running linear probe after training",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Use t-SNE instead of PCA for embedding visualization (slower but often nicer)",
    )

    # Probe-only mode
    parser.add_argument(
        "--probe_only",
        action="store_true",
        help="Only run probes and visualization on existing embeddings",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load existing embeddings from (required with --probe_only)",
    )

    # Advanced options
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (ignore GPU)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU training if available (default: CPU)",
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
        help="Save epoch-by-epoch embedding frames and create animated GIF "
             "showing learning dynamics (slower, but reveals temporal patterns)",
    )

    # Custom dataset options
    custom_group = parser.add_argument_group('Custom datasets')
    custom_group.add_argument(
        "--custom_dataset",
        type=str,
        help="Path to custom dataset. Format: 'type:path' where type is 'folder', 'embeddings', or 'custom'. "
             "Examples: 'folder:/path/to/images', 'embeddings:features.npz', 'custom:my_dataset.py:MyDatasetClass'",
    )
    custom_group.add_argument(
        "--custom_transform",
        type=str,
        help="Custom transform to apply to images (folder datasets only). "
             "Format: 'resize:224,normalize:imagenet' or 'resize:28,flatten' for MNIST-like",
    )

    return parser


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

        gpu_setting = True if args.gpu else (False if args.cpu else None)

        try:
            results = run_playground_experiment_pure_pytorch(
                config,
                verbose=not args.quiet,
                save_checkpoints=args.save_checkpoints,
                gpu=gpu_setting,
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
        if args.custom_dataset:
            parts = args.custom_dataset.split(':')
            if len(parts) < 2:
                print("Error: --custom_dataset format should be 'type:path' or 'custom:path:ClassName'")
                sys.exit(1)
            custom_type = parts[0]
            custom_path = parts[1]
            if custom_type == 'custom' and len(parts) >= 3:
                custom_class = parts[2]

        config = PlaygroundConfig(
            dataset=args.dataset,
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
        )

    gpu_setting = True if args.gpu else (False if args.cpu else None)

    try:
        results = run_playground_experiment_pure_pytorch(
            config,
            verbose=not args.quiet,
            save_checkpoints=args.save_checkpoints,
            gpu=gpu_setting,
            debug_device=args.debug_device,
            save_epoch_gifs=args.save_epoch_gifs,
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

        # Embedding visualization
        method = "tsne" if args.tsne else "pca"
        print(f"\nPlotting embeddings using {method.upper()}...")
        plot_embeddings_2d(
            results["embeddings"],
            results["labels"],
            method=method,
            out_path=str(run_dir / f"embeddings_{method}.png"),
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
    print(f"Results saved to: {run_dir}")
    print(f"\nKey files:")
    print(f"  - Config:     {run_dir / 'config.json'}")
    print(f"  - Embeddings: {run_dir / 'embeddings.npz'}")
    print(f"  - Logs:       {run_dir / 'logs.json'}")
    print(f"  - Model:      {run_dir / 'final_model.pt'}")
    if not args.no_viz:
        print(f"  - Plots:      {run_dir / 'training_curves.png'}")
        print(f"              {run_dir / f'embeddings_{method}.png'}")
        print(f"              {run_dir / 'distance_histograms.png'}")
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

    # Handle probe-only mode
    if args.probe_only:
        run_probe_only(args)
    elif args.recipe:
        run_recipe(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
