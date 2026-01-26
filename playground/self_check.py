#!/usr/bin/env python3
"""
I-Con Playground Self-Check Script

Runs a quick smoke test of the playground to verify everything is working.
This is useful for:
- Testing after installation
- Verifying changes haven't broken core functionality
- Quick sanity check before demos or classes

Usage:
    python -m playground.self_check
"""

import sys
import os
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_self_check(verbose: bool = True) -> bool:
    """
    Run a comprehensive self-check of the playground.

    Tests:
    1. Import all modules
    2. Create a minimal config
    3. Run a 1-epoch experiment on MNIST
    4. Extract embeddings
    5. Run linear probe
    6. Run kNN probe
    7. Generate visualizations (in-memory)
    8. Test recipe system

    Args:
        verbose: Whether to print detailed progress

    Returns:
        True if all tests pass, False otherwise
    """
    test_results = []
    errors = []

    def test(name: str, func):
        """Helper to run a test and record results."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
        try:
            func()
            test_results.append((name, "PASS"))
            if verbose:
                print(f"✓ {name} PASSED")
            return True
        except Exception as e:
            test_results.append((name, "FAIL"))
            errors.append((name, e, traceback.format_exc()))
            if verbose:
                print(f"✗ {name} FAILED: {e}")
            return False

    # -----------------------------------------------------------------------
    # Test 1: Import all modules
    # -----------------------------------------------------------------------
    def test_imports():
        if verbose:
            print("Importing playground modules...")
        from playground import PlaygroundConfig, run_playground_experiment
        from playground.playground_probes import run_linear_probe, run_knn_probe
        from playground.playground_viz import plot_training_curves, plot_embeddings_2d
        from playground.playground_recipes import get_recipe, list_recipes
        if verbose:
            print("All imports successful!")

    test("Import all modules", test_imports)

    # -----------------------------------------------------------------------
    # Test 2: Create minimal config
    # -----------------------------------------------------------------------
    def test_config_creation():
        from playground import PlaygroundConfig
        if verbose:
            print("Creating minimal config...")
        config = PlaygroundConfig(
            dataset="mnist",
            backbone="mlp",
            icon_mode="simclr_like",
            epochs=1,
            batch_size=32,
            embedding_dim=32,
            num_workers=0,  # No multiprocessing
            output_dir="self_check_output",
        )
        if verbose:
            print(f"Config created: {config.run_name}")
        return config

    config = None
    if test("Create minimal config", test_config_creation):
        config = test_config_creation()

    # -----------------------------------------------------------------------
    # Test 3: Run experiment
    # -----------------------------------------------------------------------
    results = None

    def test_run_experiment():
        nonlocal results
        from playground import run_playground_experiment
        if verbose:
            print("Running 1-epoch experiment on MNIST (CPU)...")
            print("This may take 1-2 minutes...")

        results = run_playground_experiment(
            config,
            verbose=False,
            save_checkpoints=False,
            gpu=False,  # Force CPU
        )

        if verbose:
            print(f"Experiment completed!")
            print(f"  Embeddings shape: {results['embeddings'].shape}")
            print(f"  Labels shape: {results['labels'].shape}")
            print(f"  Epochs logged: {len(results['logs']['train_losses'])}")

        # Validate results
        assert results["embeddings"].shape[0] > 0, "Embeddings should not be empty"
        assert results["labels"].shape[0] > 0, "Labels should not be empty"
        assert len(results["logs"]["train_losses"]) >= 1, "Should have at least 1 epoch logged"

        return results

    test("Run 1-epoch experiment", test_run_experiment)

    # -----------------------------------------------------------------------
    # Test 4: Linear probe
    # -----------------------------------------------------------------------
    def test_linear_probe():
        from playground.playground_probes import run_linear_probe
        if verbose:
            print("Running linear probe on embeddings...")

        probe_results = run_linear_probe(
            results["embeddings"],
            results["labels"],
            test_size=0.2,
            verbose=False,
        )

        if verbose:
            print(f"  Test accuracy: {probe_results['test_accuracy']:.2%}")

        assert 0.0 <= probe_results["test_accuracy"] <= 1.0, "Accuracy should be in [0, 1]"
        return probe_results

    if results is not None:
        test("Run linear probe", test_linear_probe)

    # -----------------------------------------------------------------------
    # Test 5: kNN probe
    # -----------------------------------------------------------------------
    def test_knn_probe():
        from playground.playground_probes import run_knn_probe
        if verbose:
            print("Running kNN probe (k=5)...")

        knn_results = run_knn_probe(
            results["embeddings"],
            results["labels"],
            k=5,
            test_size=0.2,
            verbose=False,
        )

        if verbose:
            print(f"  Test accuracy: {knn_results['test_accuracy']:.2%}")

        assert 0.0 <= knn_results["test_accuracy"] <= 1.0, "Accuracy should be in [0, 1]"
        return knn_results

    if results is not None:
        test("Run kNN probe", test_knn_probe)

    # -----------------------------------------------------------------------
    # Test 6: Visualizations
    # -----------------------------------------------------------------------
    def test_visualizations():
        from playground.playground_viz import plot_training_curves, plot_embeddings_2d
        if verbose:
            print("Generating visualizations (in-memory)...")

        # Training curves
        fig1 = plot_training_curves(results["logs"], show=False)
        assert fig1 is not None, "Training curves plot failed"
        if verbose:
            print("  ✓ Training curves generated")

        # PCA embedding plot
        fig2 = plot_embeddings_2d(
            results["embeddings"],
            results["labels"],
            method="pca",
            show=False,
        )
        assert fig2 is not None, "Embedding plot failed"
        if verbose:
            print("  ✓ PCA embedding plot generated")

    if results is not None:
        test("Generate visualizations", test_visualizations)

    # -----------------------------------------------------------------------
    # Test 7: Recipe system
    # -----------------------------------------------------------------------
    def test_recipes():
        from playground.playground_recipes import list_recipes, get_recipe
        if verbose:
            print("Testing recipe system...")

        # List recipes
        recipes = list_recipes()
        assert len(recipes) > 0, "Should have at least one recipe"
        if verbose:
            print(f"  Found {len(recipes)} recipes: {', '.join(recipes)}")

        # Get a recipe
        configs = get_recipe("simclr_vs_sne", dataset="mnist", epochs=1)
        assert len(configs) == 2, "simclr_vs_sne should return 2 configs"
        if verbose:
            print(f"  ✓ Recipe 'simclr_vs_sne' loaded with {len(configs)} configs")

    test("Test recipe system", test_recipes)

    # -----------------------------------------------------------------------
    # Test 8: Config validation
    # -----------------------------------------------------------------------
    def test_config_validation():
        from playground import PlaygroundConfig
        if verbose:
            print("Testing config validation...")

        # Should raise ValueError for invalid dataset
        try:
            cfg = PlaygroundConfig(dataset="invalid_dataset")
            cfg.to_icon_config()
            raise AssertionError("Should have raised ValueError for invalid dataset")
        except ValueError:
            if verbose:
                print("  ✓ Invalid dataset correctly rejected")

        # Should raise ValueError for invalid icon_mode
        try:
            cfg = PlaygroundConfig(icon_mode="invalid_mode")
            cfg.to_icon_config()
            raise AssertionError("Should have raised ValueError for invalid icon_mode")
        except ValueError:
            if verbose:
                print("  ✓ Invalid icon_mode correctly rejected")

        # Should raise ValueError for negative epochs
        try:
            cfg = PlaygroundConfig(epochs=-1)
            cfg.to_icon_config()
            raise AssertionError("Should have raised ValueError for negative epochs")
        except ValueError:
            if verbose:
                print("  ✓ Negative epochs correctly rejected")

    test("Test config validation", test_config_validation)

    # -----------------------------------------------------------------------
    # Print Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SELF-CHECK SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in test_results if status == "PASS")
    failed = sum(1 for _, status in test_results if status == "FAIL")

    for test_name, status in test_results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {test_name}: {status}")

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_results)} tests")

    if failed > 0:
        print("\nFailed tests:")
        for name, error, tb in errors:
            print(f"\n  {name}:")
            print(f"    Error: {error}")
            if verbose:
                print(f"    Traceback:\n{tb}")

        print("\n✗ SELF-CHECK FAILED")
        print("\nTroubleshooting:")
        print("  1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("  2. Verify PyTorch is working: python -c 'import torch; print(torch.__version__)'")
        print("  3. Try reducing batch_size or using CPU-only mode")
        print("  4. See README_PLAYGROUND.md for more help")
        return False
    else:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThe I-Con Playground is ready to use.")
        print("Try running: python -m playground.playground_cli --help")
        return True

    # Clean up
    import shutil
    try:
        if Path("self_check_output").exists():
            shutil.rmtree("self_check_output")
            if verbose:
                print("\nCleaned up test outputs")
    except Exception as e:
        if verbose:
            print(f"\nWarning: Could not clean up test outputs: {e}")


def main():
    """CLI entrypoint for self-check."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run I-Con Playground self-check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════╗
║          I-CON PLAYGROUND SELF-CHECK                      ║
║   Running comprehensive tests...                          ║
╚═══════════════════════════════════════════════════════════╝
    """)

    success = run_self_check(verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
