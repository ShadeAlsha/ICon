"""
Debug Script: Why do simclr_like and sne_like produce identical results?

This script traces through the loss computation to find where
the differences should appear and where they might be lost.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.playground_config import PlaygroundConfig
from model.kernel_model import Model


def debug_loss_computation():
    """Trace through the loss computation for both modes."""

    print("=" * 80)
    print("DEBUG: Comparing simclr_like vs sne_like Loss Computation")
    print("=" * 80)

    # Create configs for both modes
    simclr_config = PlaygroundConfig(
        dataset="mnist",
        backbone="mlp",
        icon_mode="simclr_like",
        temperature=0.5,
        embedding_dim=32,
    )

    sne_config = PlaygroundConfig(
        dataset="mnist",
        backbone="mlp",
        icon_mode="sne_like",
        temperature=0.5,
        embedding_dim=32,
    )

    # Print the learned distribution configs
    print("\n[1] LEARNED DISTRIBUTION CONFIGURATIONS:")
    print("-" * 40)

    simclr_icon = simclr_config.to_icon_config()
    sne_icon = sne_config.to_icon_config()

    print(f"SimCLR learned distribution: {simclr_icon.learned_distribution}")
    print(f"  - metric: {simclr_icon.learned_distribution.distance_kernel.metric}")
    print(f"  - sigma: {simclr_icon.learned_distribution.sigma}")

    print(f"\nSNE learned distribution: {sne_icon.learned_distribution}")
    print(f"  - metric: {sne_icon.learned_distribution.distance_kernel.metric}")
    print(f"  - sigma: {sne_icon.learned_distribution.sigma}")

    # Create fake batch data
    torch.manual_seed(42)
    batch_size = 32
    input_dim = 28 * 28
    num_augmentations = 2

    # Create fake images (flattened MNIST-like)
    images = torch.randn(batch_size * num_augmentations, input_dim)

    # Create indices for augmentation pairs
    # indices [0, 1, 2, ..., 31, 0, 1, 2, ..., 31]
    indices = torch.cat([torch.arange(batch_size), torch.arange(batch_size)])

    batch = {
        'image': images,
        'index': indices,
        'label': torch.randint(0, 10, (batch_size * num_augmentations,))
    }

    print("\n[2] COMPUTING EMBEDDINGS:")
    print("-" * 40)

    # Create models
    simclr_model = Model(simclr_icon)
    sne_model = Model(sne_icon)

    # IMPORTANT: Use same weights for both models
    sne_model.mapper.load_state_dict(simclr_model.mapper.state_dict())
    sne_model.linear_probe.load_state_dict(simclr_model.linear_probe.state_dict())

    # Compute embeddings (should be identical since same weights)
    with torch.no_grad():
        simclr_out = simclr_model.mapper(batch)
        sne_out = sne_model.mapper(batch)

        simclr_emb = simclr_out['embedding']
        sne_emb = sne_out['embedding']

        print(f"SimCLR embeddings shape: {simclr_emb.shape}")
        print(f"SimCLR embeddings mean: {simclr_emb.mean():.4f}, std: {simclr_emb.std():.4f}")
        print(f"SimCLR embeddings norm: {torch.norm(simclr_emb, dim=1).mean():.4f}")

        print(f"\nSNE embeddings shape: {sne_emb.shape}")
        print(f"SNE embeddings mean: {sne_emb.mean():.4f}, std: {sne_emb.std():.4f}")
        print(f"SNE embeddings norm: {torch.norm(sne_emb, dim=1).mean():.4f}")

        # Check if embeddings are identical (they should be with same weights)
        emb_diff = (simclr_emb - sne_emb).abs().max()
        print(f"\nMax embedding difference: {emb_diff:.6f} (should be ~0)")

    print("\n[3] COMPUTING DISTANCES:")
    print("-" * 40)

    # Get the distance kernels
    simclr_dist_kernel = simclr_icon.learned_distribution.distance_kernel
    sne_dist_kernel = sne_icon.learned_distribution.distance_kernel

    with torch.no_grad():
        simclr_distances = simclr_dist_kernel(simclr_emb)
        sne_distances = sne_dist_kernel(sne_emb)

        print(f"SimCLR distances (cosine):")
        print(f"  - shape: {simclr_distances.shape}")
        print(f"  - mean: {simclr_distances.mean():.4f}")
        print(f"  - std: {simclr_distances.std():.4f}")
        print(f"  - min: {simclr_distances.min():.4f}, max: {simclr_distances.max():.4f}")

        print(f"\nSNE distances (euclidean):")
        print(f"  - shape: {sne_distances.shape}")
        print(f"  - mean: {sne_distances.mean():.4f}")
        print(f"  - std: {sne_distances.std():.4f}")
        print(f"  - min: {sne_distances.min():.4f}, max: {sne_distances.max():.4f}")

        # Check if distances are different
        dist_diff = (simclr_distances - sne_distances).abs()
        print(f"\nDistance difference:")
        print(f"  - mean: {dist_diff.mean():.4f}")
        print(f"  - max: {dist_diff.max():.4f}")

        if dist_diff.max() < 1e-5:
            print("\n  *** WARNING: Distances are nearly identical! ***")
        else:
            print("\n  *** GOOD: Distances are different as expected ***")

    print("\n[4] COMPUTING LEARNED DISTRIBUTIONS:")
    print("-" * 40)

    batch_with_simclr_emb = {**batch, 'embedding': simclr_emb}
    batch_with_sne_emb = {**batch, 'embedding': sne_emb}

    with torch.no_grad():
        simclr_learned = simclr_icon.learned_distribution(batch_with_simclr_emb)
        sne_learned = sne_icon.learned_distribution(batch_with_sne_emb)

        print(f"SimCLR learned distribution:")
        print(f"  - shape: {simclr_learned.shape}")
        print(f"  - mean: {simclr_learned.mean():.6f}")
        print(f"  - std: {simclr_learned.std():.6f}")
        print(f"  - entropy: {-(simclr_learned * torch.log(simclr_learned + 1e-10)).sum(dim=1).mean():.4f}")

        print(f"\nSNE learned distribution:")
        print(f"  - shape: {sne_learned.shape}")
        print(f"  - mean: {sne_learned.mean():.6f}")
        print(f"  - std: {sne_learned.std():.6f}")
        print(f"  - entropy: {-(sne_learned * torch.log(sne_learned + 1e-10)).sum(dim=1).mean():.4f}")

        # Check if distributions are different
        dist_diff = (simclr_learned - sne_learned).abs()
        print(f"\nLearned distribution difference:")
        print(f"  - mean: {dist_diff.mean():.6f}")
        print(f"  - max: {dist_diff.max():.6f}")

        if dist_diff.max() < 1e-5:
            print("\n  *** WARNING: Learned distributions are nearly identical! ***")
            print("  This is the BUG - different metrics should produce different distributions!")
        else:
            print("\n  *** GOOD: Learned distributions are different ***")

    print("\n[5] COMPUTING SUPERVISORY DISTRIBUTION:")
    print("-" * 40)

    with torch.no_grad():
        simclr_sup = simclr_icon.supervisory_distribution(batch)
        sne_sup = sne_icon.supervisory_distribution(batch)

        print(f"SimCLR supervisory distribution:")
        print(f"  - shape: {simclr_sup.shape}")
        print(f"  - non-zero per row: {(simclr_sup > 0).sum(dim=1).float().mean():.2f}")

        print(f"\nSNE supervisory distribution:")
        print(f"  - shape: {sne_sup.shape}")
        print(f"  - non-zero per row: {(sne_sup > 0).sum(dim=1).float().mean():.2f}")

        # Check if supervisory distributions are identical (they should be!)
        sup_diff = (simclr_sup - sne_sup).abs().max()
        print(f"\nSupervisory distribution difference: {sup_diff:.6f}")
        print("  (Should be 0 - both use Augmentation)")

    print("\n[6] COMPUTING LOSS:")
    print("-" * 40)

    # Use the divergence function
    from playground.divergences import kl_divergence

    with torch.no_grad():
        simclr_loss = kl_divergence(simclr_sup, simclr_learned, log_q=False)
        sne_loss = kl_divergence(sne_sup, sne_learned, log_q=False)

        print(f"SimCLR KL loss: {simclr_loss:.6f}")
        print(f"SNE KL loss: {sne_loss:.6f}")
        print(f"Loss difference: {abs(simclr_loss - sne_loss):.6f}")

        if abs(simclr_loss - sne_loss) < 1e-5:
            print("\n  *** CRITICAL BUG: Losses are nearly identical! ***")
        else:
            print("\n  *** GOOD: Losses are different ***")

    print("\n[7] GRADIENT CHECK:")
    print("-" * 40)

    # Now check gradients
    simclr_model.zero_grad()
    sne_model.zero_grad()

    # Forward pass
    simclr_results = simclr_model._compute_loss(batch)
    sne_results = sne_model._compute_loss(batch)

    simclr_total_loss = sum(simclr_results['losses'].values())
    sne_total_loss = sum(sne_results['losses'].values())

    print(f"SimCLR total loss: {simclr_total_loss:.6f}")
    print(f"SNE total loss: {sne_total_loss:.6f}")

    simclr_total_loss.backward()
    sne_total_loss.backward()

    # Compare gradients
    simclr_grad_norm = sum(p.grad.norm().item() for p in simclr_model.mapper.parameters() if p.grad is not None)
    sne_grad_norm = sum(p.grad.norm().item() for p in sne_model.mapper.parameters() if p.grad is not None)

    print(f"\nSimCLR gradient norm: {simclr_grad_norm:.6f}")
    print(f"SNE gradient norm: {sne_grad_norm:.6f}")

    if abs(simclr_grad_norm - sne_grad_norm) < 1e-4:
        print("\n  *** CRITICAL BUG: Gradients are nearly identical! ***")
    else:
        print("\n  *** GOOD: Gradients are different ***")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    debug_loss_computation()
