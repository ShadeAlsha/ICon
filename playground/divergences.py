"""
Divergence Functions for I-Con Playground

Provides clean abstractions for different divergence measures between
the supervisory distribution p(j|i) and learned distribution q(j|i).

Supported divergences:
- KL(p||q): Forward Kullback-Leibler divergence (default, cross-entropy)
- RKL(q||p): Reverse Kullback-Leibler divergence
- JS(p||q): Jensen-Shannon divergence (symmetric)
"""

import torch
import torch.nn.functional as F
from typing import Literal


SUPPORTED_DIVERGENCES = ["kl", "rkl", "js"]


def kl_divergence(p: torch.Tensor, q: torch.Tensor, log_q: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute forward KL divergence: KL(p || q) = sum(p * log(p/q))

    This is equivalent to cross-entropy when p is normalized.
    Measures how much information is lost when q is used to approximate p.

    Args:
        p: Target/supervisory distribution (batch_size, batch_size)
        q: Learned distribution (batch_size, batch_size)
        log_q: If True, q is already in log-space
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value (averaged over batch)
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} != q {q.shape}")

    # Flatten and filter
    p_flat = p.flatten()
    q_flat = q.flatten()
    non_zero_mask = p_flat > eps

    p_filtered = p_flat[non_zero_mask]
    q_filtered = q_flat[non_zero_mask]

    # Compute cross-entropy (equivalent to KL when p is normalized)
    log_q_vals = q_filtered if log_q else torch.log(q_filtered.clamp(min=eps))
    cross_entropy = -torch.sum(p_filtered * log_q_vals)

    return cross_entropy / p.shape[0]


def reverse_kl_divergence(p: torch.Tensor, q: torch.Tensor, log_q: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute reverse KL divergence: KL(q || p) = sum(q * log(q/p))

    This is the "mode-seeking" divergence, in contrast to KL(p||q) which is "mean-seeking".
    Useful for avoiding over-smoothing in learned distributions.

    Args:
        p: Target/supervisory distribution (batch_size, batch_size)
        q: Learned distribution (batch_size, batch_size)
        log_q: If True, q is already in log-space
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value (averaged over batch)
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} != q {q.shape}")

    # Convert q from log space if needed
    if log_q:
        q_prob = torch.exp(q)
    else:
        q_prob = q

    # Clamp for stability
    p_clamped = p.clamp(min=eps)
    q_clamped = q_prob.clamp(min=eps)

    # Flatten and filter where q > 0
    q_flat = q_clamped.flatten()
    p_flat = p_clamped.flatten()
    non_zero_mask = q_flat > eps

    q_filtered = q_flat[non_zero_mask]
    p_filtered = p_flat[non_zero_mask]

    # KL(q||p) = sum(q * log(q/p))
    reverse_kl = torch.sum(q_filtered * (torch.log(q_filtered) - torch.log(p_filtered)))

    return reverse_kl / p.shape[0]


def js_divergence(p: torch.Tensor, q: torch.Tensor, log_q: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence: JS(p||q) = 0.5 * KL(p||M) + 0.5 * KL(q||M)
    where M = 0.5 * (p + q)

    This is a symmetric divergence, bounded in [0, log(2)].
    Provides a balanced alternative to KL divergences.

    Args:
        p: Target/supervisory distribution (batch_size, batch_size)
        q: Learned distribution (batch_size, batch_size)
        log_q: If True, q is already in log-space
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value (averaged over batch)
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} != q {q.shape}")

    # Convert q from log space if needed
    if log_q:
        q_prob = torch.exp(q)
    else:
        q_prob = q

    # Clamp for stability
    p_clamped = p.clamp(min=eps)
    q_clamped = q_prob.clamp(min=eps)

    # Compute mixture distribution M = 0.5 * (p + q)
    m = 0.5 * (p_clamped + q_clamped)

    # JS = 0.5 * KL(p||M) + 0.5 * KL(q||M)
    # Use PyTorch's built-in kl_div with reduction='batchmean'
    kl_p_m = F.kl_div(m.log(), p_clamped, reduction='batchmean')
    kl_q_m = F.kl_div(m.log(), q_clamped, reduction='batchmean')

    js = 0.5 * (kl_p_m + kl_q_m)

    return js


def get_divergence_fn(divergence: str):
    """
    Get divergence function by name.

    Args:
        divergence: One of 'kl', 'rkl', 'js'

    Returns:
        Divergence function with signature (p, q, log_q) -> scalar

    Raises:
        ValueError: If divergence name is not supported
    """
    divergence = divergence.lower()

    divergence_map = {
        'kl': kl_divergence,
        'rkl': reverse_kl_divergence,
        'js': js_divergence,
    }

    if divergence not in divergence_map:
        raise ValueError(
            f"Unsupported divergence: {divergence}. "
            f"Choose from: {SUPPORTED_DIVERGENCES}"
        )

    return divergence_map[divergence]


# For backward compatibility, keep a function that matches the old loss API
def compute_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    divergence: Literal["kl", "rkl", "js"] = "kl",
    log_q: bool = True,
) -> torch.Tensor:
    """
    Compute divergence between distributions p and q.

    Args:
        p: Target/supervisory distribution
        q: Learned distribution (in log space if log_q=True)
        divergence: Which divergence to use ('kl', 'rkl', 'js')
        log_q: Whether q is in log space

    Returns:
        Scalar divergence value
    """
    div_fn = get_divergence_fn(divergence)
    return div_fn(p, q, log_q=log_q)
