"""
Tests for divergence functions.

These tests verify:
1. Divergence functions compute correctly
2. Different divergences produce different values
3. Divergences handle log/non-log inputs correctly
4. Edge cases (uniform distributions, identical distributions)
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.divergences import (
    kl_divergence,
    reverse_kl_divergence,
    js_divergence,
    get_divergence_fn,
    compute_divergence,
    SUPPORTED_DIVERGENCES,
)


class TestDivergenceFunctions:
    """Test suite for divergence functions."""

    def test_divergence_list(self):
        """Test that supported divergences list is correct."""
        assert "kl" in SUPPORTED_DIVERGENCES
        assert "rkl" in SUPPORTED_DIVERGENCES
        assert "js" in SUPPORTED_DIVERGENCES
        assert len(SUPPORTED_DIVERGENCES) == 3

    def test_kl_with_identical_distributions(self):
        """Cross-entropy H(p,p) should equal entropy H(p) (non-zero but constant)."""
        p = torch.softmax(torch.randn(8, 8), dim=1)
        log_p = torch.log(p)

        ce1 = kl_divergence(p, log_p, log_q=True)
        ce2 = kl_divergence(p, log_p, log_q=True)

        # Should be identical (deterministic)
        assert abs(ce1.item() - ce2.item()) < 1e-6, \
            f"H(p,p) should be deterministic, got {ce1.item()} vs {ce2.item()}"

        # Should be positive (entropy is always positive)
        assert ce1.item() > 0, f"Cross-entropy should be positive, got {ce1.item()}"

    def test_rkl_with_identical_distributions(self):
        """RKL(p||p) should be close to zero."""
        p = torch.softmax(torch.randn(8, 8), dim=1)
        log_p = torch.log(p)

        rkl = reverse_kl_divergence(p, log_p, log_q=True)
        assert rkl.item() < 1e-5, f"RKL(p||p) should be ~0, got {rkl.item()}"

    def test_js_with_identical_distributions(self):
        """JS(p||p) should be close to zero."""
        p = torch.softmax(torch.randn(8, 8), dim=1)
        log_p = torch.log(p)

        js = js_divergence(p, log_p, log_q=True)
        assert js.item() < 1e-5, f"JS(p||p) should be ~0, got {js.item()}"

    def test_kl_vs_rkl_different(self):
        """KL(p||q) != KL(q||p) for different distributions."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(8, 8), dim=1)
        q_logits = torch.randn(8, 8)
        q = torch.softmax(q_logits, dim=1)
        log_q = torch.log(q)

        kl = kl_divergence(p, log_q, log_q=True)
        rkl = reverse_kl_divergence(p, log_q, log_q=True)

        # They should be different
        assert abs(kl.item() - rkl.item()) > 0.01, \
            f"KL and RKL should differ, got KL={kl.item()}, RKL={rkl.item()}"

    def test_kl_vs_js_different(self):
        """KL(p||q) != JS(p||q) for different distributions."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(8, 8), dim=1)
        q_logits = torch.randn(8, 8)
        q = torch.softmax(q_logits, dim=1)
        log_q = torch.log(q)

        kl = kl_divergence(p, log_q, log_q=True)
        js = js_divergence(p, log_q, log_q=True)

        # They should be different
        assert abs(kl.item() - js.item()) > 0.01, \
            f"KL and JS should differ, got KL={kl.item()}, JS={js.item()}"

    def test_js_symmetry(self):
        """JS(p||q) should be approximately equal to JS(q||p)."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(8, 8), dim=1)
        q = torch.softmax(torch.randn(8, 8), dim=1)
        log_p = torch.log(p)
        log_q = torch.log(q)

        js_pq = js_divergence(p, log_q, log_q=True)
        js_qp = js_divergence(q, log_p, log_q=True)

        # Should be symmetric (within numerical precision)
        assert abs(js_pq.item() - js_qp.item()) < 1e-4, \
            f"JS should be symmetric, got JS(p||q)={js_pq.item()}, JS(q||p)={js_qp.item()}"

    def test_js_bounded(self):
        """JS divergence should be bounded in [0, log(2)]."""
        torch.manual_seed(42)
        # Create very different distributions
        p = torch.zeros(8, 8) + 1e-10
        p[0, 0] = 1.0
        p = p / p.sum(dim=1, keepdim=True)

        q = torch.zeros(8, 8) + 1e-10
        q[0, 7] = 1.0
        q = q / q.sum(dim=1, keepdim=True)
        log_q = torch.log(q)

        js = js_divergence(p, log_q, log_q=True)

        # JS should be in [0, log(2)] ≈ [0, 0.693]
        assert 0 <= js.item() <= 0.7, \
            f"JS should be in [0, log(2)], got {js.item()}"

    def test_log_vs_nonlog_input(self):
        """Test that log_q flag works correctly."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(4, 4), dim=1)
        q = torch.softmax(torch.randn(4, 4), dim=1)
        log_q = torch.log(q)

        # KL with log input
        kl_log = kl_divergence(p, log_q, log_q=True)

        # KL with non-log input
        kl_nonlog = kl_divergence(p, q, log_q=False)

        # Should be very close
        assert abs(kl_log.item() - kl_nonlog.item()) < 1e-5, \
            f"KL with log/nonlog should match, got {kl_log.item()} vs {kl_nonlog.item()}"

    def test_get_divergence_fn(self):
        """Test divergence function getter."""
        kl_fn = get_divergence_fn("kl")
        rkl_fn = get_divergence_fn("rkl")
        js_fn = get_divergence_fn("js")

        assert kl_fn == kl_divergence
        assert rkl_fn == reverse_kl_divergence
        assert js_fn == js_divergence

    def test_get_divergence_fn_case_insensitive(self):
        """Test that divergence names are case-insensitive."""
        kl_fn1 = get_divergence_fn("kl")
        kl_fn2 = get_divergence_fn("KL")
        kl_fn3 = get_divergence_fn("Kl")

        assert kl_fn1 == kl_fn2 == kl_fn3

    def test_get_divergence_fn_invalid(self):
        """Test that invalid divergence names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported divergence"):
            get_divergence_fn("invalid")

    def test_compute_divergence_wrapper(self):
        """Test the compute_divergence wrapper function."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(4, 4), dim=1)
        log_q = torch.log(torch.softmax(torch.randn(4, 4), dim=1))

        kl = compute_divergence(p, log_q, divergence="kl", log_q=True)
        rkl = compute_divergence(p, log_q, divergence="rkl", log_q=True)
        js = compute_divergence(p, log_q, divergence="js", log_q=True)

        # All should return scalars
        assert kl.numel() == 1
        assert rkl.numel() == 1
        assert js.numel() == 1

        # All should be non-negative
        assert kl.item() >= 0
        assert rkl.item() >= 0
        assert js.item() >= 0

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        p = torch.randn(4, 4)
        q = torch.randn(8, 8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            kl_divergence(p, q, log_q=True)

        with pytest.raises(ValueError, match="Shape mismatch"):
            reverse_kl_divergence(p, q, log_q=True)

        with pytest.raises(ValueError, match="Shape mismatch"):
            js_divergence(p, q, log_q=True)

    def test_gradients_flow(self):
        """Test that gradients flow through all divergences."""
        # Test KL
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(4, 4), dim=1)
        q_logits = torch.randn(4, 4, requires_grad=True)
        log_q = torch.log_softmax(q_logits, dim=1)

        kl = kl_divergence(p, log_q, log_q=True)
        kl.backward()
        assert q_logits.grad is not None
        assert not torch.all(q_logits.grad == 0), "KL gradients should be non-zero"

        # Test RKL (new computation graph)
        torch.manual_seed(43)
        p2 = torch.softmax(torch.randn(4, 4), dim=1)
        q_logits2 = torch.randn(4, 4, requires_grad=True)
        log_q2 = torch.log_softmax(q_logits2, dim=1)

        rkl = reverse_kl_divergence(p2, log_q2, log_q=True)
        rkl.backward()
        assert q_logits2.grad is not None
        assert not torch.all(q_logits2.grad == 0), "RKL gradients should be non-zero"

        # Test JS (new computation graph)
        torch.manual_seed(44)
        p3 = torch.softmax(torch.randn(4, 4), dim=1)
        q_logits3 = torch.randn(4, 4, requires_grad=True)
        log_q3 = torch.log_softmax(q_logits3, dim=1)

        js = js_divergence(p3, log_q3, log_q=True)
        js.backward()
        assert q_logits3.grad is not None
        assert not torch.all(q_logits3.grad == 0), "JS gradients should be non-zero"

    def test_uniform_distributions(self):
        """Test divergences with uniform distributions."""
        # Both uniform
        p = torch.ones(4, 4) / 4
        q = torch.ones(4, 4) / 4
        log_q = torch.log(q)

        kl = kl_divergence(p, log_q, log_q=True)
        js = js_divergence(p, log_q, log_q=True)

        # KL computes cross-entropy, which for uniform is log(4) = 1.386
        # JS for identical distributions should be close to zero
        assert js.item() < 1e-5, f"JS(p||p) should be ~0 for identical distributions, got {js.item()}"

        # Cross-entropy should be positive and equal to entropy of uniform
        expected_entropy = -torch.sum((p[0] * torch.log(p[0])))  # -sum(1/4 * log(1/4)) = log(4)
        assert abs(kl.item() - expected_entropy.item()) < 1e-4, \
            f"Cross-entropy should equal entropy of uniform distribution"

    def test_batch_averaging(self):
        """Test that divergences properly average over batch dimension."""
        torch.manual_seed(42)
        # Batch of 8
        p = torch.softmax(torch.randn(8, 8), dim=1)
        log_q = torch.log(torch.softmax(torch.randn(8, 8), dim=1))

        kl = kl_divergence(p, log_q, log_q=True)

        # Result should be scalar (averaged)
        assert kl.shape == torch.Size([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
