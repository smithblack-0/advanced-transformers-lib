"""Tests for LoadBalanceLoss custom autograd Function.

Invariants verified:
- Forward loss formula: sum_l |f_l - 1/L|
- Forward output is a scalar
- Loss is zero when routing is perfectly balanced
- Backward writes L_grad * sign(f_l - 1/L) to expert_bias
- routing_freqs receives no gradient
- grad_b scales proportionally with L_grad (training loop loss weighting works correctly)
"""

import pytest
import torch

from src.shram.model.attention.load_balance_loss import LoadBalanceLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_loss(
    routing_freqs: torch.Tensor,
    bias_requires_grad: bool = True,
    freqs_requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct expert_bias and apply LoadBalanceLoss.apply, returning (loss, bias)."""
    expert_bias = torch.zeros(routing_freqs.shape[0], requires_grad=bias_requires_grad)
    routing_freqs = routing_freqs.clone()
    if freqs_requires_grad:
        routing_freqs.requires_grad_(True)
    loss = LoadBalanceLoss.apply(expert_bias, routing_freqs)
    return loss, expert_bias


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestLoadBalanceLossForward:
    def test_loss_formula(self):
        """Forward must compute sum_l |f_l - 1/L|."""
        routing_freqs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        loss, _ = apply_loss(routing_freqs)

        L = routing_freqs.shape[0]
        expected = float(sum(abs(f - 1.0 / L) for f in routing_freqs.tolist()))
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_loss_is_scalar(self):
        """Forward must return a zero-dimensional tensor."""
        routing_freqs = torch.ones(4) / 4
        loss, _ = apply_loss(routing_freqs)
        assert loss.shape == ()

    def test_loss_is_zero_when_balanced(self):
        """Loss must be exactly zero when all heads have equal frequency 1/L."""
        L = 8
        routing_freqs = torch.ones(L) / L
        loss, _ = apply_loss(routing_freqs)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_loss_is_nonnegative(self):
        """Sum of absolute values is always non-negative."""
        routing_freqs = torch.tensor([0.6, 0.1, 0.2, 0.1])
        loss, _ = apply_loss(routing_freqs)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

class TestLoadBalanceLossBackward:
    def test_grad_b_formula(self):
        """Backward must write L_grad * sign(f_l - 1/L) to expert_bias.grad."""
        routing_freqs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        loss, expert_bias = apply_loss(routing_freqs)
        loss.backward()

        L = routing_freqs.shape[0]
        expected_grad = torch.sign(routing_freqs - 1.0 / L)
        assert torch.allclose(expert_bias.grad, expected_grad)

    def test_routing_freqs_receives_no_gradient(self):
        """routing_freqs must not receive a gradient — its origin is discrete TopK."""
        routing_freqs = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)
        expert_bias = torch.zeros(4, requires_grad=True)
        loss = LoadBalanceLoss.apply(expert_bias, routing_freqs)
        loss.backward()

        assert routing_freqs.grad is None

    def test_grad_b_scales_with_l_grad(self):
        """Multiplying the loss by a scalar must multiply grad_b by the same scalar.

        This is the 'behaves like a normal auxiliary loss under scaling' invariant:
        a training loop that applies loss_weight * load_balance_loss must see
        loss_weight * grad_b, not just grad_b.
        """
        routing_freqs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        L = routing_freqs.shape[0]

        # Unscaled backward
        b1 = torch.zeros(L, requires_grad=True)
        loss1 = LoadBalanceLoss.apply(b1, routing_freqs.clone())
        loss1.backward()
        grad_unscaled = b1.grad.clone()

        # Scaled by 3.0
        b2 = torch.zeros(L, requires_grad=True)
        loss2 = LoadBalanceLoss.apply(b2, routing_freqs.clone())
        (3.0 * loss2).backward()
        grad_scaled = b2.grad.clone()

        assert torch.allclose(3.0 * grad_unscaled, grad_scaled, atol=1e-6)

    def test_grad_direction_matches_imbalance(self):
        """sign(f_l - 1/L) points toward correction: overloaded heads get positive
        grad_b so the optimizer decreases their bias, reducing future selection."""
        routing_freqs = torch.tensor([0.6, 0.1, 0.2, 0.1])  # head 0 overloaded
        loss, expert_bias = apply_loss(routing_freqs)
        loss.backward()

        L = routing_freqs.shape[0]
        # Head 0 is overloaded: f_0 - 1/L > 0, so grad_b[0] should be positive.
        # AdamW will subtract a positive gradient from b[0], reducing head 0's bias,
        # which reduces its future selection probability.
        assert expert_bias.grad[0].item() > 0.0
        # Head 1 is underloaded: grad_b[1] should be negative.
        assert expert_bias.grad[1].item() < 0.0
