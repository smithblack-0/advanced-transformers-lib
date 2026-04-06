"""Tests for MoSRAHRouter.

Invariants verified:
- Output shapes: selected_heads (B, N, K), routing_probs (B, N, K), loss scalar
- routing_probs sum to 1 per token and are non-negative
- routing_probs are computed from unbiased scores, not biased scores
- selected_heads are valid indices in [0, L-1] and are distinct per token
- expert_bias influences selection but not routing_probs
- expert_bias receives a gradient through load_balance_loss
- routing_projection has no bias parameter
"""

import pytest
import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.router import MoSRAHRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    """Small config valid for router tests. num_selected_heads < num_mosrah_heads
    so TopK is genuinely sparse."""
    defaults = dict(
        hidden_size=64,
        num_mosrah_heads=8,
        num_selected_heads=4,
        head_dim=16,
        num_sliding_window_heads=4,
        window_size=16,
        intermediate_size=128,
        num_hidden_layers=2,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_selected_heads_shape(self):
        """selected_heads must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        selected_heads, _, _ = router(x)
        assert selected_heads.shape == (2, 8, config.num_selected_heads)

    def test_routing_probs_shape(self):
        """routing_probs must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        _, routing_probs, _ = router(x)
        assert routing_probs.shape == (2, 8, config.num_selected_heads)

    def test_load_balance_loss_is_scalar(self):
        """load_balance_loss must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        _, _, load_balance_loss = router(x)
        assert load_balance_loss.shape == ()


# ---------------------------------------------------------------------------
# Routing probabilities
# ---------------------------------------------------------------------------

class TestRoutingProbabilities:
    def test_routing_probs_sum_to_one(self):
        """Each token's routing_probs must sum to 1 — they are a probability distribution
        over the K selected heads."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(3, 10, 64)
        _, routing_probs, _ = router(x)
        token_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)

    def test_routing_probs_are_nonnegative(self):
        """Softmax outputs are non-negative; gathering and renormalizing preserves this."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        _, routing_probs, _ = router(x)
        assert (routing_probs >= 0).all()

    def test_routing_probs_use_unbiased_scores(self):
        """routing_probs must match manually recomputed P from unbiased routing_scores.

        This is the critical invariant that separates selection (which uses expert_bias)
        from reduction weighting (which must not). Verified by recomputing P independently
        from the unbiased softmax at the returned selected_heads indices.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            selected_heads, routing_probs, _ = router(x)

            # Recompute unbiased scores without going through the router's full forward
            logits = router.routing_projection(x)
            unbiased_scores = torch.softmax(logits, dim=-1)
            gathered = unbiased_scores.gather(dim=-1, index=selected_heads)
            expected_routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)

        assert torch.allclose(routing_probs, expected_routing_probs, atol=1e-6)


# ---------------------------------------------------------------------------
# Selected heads
# ---------------------------------------------------------------------------

class TestSelectedHeads:
    def test_selected_heads_in_valid_range(self):
        """selected_heads values must be valid head indices in [0, L-1]."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        selected_heads, _, _ = router(x)
        assert (selected_heads >= 0).all()
        assert (selected_heads < config.num_mosrah_heads).all()

    def test_selected_heads_are_distinct_per_token(self):
        """TopK on distinct Softmax outputs must return K distinct head indices per token.

        Softmax produces strictly positive values for all L heads; ties are impossible
        in practice, so K selected indices must be distinct.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        selected_heads, _, _ = router(x)
        B, N, K = selected_heads.shape
        for b in range(B):
            for n in range(N):
                assert selected_heads[b, n].unique().shape[0] == K


# ---------------------------------------------------------------------------
# Bias influences selection only
# ---------------------------------------------------------------------------

class TestBiasInfluencesSelectionOnly:
    def test_large_bias_forces_head_selection(self):
        """A large expert_bias on head 0 must cause head 0 to appear in every token's
        selection — demonstrating that expert_bias drives selection."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(1, 6, 64)

        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            selected_heads, _, _ = router(x)

        # Head 0 must be selected for every token when its bias is enormous.
        assert (selected_heads == 0).any(dim=-1).all()

    def test_bias_does_not_affect_routing_probs_at_same_indices(self):
        """Given the same selected_heads, routing_probs must be identical regardless of
        expert_bias — because P is computed from unbiased routing_scores only.

        Verified by fixing selected_heads via a large bias, then recomputing P from
        unbiased scores at those indices and confirming it matches.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(1, 4, 64)

        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            selected_heads, routing_probs, _ = router(x)

            logits = router.routing_projection(x)
            unbiased = torch.softmax(logits, dim=-1)
            gathered = unbiased.gather(dim=-1, index=selected_heads)
            expected = gathered / gathered.sum(dim=-1, keepdim=True)

        assert torch.allclose(routing_probs, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:
    def test_expert_bias_receives_gradient(self):
        """expert_bias must accumulate a gradient after backward on load_balance_loss."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        _, _, load_balance_loss = router(x)
        load_balance_loss.backward()

        assert router.expert_bias.grad is not None
        assert router.expert_bias.grad.shape == (config.num_mosrah_heads,)

    def test_expert_bias_gradient_is_not_all_zero(self):
        """With an unbalanced router, expert_bias.grad must be non-zero — all-zero grad
        would mean the load balancing operator has no effect on training."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        _, _, load_balance_loss = router(x)
        load_balance_loss.backward()

        # At initialization with zero biases and random weights the routing will be
        # imperfectly balanced, so at least one head's gradient should be non-zero.
        assert router.expert_bias.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Architecture invariants
# ---------------------------------------------------------------------------

class TestArchitectureInvariants:
    def test_routing_projection_has_no_bias(self):
        """routing_projection must be bias-free (paper specifies xW_r with no bias term).
        expert_bias is the only bias-like parameter and has an entirely separate role."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.routing_projection.bias is None

    def test_expert_bias_shape(self):
        """expert_bias must have shape (num_mosrah_heads,) — one scalar per head."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.expert_bias.shape == (config.num_mosrah_heads,)

    def test_expert_bias_is_parameter(self):
        """expert_bias must be an nn.Parameter so the optimizer sees and updates it."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert isinstance(router.expert_bias, torch.nn.Parameter)
