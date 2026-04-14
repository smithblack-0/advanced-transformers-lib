"""Tests for MoSRAHRouter.

Invariants verified:
- Output shapes: selected_heads (B, N, K), routing_probs (B, N, K), loss scalar,
  max_vio scalar
- routing_probs sum to 1 per token and are non-negative
- routing_probs are computed from unbiased scores, not biased scores
- selected_heads are valid indices in [0, L-1] and are distinct per token
- expert_bias influences selection but not routing_probs
- expert_bias receives a gradient through load_balance_loss
- routing_projection has no bias parameter
- max_vio is exactly 0 for perfectly uniform routing frequencies
- max_vio is exactly 1 when the most overloaded head receives double its fair share
- max_vio produces the correct value for a known intermediate routing imbalance
- max_vio is detached from the autograd graph
- dead outer tokens do not affect load_balance_loss
- dead outer tokens do not affect max_vio
- all-live active_mask gives routing frequencies equivalent to the pre-masking formula
"""

import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.router import MoSRAHRouter


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
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        selected_heads, _, _, _ = router(x, active_mask)
        assert selected_heads.shape == (2, 8, config.num_selected_heads)

    def test_routing_probs_shape(self):
        """routing_probs must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _, _ = router(x, active_mask)
        assert routing_probs.shape == (2, 8, config.num_selected_heads)

    def test_load_balance_loss_is_scalar(self):
        """load_balance_loss must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, load_balance_loss, _ = router(x, active_mask)
        assert load_balance_loss.shape == ()

    def test_max_vio_is_scalar(self):
        """max_vio must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, _, max_vio = router(x, active_mask)
        assert max_vio.shape == ()


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
        active_mask = torch.ones(3, 10, dtype=torch.bool)
        _, routing_probs, _, _ = router(x, active_mask)
        token_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)

    def test_routing_probs_are_nonnegative(self):
        """Softmax outputs are non-negative; gathering and renormalizing preserves this."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _, _ = router(x, active_mask)
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
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        with torch.no_grad():
            selected_heads, routing_probs, _, _ = router(x, active_mask)

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
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        selected_heads, _, _, _ = router(x, active_mask)
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
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        selected_heads, _, _, _ = router(x, active_mask)
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
        active_mask = torch.ones(1, 6, dtype=torch.bool)

        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            selected_heads, _, _, _ = router(x, active_mask)

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
        active_mask = torch.ones(1, 4, dtype=torch.bool)

        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            selected_heads, routing_probs, _, _ = router(x, active_mask)

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
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, load_balance_loss, _ = router(x, active_mask)
        load_balance_loss.backward()

        assert router.expert_bias.grad is not None
        assert router.expert_bias.grad.shape == (config.num_mosrah_heads,)

    def test_expert_bias_gradient_is_not_all_zero(self):
        """With an unbalanced router, expert_bias.grad must be non-zero — all-zero grad
        would mean the load balancing operator has no effect on training."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, load_balance_loss, _ = router(x, active_mask)
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


# ---------------------------------------------------------------------------
# MaxVio
# ---------------------------------------------------------------------------

class TestMaxVio:
    """Tests for the _compute_max_vio helper and the max_vio forward output.

    The helper is tested directly with synthetic routing_freqs tensors, bypassing
    TopK entirely. This avoids the tie-breaking ambiguity that arises when all
    routing scores are equal and makes the expected values exact and analytical.

    All three test cases use L=4 heads and analytically derived expected values.
    """

    def test_max_vio_zero_for_uniform_frequencies(self):
        """MaxVio must be exactly 0 when all heads receive equal routing frequency.

        With perfectly balanced routing (f_l = 1/L for all l), every term
        (f_l - 1/L) is zero, so L * max(f_l - 1/L) = 0.
        """
        L = 4
        routing_freqs = torch.full((L,), 1.0 / L)
        max_vio = MoSRAHRouter._compute_max_vio(routing_freqs, L)
        assert torch.isclose(max_vio, torch.tensor(0.0), atol=1e-6)

    def test_max_vio_one_for_double_fair_share(self):
        """MaxVio must be exactly 1 when one head receives double its fair share.

        With L=4, fair share is 0.25. Head 0 gets 0.5 (= 2/L); the remaining
        three heads share the rest equally. MaxVio = 4 * (0.5 - 0.25) = 1.0.
        """
        L = 4
        overloaded_freq = 2.0 / L                          # 0.5
        remainder = (1.0 - overloaded_freq) / (L - 1)     # 1/6 each
        routing_freqs = torch.tensor(
            [overloaded_freq] + [remainder] * (L - 1)
        )
        max_vio = MoSRAHRouter._compute_max_vio(routing_freqs, L)
        assert torch.isclose(max_vio, torch.tensor(1.0), atol=1e-6)

    def test_max_vio_intermediate_value(self):
        """MaxVio must equal 0.5 when one head receives 1.5× its fair share.

        With L=4, fair share is 0.25. Head 0 gets 0.375 (= 1.5/L); the remaining
        three heads share the rest equally. MaxVio = 4 * (0.375 - 0.25) = 0.5.
        """
        L = 4
        overloaded_freq = 1.5 / L                          # 0.375
        remainder = (1.0 - overloaded_freq) / (L - 1)     # 0.625 / 3
        routing_freqs = torch.tensor(
            [overloaded_freq] + [remainder] * (L - 1)
        )
        max_vio = MoSRAHRouter._compute_max_vio(routing_freqs, L)
        assert torch.isclose(max_vio, torch.tensor(0.5), atol=1e-6)

    def test_max_vio_is_detached(self):
        """max_vio must not be part of the autograd graph.

        MaxVio is a monitoring scalar. It must never contribute gradients to any
        parameter regardless of how the caller uses it.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, _, max_vio = router(x, active_mask)
        assert not max_vio.requires_grad


# ---------------------------------------------------------------------------
# Masked continuation behavior
# ---------------------------------------------------------------------------

class TestMaskedContinuationBehavior:
    def test_dead_tokens_do_not_affect_load_balance_loss(self):
        """Changing a dead token's hidden state must not affect load_balance_loss.

        Verified by marking a token dead, computing load_balance_loss, then replacing
        that token's hidden state with a drastically different value and confirming
        the loss is unchanged. The large multiplier ensures the dead token's routing
        selection almost certainly changes, so any leak would be detected.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        B, N = 2, 8
        active_mask = torch.ones(B, N, dtype=torch.bool)
        active_mask[0, 3] = False

        torch.manual_seed(11)
        x = torch.randn(B, N, config.hidden_size)
        _, _, loss_a, _ = router(x, active_mask)

        x_modified = x.clone()
        x_modified[0, 3] = torch.randn(config.hidden_size) * 100.0
        _, _, loss_b, _ = router(x_modified, active_mask)

        torch.testing.assert_close(loss_a, loss_b)

    def test_dead_tokens_do_not_affect_max_vio(self):
        """Changing a dead token's hidden state must not affect max_vio."""
        config = small_config()
        router = MoSRAHRouter(config)
        B, N = 2, 8
        active_mask = torch.ones(B, N, dtype=torch.bool)
        active_mask[1, 5] = False

        torch.manual_seed(17)
        x = torch.randn(B, N, config.hidden_size)
        _, _, _, vio_a = router(x, active_mask)

        x_modified = x.clone()
        x_modified[1, 5] = torch.randn(config.hidden_size) * 100.0
        _, _, _, vio_b = router(x_modified, active_mask)

        torch.testing.assert_close(vio_a, vio_b)

    def test_all_live_mask_gives_routing_freqs_equivalent_to_pre_masking_formula(self):
        """With all tokens live, routing_freqs must equal assignment.sum() / (B*N*K).

        Verified by independently computing the old-style (pre-masking) routing
        frequencies from the returned selected_heads and confirming max_vio matches.
        This ensures the all-live path through the masked formula is numerically
        equivalent to the original unmasked computation.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        B, N = 2, 8
        K = config.num_selected_heads
        L = config.num_mosrah_heads
        active_mask = torch.ones(B, N, dtype=torch.bool)

        torch.manual_seed(13)
        x = torch.randn(B, N, config.hidden_size)

        with torch.no_grad():
            selected_heads, _, _, max_vio = router(x, active_mask)

            assignment_mask = torch.zeros(B, N, L)
            assignment_mask.scatter_(-1, selected_heads, 1.0)
            expected_freqs = assignment_mask.sum(dim=(0, 1)) / (B * N * K)
            expected_max_vio = MoSRAHRouter._compute_max_vio(expected_freqs, L)

        torch.testing.assert_close(max_vio, expected_max_vio)
