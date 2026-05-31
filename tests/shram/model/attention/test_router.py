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

import math

import pytest
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
        embedding_width=64,
        num_mosrah_heads=8,
        num_selected_heads=4,
        head_dim=16,
        num_sliding_window_heads=4,
        window_size=16,
        mlp_width=128,
        num_decoder_layers=2,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def _reference_routing_freqs(
    selected_heads: torch.Tensor,
    num_heads: int,
    p: float,
) -> torch.Tensor:
    """Reference p-mean routing frequencies with no masking.

    Counts every token unconditionally, divides by N*K per batch item, then
    applies p-mean over the batch dimension. Trivially correct by inspection;
    used to verify the production masked path degenerates correctly when all
    tokens are live.

    Args:
        selected_heads: Head indices of shape (B, N, K).
        num_heads: Total number of heads L.
        p: p-mean exponent.

    Returns:
        p-mean aggregated routing frequencies of shape (L,).
    """
    B, N, K = selected_heads.shape
    per_item_freqs = torch.zeros(B, num_heads)
    for b in range(B):
        counts = torch.zeros(num_heads)
        for n in range(N):
            for k in range(K):
                counts[selected_heads[b, n, k]] += 1
        per_item_freqs[b] = counts / (N * K)
    return (per_item_freqs ** p).mean(dim=0) ** (1.0 / p)


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
        selected_heads, _, _, _ = router(x, active_mask, None)
        assert selected_heads.shape == (2, 8, config.num_selected_heads)

    def test_routing_probs_shape(self):
        """routing_probs must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _, _ = router(x, active_mask, None)
        assert routing_probs.shape == (2, 8, config.num_selected_heads)

    def test_load_balance_loss_is_scalar(self):
        """load_balance_loss must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, load_balance_loss, _ = router(x, active_mask, None)
        assert load_balance_loss.shape == ()

    def test_max_vio_is_scalar(self):
        """max_vio must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, _, max_vio = router(x, active_mask, None)
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
        _, routing_probs, _, _ = router(x, active_mask, None)
        token_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)

    def test_routing_probs_are_nonnegative(self):
        """Softmax outputs are non-negative; gathering and renormalizing preserves this."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _, _ = router(x, active_mask, None)
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
            selected_heads, routing_probs, _, _ = router(x, active_mask, None)

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
        selected_heads, _, _, _ = router(x, active_mask, None)
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
        selected_heads, _, _, _ = router(x, active_mask, None)
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
            selected_heads, _, _, _ = router(x, active_mask, None)

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
            selected_heads, routing_probs, _, _ = router(x, active_mask, None)

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
        _, _, load_balance_loss, _ = router(x, active_mask, None)
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
        _, _, load_balance_loss, _ = router(x, active_mask, None)
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
        _, _, _, max_vio = router(x, active_mask, None)
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
        x = torch.randn(B, N, config.embedding_width)
        _, _, loss_a, _ = router(x, active_mask, None)

        x_modified = x.clone()
        x_modified[0, 3] = torch.randn(config.embedding_width) * 100.0
        _, _, loss_b, _ = router(x_modified, active_mask, None)

        torch.testing.assert_close(loss_a, loss_b)

    def test_dead_tokens_do_not_affect_max_vio(self):
        """Changing a dead token's hidden state must not affect max_vio."""
        config = small_config()
        router = MoSRAHRouter(config)
        B, N = 2, 8
        active_mask = torch.ones(B, N, dtype=torch.bool)
        active_mask[1, 5] = False

        torch.manual_seed(17)
        x = torch.randn(B, N, config.embedding_width)
        _, _, _, vio_a = router(x, active_mask, None)

        x_modified = x.clone()
        x_modified[1, 5] = torch.randn(config.embedding_width) * 100.0
        _, _, _, vio_b = router(x_modified, active_mask, None)

        torch.testing.assert_close(vio_a, vio_b)

    def test_all_live_mask_gives_routing_freqs_equivalent_to_pre_masking_formula(self):
        """With all tokens live, max_vio must match the no-masking reference implementation.

        Uses _reference_routing_freqs — a trivially correct no-masking stub — as an
        independent oracle. Agreement confirms the masked production path degenerates
        correctly to the simple all-live case.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        B, N = 2, 8
        L = config.num_mosrah_heads
        active_mask = torch.ones(B, N, dtype=torch.bool)

        torch.manual_seed(13)
        x = torch.randn(B, N, config.embedding_width)

        with torch.no_grad():
            selected_heads, _, _, max_vio = router(x, active_mask, None)
            expected_freqs = _reference_routing_freqs(selected_heads, L, config.load_balance_p)
            expected_max_vio = MoSRAHRouter._compute_max_vio(expected_freqs, L)

        torch.testing.assert_close(max_vio, expected_max_vio)


# ---------------------------------------------------------------------------
# balance_capacity
# ---------------------------------------------------------------------------

class TestBalanceCapacity:
    """Tests for MoSRAHRouter.balance_capacity.

    Called directly as a static method with synthetic logits to verify
    each path independently of the full router forward pass.
    """

    def test_training_passthrough_when_n_below_capacity(self):
        """When N < capacity in training mode, logits are returned unchanged."""
        logits = torch.randn(2, 3, 4)
        result = MoSRAHRouter.balance_capacity(logits, None, capacity=8, min_choices=1, max_rounds=10)
        assert torch.equal(result, logits)

    def test_training_correct_tokens_masked(self):
        """Both capacity constraints hold when N > capacity.

        N=5, L=2, capacity=3: total capacity (6) exceeds N*min_choices (5), so
        a valid assignment exists. Verifies column and row bounds both hold.
        """
        logits = torch.tensor([[[5.0, 0.4], [3.0, 0.3], [1.0, 0.2], [4.0, 0.1], [2.0, 0.0]]])  # (1, 5, 2)
        result = MoSRAHRouter.balance_capacity(logits, None, capacity=3, min_choices=1, max_rounds=10)
        unmasked = result > -1e7
        assert (unmasked.sum(dim=-2) <= 3).all()   # column bound: at most 3 tokens per expert
        assert (unmasked.sum(dim=-1) >= 1).all()   # row bound: every token has at least 1 expert

    def test_training_surviving_count_equals_capacity(self):
        """With N > capacity, at most capacity logits per (batch, head) are unmasked,
        and every token retains at least min_choices experts."""
        B, N, L, capacity, min_choices = 2, 10, 4, 3, 1
        torch.manual_seed(0)
        logits = torch.randn(B, N, L)
        result = MoSRAHRouter.balance_capacity(logits, None, capacity=capacity, min_choices=min_choices, max_rounds=10)
        unmasked = result > -1e7
        assert (unmasked.sum(dim=-2) <= capacity).all()   # column bound
        assert (unmasked.sum(dim=-1) >= min_choices).all()  # row bound

    def test_training_no_masking_when_n_equals_capacity(self):
        """When N == capacity every token is within the limit — nothing is masked."""
        torch.manual_seed(2)
        logits = torch.randn(2, 4, 3)
        result = MoSRAHRouter.balance_capacity(logits, None, capacity=4, min_choices=1, max_rounds=10)
        assert torch.equal(result, logits)

    def test_inference_full_head_blocks_all_tokens(self):
        """When used_capacity equals capacity, all tokens for that head are masked.

        With L=2 and min_choices=1, the token still has the other head available,
        so the row bound is satisfied while the full head is correctly blocked.
        """
        B, N, L, capacity = 1, 4, 2, 4
        torch.manual_seed(3)
        logits = torch.randn(B, N, L)
        # head 0 fully used, head 1 empty
        used_capacity = torch.tensor([[capacity, 0]])
        result = MoSRAHRouter.balance_capacity(logits, used_capacity, capacity=capacity, min_choices=1, max_rounds=10)
        assert (result[0, :, 0] <= -1e7).all()

    def test_inference_correct_tokens_survive_per_head(self):
        """Both constraints hold for known logits with per-head used_capacity.

        Uses a (1, 3, 2) case: head 0 has remaining=2, head 1 has remaining=1.
        Total remaining (3) equals N*min_choices (3) — feasible but tight.
        """
        # B=1, N=3, L=2, capacity=4
        # head 0: used=2 → remaining=2; head 1: used=3 → remaining=1
        logits = torch.tensor([[[10.0, 9.0], [7.0, 6.0], [4.0, 3.0]]])
        used_capacity = torch.tensor([[2, 3]])
        result = MoSRAHRouter.balance_capacity(logits, used_capacity, capacity=4, min_choices=1, max_rounds=10)
        unmasked = result > -1e7
        remaining = (torch.tensor([[4, 4]]) - used_capacity).clamp(min=0)  # [[2, 1]]
        assert (unmasked.sum(dim=-2) <= remaining).all()   # column bound
        assert (unmasked.sum(dim=-1) >= 1).all()           # row bound

    def test_inference_n_less_than_capacity_empty_head_passes(self):
        """With N < capacity and an empty head, the single token must not be masked.

        This is the standard decode-step case: one new token, head has room.
        """
        B, N, L, capacity = 1, 1, 2, 8
        torch.manual_seed(4)
        logits = torch.randn(B, N, L)
        used_capacity = torch.zeros(B, L, dtype=torch.long)
        result = MoSRAHRouter.balance_capacity(logits, used_capacity, capacity=capacity, min_choices=1, max_rounds=10)
        assert (result > -1e7).all()

    def test_inference_n_less_than_capacity_full_head_blocks(self):
        """With N < capacity and a full head, that head's token must be masked.

        N=1, L=2, min_choices=1: head 0 is full so its token is masked, but
        head 1 is empty so the token still has one valid choice — row bound met.
        """
        B, N, L, capacity = 1, 1, 2, 8
        torch.manual_seed(5)
        logits = torch.randn(B, N, L)
        # head 0 full, head 1 empty
        used_capacity = torch.tensor([[capacity, 0]])
        result = MoSRAHRouter.balance_capacity(logits, used_capacity, capacity=capacity, min_choices=1, max_rounds=10)
        assert (result[0, :, 0] <= -1e7).all()
        assert (result[0, :, 1] > -1e7).all()

    def test_both_constraints_satisfied_training(self):
        """Both column and row bounds hold simultaneously under a tight capacity budget.

        N=32, L=4, capacity=25, min_choices=3. Total capacity (100) > N*K (96),
        so a valid assignment exists, but the budget is tight enough to require
        real enforcement.
        """
        B, N, L, capacity, min_choices = 1, 32, 4, 25, 3
        torch.manual_seed(7)
        logits = torch.randn(B, N, L)
        result = MoSRAHRouter.balance_capacity(logits, None, capacity=capacity, min_choices=min_choices, max_rounds=10)
        unmasked = result > -1e7
        assert (unmasked.sum(dim=-2) <= capacity).all(), "column bound violated"
        assert (unmasked.sum(dim=-1) >= min_choices).all(), "row bound violated"

    def test_both_constraints_satisfied_inference(self):
        """Both bounds hold simultaneously with mixed per-head remaining capacities."""
        B, N, L, capacity, min_choices = 1, 16, 4, 20, 2
        torch.manual_seed(8)
        logits = torch.randn(B, N, L)
        # Give each head a different used value; remaining = [18, 15, 12, 9]
        used_capacity = torch.tensor([[2, 5, 8, 11]])
        remaining = (capacity - used_capacity).clamp(min=0)  # (1, 4)
        result = MoSRAHRouter.balance_capacity(logits, used_capacity, capacity=capacity, min_choices=min_choices, max_rounds=10)
        unmasked = result > -1e7
        assert (unmasked.sum(dim=-2) <= remaining).all(), "column bound violated"
        assert (unmasked.sum(dim=-1) >= min_choices).all(), "row bound violated"

    def test_non_convergence_raises(self):
        """Infeasible config (total capacity < N * K) must raise RuntimeError in eager."""
        # L=4, capacity=2 → total=8; N=8, min_choices=3 → demand=24. Infeasible.
        B, N, L, capacity, min_choices = 1, 8, 4, 2, 3
        logits = torch.randn(B, N, L)
        with pytest.raises(RuntimeError):
            MoSRAHRouter.balance_capacity(logits, None, capacity=capacity, min_choices=min_choices, max_rounds=10)


# ---------------------------------------------------------------------------
# get_threshold
# ---------------------------------------------------------------------------

# Alias the static method so tests call the production code directly without
# being coupled to the class name in every assertion.
get_threshold = MoSRAHRouter.get_threshold


def make_tensor(*shape):
    """Deterministic tensor for reproducible tests."""
    return torch.arange(math.prod(shape), dtype=torch.float).reshape(shape)


class TestGetThresholdContract:
    """
    Tests verify the contract: value >= threshold iff value ranks within top n.
    Shape, device, and sentinel behaviour are verified separately.
    """

    def test_int_n_threshold_separates_top_n_dim_last(self):
        """For every position, exactly n values must be >= threshold per row."""
        t = torch.randn(3, 8)
        for n in range(1, 8):
            threshold = get_threshold(t, dim=-1, n=n)
            count = (t >= threshold).sum(dim=-1)
            assert (count == n).all(), \
                f"n={n}: expected {n} values >= threshold per row, got {count}"

    def test_int_n_threshold_separates_top_n_dim_second_last(self):
        """For every column, exactly n values must be >= threshold."""
        t = torch.randn(2, 8, 4)
        for n in range(1, 8):
            threshold = get_threshold(t, dim=-2, n=n)
            count = (t >= threshold).sum(dim=-2)
            assert (count == n).all(), \
                f"n={n}: expected {n} values >= threshold per column"

    def test_tensor_n_threshold_separates_top_n(self):
        """Each row gets its own rank; the contract holds per row."""
        t = torch.randn(4, 8)
        n = torch.randint(1, 8, (4,))
        threshold = get_threshold(t, dim=-1, n=n)
        for i in range(t.shape[0]):
            count = (t[i] >= threshold[i, 0]).sum()
            assert count.item() == n[i].item(), \
                f"row {i}: expected {n[i]} values >= threshold, got {count}"

    def test_tensor_n_column_dim(self):
        """Tensor-n contract holds for the second-to-last dimension."""
        t = torch.randn(2, 8, 4)
        n = torch.randint(1, 8, (2, 4))
        threshold = get_threshold(t, dim=-2, n=n)
        for b in range(2):
            for j in range(4):
                count = (t[b, :, j] >= threshold[b, 0, j]).sum()
                assert count.item() == n[b, j].item()


class TestGetThresholdSentinels:

    def test_int_n_zero_returns_inf(self):
        """n=0 must return +inf so that no value passes the >= threshold check."""
        t = torch.randn(4, 8)
        result = get_threshold(t, dim=-1, n=0)
        assert result.isinf().all()
        assert (result > 0).all()

    def test_int_n_overflow_returns_neg_inf(self):
        """n > dim_length must return -inf so that every value passes."""
        t = torch.randn(4, 8)
        result = get_threshold(t, dim=-1, n=9)   # dim_length == 8
        assert result.isinf().all()
        assert (result < 0).all()

    def test_tensor_n_zero_positions_return_inf(self):
        """Tensor n=0 per row must return +inf."""
        t = torch.randn(4, 8)
        n = torch.zeros(4, dtype=torch.long)
        result = get_threshold(t, dim=-1, n=n)
        assert result.isinf().all()
        assert (result > 0).all()

    def test_tensor_n_overflow_positions_return_neg_inf(self):
        """Tensor n > dim_length per row must return -inf."""
        t = torch.randn(4, 8)
        n = torch.full((4,), 9, dtype=torch.long)   # dim_length == 8
        result = get_threshold(t, dim=-1, n=n)
        assert result.isinf().all()
        assert (result < 0).all()

    def test_tensor_n_mixed_sentinels_and_valid(self):
        """A mix of n=0, valid n, and n>dim_length must produce the correct sentinels."""
        t = torch.randn(3, 8)
        # row 0: n=0 (+inf), row 1: n=4 (valid), row 2: n=9 (-inf)
        n = torch.tensor([0, 4, 9])
        result = get_threshold(t, dim=-1, n=n)
        assert math.isinf(result[0, 0].item()) and result[0, 0] > 0
        assert not result[1, 0].isinf()
        assert math.isinf(result[2, 0].item()) and result[2, 0] < 0


class TestGetThresholdShape:

    def test_keepdim_int_n_dim_last(self):
        """Result must have keepdim shape when reducing along the last dimension."""
        t = torch.randn(2, 5, 8)
        result = get_threshold(t, dim=-1, n=3)
        assert tuple(result.shape) == (2, 5, 1)

    def test_keepdim_int_n_dim_second_last(self):
        """Result must have keepdim shape when reducing along the second-to-last dim."""
        t = torch.randn(2, 8, 4)
        result = get_threshold(t, dim=-2, n=3)
        assert tuple(result.shape) == (2, 1, 4)

    def test_keepdim_tensor_n_dim_last(self):
        """Tensor-n result must have keepdim shape along the last dimension."""
        t = torch.randn(2, 5, 8)
        n = torch.randint(1, 8, (2, 5))
        result = get_threshold(t, dim=-1, n=n)
        assert tuple(result.shape) == (2, 5, 1)

    def test_keepdim_tensor_n_dim_second_last(self):
        """Tensor-n result must have keepdim shape along the second-to-last dim."""
        t = torch.randn(2, 8, 4)
        n = torch.randint(1, 8, (2, 4))
        result = get_threshold(t, dim=-2, n=n)
        assert tuple(result.shape) == (2, 1, 4)

    def test_dtype_and_device_preserved(self):
        """Output dtype and device must match the input tensor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.randn(4, 8, device=device, dtype=torch.float32)
        result = get_threshold(t, dim=-1, n=3)
        assert result.device.type == device
        assert result.dtype == torch.float32


class TestGetThresholdIntTensorAgreement:
    """Int and tensor paths must agree on valid n values."""

    def test_paths_agree_dim_last(self):
        """int n and tensor n must produce identical thresholds for every valid rank."""
        t = torch.randn(3, 8)
        for n_val in range(1, 9):
            int_result    = get_threshold(t, dim=-1, n=n_val)
            tensor_result = get_threshold(t, dim=-1, n=torch.full((3,), n_val))
            assert torch.allclose(int_result, tensor_result), \
                f"paths disagree at n={n_val}"

    def test_paths_agree_dim_second_last(self):
        """int n and tensor n must agree along the second-to-last dimension."""
        t = torch.randn(2, 8, 4)
        for n_val in range(1, 9):
            int_result    = get_threshold(t, dim=-2, n=n_val)
            tensor_result = get_threshold(t, dim=-2,
                                          n=torch.full((2, 4), n_val))
            assert torch.allclose(int_result, tensor_result), \
                f"paths disagree at n={n_val}"
