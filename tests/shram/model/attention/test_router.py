"""Tests for MoSRAHRouter.

Invariants verified:
- Output shapes: selected_heads (B, N, K), routing_probs (B, N, K), loss scalar,
  max_vio scalar
- routing_probs sum to 1 per token and are non-negative
- routing_probs incorporate expert_bias via semantic_logits (logits * routing_scale + expert_bias.detach())
- selected_heads are valid indices in [0, L-1] and are distinct per token
- expert_bias influences both selection and routing_probs via the semantic gradient channel
- expert_bias receives a gradient through load_balance_loss
- routing_projection has no bias parameter
- routing_scale is a near-zero nn.Parameter that survives HuggingFace _init_weights
- task loss backward populates routing_projection.weight.grad, not expert_bias.grad
- load balance loss backward populates expert_bias.grad, not routing_projection.weight.grad
- assignment_probs are computed before balance_capacity, preventing -1e8 contamination
- max_vio is exactly 0 for perfectly uniform routing frequencies
- max_vio is exactly 1 when the most overloaded head receives double its fair share
- max_vio produces the correct value for a known intermediate routing imbalance
- max_vio is detached from the autograd graph
- dead outer tokens do not affect load_balance_loss
- dead outer tokens do not affect max_vio
- all-live active_mask gives routing frequencies equivalent to the pre-masking formula
- router_diagnostics separates routing decisions from routing feedback
- load_balance_loss has gradient; all other diagnostic scalars are detached
- bias_std is zero when expert_bias is zero
- logit_std equals raw_logit_std when expert_bias is zero
- bias_alignment is negative when expert_bias opposes routing logit direction
- bias_alignment is positive when expert_bias reinforces routing logit direction
- compiled and eager router diagnostics are numerically identical
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
        selected_heads, _, _ = router(x, active_mask, None)
        assert selected_heads.shape == (2, 8, config.num_selected_heads)

    def test_routing_probs_shape(self):
        """routing_probs must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask, None)
        assert routing_probs.shape == (2, 8, config.num_selected_heads)

    def test_load_balance_loss_is_scalar(self):
        """load_balance_loss must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        load_balance_loss = diagnostics["load_balance_loss"]
        assert load_balance_loss.shape == ()

    def test_max_vio_is_scalar(self):
        """max_vio must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        max_vio = diagnostics["max_vio"]
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
        _, routing_probs, _ = router(x, active_mask, None)
        token_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)

    def test_routing_probs_are_nonnegative(self):
        """Softmax outputs are non-negative; gathering and renormalizing preserves this."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask, None)
        assert (routing_probs >= 0).all()

    def test_routing_probs_use_semantic_logits(self):
        """routing_probs must match P recomputed from semantic_logits.

        Routing_probs are gathered from softmax(semantic_logits) = softmax(logits *
        routing_scale + expert_bias.detach()) at selected_heads indices and renormalized.
        Verified by recomputing independently using the same biased values.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        with torch.no_grad():
            selected_heads, routing_probs, _ = router(x, active_mask, None)

            # Recompute routing_probs through the semantic pathway
            logits = router.routing_projection(x) * router.routing_scale
            semantic_logits = logits + router.expert_bias
            semantic_scores = torch.softmax(semantic_logits, dim=-1)
            gathered = semantic_scores.gather(dim=-1, index=selected_heads)
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
        selected_heads, _, _ = router(x, active_mask, None)
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
        selected_heads, _, _ = router(x, active_mask, None)
        B, N, K = selected_heads.shape
        for b in range(B):
            for n in range(N):
                assert selected_heads[b, n].unique().shape[0] == K


# ---------------------------------------------------------------------------
# Bias routing behavior
# ---------------------------------------------------------------------------

class TestBiasRoutingBehavior:
    def test_large_bias_forces_head_selection(self):
        """A large expert_bias on head 0 must cause head 0 to appear in every token's
        selection — demonstrating that expert_bias drives selection via semantic_logits."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(1, 6, 64)
        active_mask = torch.ones(1, 6, dtype=torch.bool)

        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            selected_heads, _, _ = router(x, active_mask, None)

        # Head 0 must be selected for every token when its bias is enormous.
        assert (selected_heads == 0).any(dim=-1).all()

    def test_bias_incorporated_in_routing_probs(self):
        """With non-zero expert_bias, routing_probs must differ from the zero-bias case.

        Under the two-pathway architecture, routing_probs are gathered from
        softmax(semantic_logits) = softmax(logits + expert_bias.detach()). A large
        bias on one head shifts the softmax distribution, producing different routing_probs
        than the zero-bias baseline.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.randn(1, 4, 64)
        active_mask = torch.ones(1, 4, dtype=torch.bool)

        with torch.no_grad():
            router.expert_bias.zero_()
            _, routing_probs_zero_bias, _ = router(x, active_mask, None)

            router.expert_bias.zero_()
            router.expert_bias[0] = 100.0
            _, routing_probs_biased, _ = router(x, active_mask, None)

        assert not torch.allclose(routing_probs_zero_bias, routing_probs_biased, atol=1e-4), (
            "routing_probs must differ when expert_bias is non-zero — bias must be "
            "incorporated via the semantic gradient channel"
        )


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
        _, _, diagnostics = router(x, active_mask, None)
        load_balance_loss = diagnostics["load_balance_loss"]
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
        _, _, diagnostics = router(x, active_mask, None)
        load_balance_loss = diagnostics["load_balance_loss"]
        load_balance_loss.backward()

        # At initialization with zero biases and random weights the routing will be
        # imperfectly balanced, so at least one head's gradient should be non-zero.
        assert router.expert_bias.grad.abs().sum().item() > 0.0

    def test_routing_projection_weight_grad_is_none_after_load_balance_loss_backward(self):
        """Backward on load_balance_loss must not populate routing_projection.weight.grad.

        Gradient isolation invariant: assignment probabilities are computed via
        softmax(logits.detach() + expert_bias), so there is no autograd path from
        load_balance_loss back to routing_projection.weight.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, None)
        diagnostics["load_balance_loss"].backward()

        assert router.routing_projection.weight.grad is None


# ---------------------------------------------------------------------------
# Two-pathway gradient isolation (Unit 24.C)
# ---------------------------------------------------------------------------

class TestGradientIsolationTwoPathway:
    """Tests certifying the two-pathway gradient architecture of Unit 24.C.

    semantic_logits = logits + expert_bias.detach() drives selection and routing_probs.
    load_balancing_logits = logits.detach() + expert_bias drives assignment_probs.
    Each pathway isolates one parameter set from the other's loss.
    """

    def test_task_loss_does_not_reach_expert_bias(self):
        """Backward on task loss must not populate expert_bias.grad.

        semantic_logits uses expert_bias.detach() — there is no autograd path
        from routing_probs or selected_heads back to expert_bias.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, routing_probs, _ = router(x, active_mask, None)
        routing_probs.sum().backward()

        assert router.routing_projection.weight.grad is not None
        assert router.expert_bias.grad is None

    def test_load_balance_loss_does_not_reach_routing_projection(self):
        """Backward on load_balance_loss must not populate routing_projection.weight.grad.

        load_balancing_logits uses logits.detach() — there is no autograd path
        from assignment_probs back to routing_projection.weight.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, None)
        diagnostics["load_balance_loss"].backward()

        assert router.expert_bias.grad is not None
        assert router.routing_projection.weight.grad is None

    def test_assignment_probs_not_contaminated_by_capacity_masking(self):
        """Load balance gradients must be finite when a preferred expert is over capacity.

        Post-capacity bug: softmax over -1e8-masked logits gives p_0 = exp(-1e8) / sum,
        which underflows to 0.0 in float32. CE loss computes log(0) = -inf, producing
        inf/NaN gradients through expert_bias.

        Pre-capacity fix (correct): softmax(load_balancing_logits) is computed before
        balance_capacity. With expert_bias[0] = 50, p_0 ≈ 1.0 and log(p_0) ≈ 0 —
        gradient is bounded.
        """
        torch.manual_seed(0)
        # Sparse routing so one expert can be genuinely over capacity in inference mode.
        config = small_config(
            num_mosrah_heads=4,
            num_selected_heads=2,
            training_sequence_length=8,
            inference_sequence_length=8,
            mosrah_overallocation_factor=2.0,
        )
        router = MoSRAHRouter(config)

        capacity = config.mosrah_cache_length
        B, N, L = 1, 2, 4
        used_capacity = torch.zeros(B, L, dtype=torch.long)
        used_capacity[0, 0] = capacity  # expert 0 fully at capacity

        # Large positive bias toward expert 0 (already at capacity).
        with torch.no_grad():
            router.expert_bias.zero_()
            router.expert_bias[0] = 50.0

        x = torch.randn(B, N, config.embedding_width)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, used_capacity)
        diagnostics["load_balance_loss"].backward()

        # Post-capacity bug: p_0 = 0.0 (float32 underflow) → log(0) = -inf → inf grad.
        # Pre-capacity fix: p_0 ≈ 1.0 (from bias=50) → log(1) ≈ 0 → bounded grad.
        assert torch.isfinite(router.expert_bias.grad).all(), (
            "expert_bias.grad is non-finite — assignment_probs may have been computed "
            "post-balance_capacity, producing log(0) from the -1e8 sentinel"
        )
        assert router.expert_bias.grad.abs().max().item() < 1e6


# ---------------------------------------------------------------------------
# Router diagnostics
# ---------------------------------------------------------------------------

class TestRouterDiagnostics:
    """Tests for the router_diagnostics dict returned from MoSRAHRouter.forward.

    Verifies that routing decisions and routing feedback are structurally
    separated, that gradients flow only through load_balance_loss, and that
    the four load-balance health scalars have the correct relationships.
    """

    def test_load_balance_loss_has_gradient(self):
        """load_balance_loss must retain its gradient; it is the training signal."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["load_balance_loss"].requires_grad

    def test_diagnostic_scalars_are_detached(self):
        """All diagnostic scalars except load_balance_loss must be detached."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        for key in ("max_vio", "bias_std", "raw_logit_std", "logit_std", "bias_alignment"):
            assert not diagnostics[key].requires_grad, (
                f"diagnostic scalar '{key}' must be detached but requires_grad is True"
            )

    def test_bias_std_zero_when_bias_zero(self):
        """bias_std must be zero when expert_bias is the zero vector."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            router.expert_bias.zero_()
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_std"].item() == 0.0

    def test_logit_std_equals_raw_logit_std_when_bias_zero(self):
        """logit_std must equal raw_logit_std when expert_bias is zero.

        With a zero bias, (logits + expert_bias) == logits, so the combined
        spread is identical to the unbiased spread.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            router.expert_bias.zero_()
        _, _, diagnostics = router(x, active_mask, None)
        assert torch.allclose(diagnostics["logit_std"], diagnostics["raw_logit_std"], atol=1e-6)

    def test_bias_alignment_negative_when_bias_opposes_logits(self):
        """bias_alignment must be negative when expert_bias points opposite to logits.

        Computed by measuring the mean logit direction then setting expert_bias
        to its negation, guaranteeing a negative mean cosine similarity.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.randn(1, 8, 64)
        active_mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            logits = router.routing_projection(x)          # (1, 8, L)
            mean_logit_direction = logits.mean(dim=(0, 1)) # (L,)
            router.expert_bias.copy_(-mean_logit_direction)
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_alignment"].item() < 0, (
            f"expected negative bias_alignment for opposing bias, "
            f"got {diagnostics['bias_alignment'].item()}"
        )

    def test_bias_alignment_positive_when_bias_reinforces_logits(self):
        """bias_alignment must be positive when expert_bias reinforces logit direction.

        Computed by setting expert_bias to the mean logit direction, guaranteeing
        a positive mean cosine similarity.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.randn(1, 8, 64)
        active_mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            logits = router.routing_projection(x)
            mean_logit_direction = logits.mean(dim=(0, 1))
            router.expert_bias.copy_(mean_logit_direction)
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_alignment"].item() > 0, (
            f"expected positive bias_alignment for reinforcing bias, "
            f"got {diagnostics['bias_alignment'].item()}"
        )


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
# Routing scale (Unit 24.B)
# ---------------------------------------------------------------------------

class TestRoutingScale:
    """Tests certifying the routing_scale scalar gate of Unit 24.B.

    HuggingFace init survival is tested in test_end_to_end.py
    (TestIntegrationRoutingScale) — that invariant requires the full model
    construction path and does not belong at unit level.
    """

    def test_routing_scale_is_scalar(self):
        """routing_scale must have shape (1,) — a single scalar gate."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.routing_scale.shape == (1,)

    def test_routing_scale_is_parameter(self):
        """routing_scale must be an nn.Parameter so the optimizer can update it."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert isinstance(router.routing_scale, torch.nn.Parameter)


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
        _, _, diagnostics = router(x, active_mask, None)
        max_vio = diagnostics["max_vio"]
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
        _, _, diag_a = router(x, active_mask, None)
        loss_a = diag_a["load_balance_loss"]

        x_modified = x.clone()
        x_modified[0, 3] = torch.randn(config.embedding_width) * 100.0
        _, _, diag_b = router(x_modified, active_mask, None)
        loss_b = diag_b["load_balance_loss"]

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
        _, _, diag_a = router(x, active_mask, None)
        vio_a = diag_a["max_vio"]

        x_modified = x.clone()
        x_modified[1, 5] = torch.randn(config.embedding_width) * 100.0
        _, _, diag_b = router(x_modified, active_mask, None)
        vio_b = diag_b["max_vio"]

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
            selected_heads, _, diagnostics = router(x, active_mask, None)
            max_vio = diagnostics["max_vio"]
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
        with pytest.raises((RuntimeError, AssertionError)):
            MoSRAHRouter.balance_capacity(logits, None, capacity=capacity, min_choices=min_choices, max_rounds=10)


class TestRouterRealizedCapacityFuzz:
    """Fuzz tests for the router's realized selected-head capacity contract.

    Packing consumes the final `selected_heads` index tensor, not the router's
    internal capacity mask. The contract tested here is therefore only the
    downstream-relevant one: after routing, no active `(batch, expert)` bucket
    may contain more routed token copies than `config.mosrah_packed_length`.

    This suite fixes the architecture/config, reads the actual packed capacity
    from config, varies runtime sequence length to hit a target capacity-use
    ratio, and counts the realized selected-head indices.
    """

    NUM_TRIALS = 100
    BATCH_SIZE = 4
    TRAINING_SEQUENCE_LENGTH = 256
    MAX_BID_ROUNDS = 64

    SPARSITY_PROFILES = (
        (16, 16),
        (32, 16),
        (64, 16),
    )

    @staticmethod
    def _make_config(num_mosrah_heads: int, num_selected_heads: int) -> ShramConfig:
        """Build the router config for one sparsity profile.

        The production path obtains packed capacity from
        `config.mosrah_packed_length`. The test does the same so it certifies
        the same capacity boundary consumed later by expert packing.
        """

        return small_config(
            embedding_width=64,
            num_mosrah_heads=num_mosrah_heads,
            num_selected_heads=num_selected_heads,
            num_sliding_window_heads=4,
            head_dim=16,
            training_sequence_length=(
                TestRouterRealizedCapacityFuzz.TRAINING_SEQUENCE_LENGTH
            ),
            inference_sequence_length=(
                TestRouterRealizedCapacityFuzz.TRAINING_SEQUENCE_LENGTH
            ),
            mosrah_overallocation_factor=1.25,
            max_bid_rounds=TestRouterRealizedCapacityFuzz.MAX_BID_ROUNDS,
            use_cache=False,
        )

    @staticmethod
    def _runtime_length_for_capacity_use(
        capacity_use: float,
        capacity: int,
        num_mosrah_heads: int,
        num_selected_heads: int,
    ) -> int:
        """Return runtime token count for the requested capacity-use ratio.

        Capacity use is `N * K / (C * L)`, where `N` is runtime token count,
        `K` is selected heads per token, `C` is per-expert packed capacity, and
        `L` is total MoSRAH heads. In practice `L`, `K`, and `C` are fixed by
        architecture/config; `N` is the runtime load knob.
        """

        return int(
            capacity_use
            * capacity
            * num_mosrah_heads
            / num_selected_heads
        )

    @staticmethod
    def _count_selected_heads(
        selected_heads: torch.Tensor,
        active_mask: torch.Tensor,
        num_mosrah_heads: int,
    ) -> torch.Tensor:
        """Count active routed token copies per `(batch, expert)` bucket."""

        batch_size = selected_heads.shape[0]

        counts = torch.zeros(
            batch_size,
            num_mosrah_heads,
            dtype=torch.long,
            device=selected_heads.device,
        )

        active_selected_heads = selected_heads.masked_fill(
            ~active_mask.unsqueeze(-1),
            0,
        )

        active_copies = active_mask.unsqueeze(-1).expand_as(selected_heads)
        src = active_copies.to(dtype=torch.long)

        counts.scatter_add_(
            dim=-1,
            index=active_selected_heads.reshape(batch_size, -1),
            src=src.reshape(batch_size, -1),
        )

        return counts

    def _run_realized_capacity_fuzz(
        self,
        device: torch.device,
        capacity_use: float,
    ) -> None:
        """Run randomized realized-capacity trials for one capacity-use ratio."""

        generator = torch.Generator(device=device)
        generator.manual_seed(19317)

        for num_mosrah_heads, num_selected_heads in self.SPARSITY_PROFILES:
            config = self._make_config(num_mosrah_heads, num_selected_heads)
            router = MoSRAHRouter(config).to(device)

            capacity = config.mosrah_packed_length
            runtime_length = self._runtime_length_for_capacity_use(
                capacity_use,
                capacity,
                num_mosrah_heads,
                num_selected_heads,
            )

            assert runtime_length > 0

            actual_capacity_use = (
                runtime_length
                * num_selected_heads
                / (capacity * num_mosrah_heads)
            )

            for trial in range(self.NUM_TRIALS):
                x = torch.randn(
                    self.BATCH_SIZE,
                    runtime_length,
                    config.embedding_width,
                    generator=generator,
                    device=device,
                )
                active_mask = torch.ones(
                    self.BATCH_SIZE,
                    runtime_length,
                    dtype=torch.bool,
                    device=device,
                )

                selected_heads, _, _ = router(
                    x,
                    active_mask,
                    used_capacity=None,
                )

                assert selected_heads.shape == (
                    self.BATCH_SIZE,
                    runtime_length,
                    num_selected_heads,
                )

                counts = self._count_selected_heads(
                    selected_heads,
                    active_mask,
                    num_mosrah_heads,
                )

                max_count = counts.max().item()

                assert max_count <= capacity, (
                    "router exceeded realized packed capacity: "
                    f"requested_capacity_use={capacity_use}, "
                    f"actual_capacity_use={actual_capacity_use}, "
                    f"trial={trial}, "
                    f"B={self.BATCH_SIZE}, "
                    f"N={runtime_length}, "
                    f"L={num_mosrah_heads}, "
                    f"K={num_selected_heads}, "
                    f"C={capacity}, "
                    f"max_count={max_count}, "
                    f"counts={counts}"
                )

    def test_realized_capacity_fuzz_50_percent(self, device):
        """Final selected_heads must obey capacity at 50% packed-capacity use."""

        self._run_realized_capacity_fuzz(device, capacity_use=0.50)

    def test_realized_capacity_fuzz_80_percent(self, device):
        """Final selected_heads must obey capacity at 80% packed-capacity use."""

        self._run_realized_capacity_fuzz(device, capacity_use=0.80)

    def test_realized_capacity_fuzz_90_percent(self, device):
        """Final selected_heads must obey capacity at 90% packed-capacity use."""

        self._run_realized_capacity_fuzz(device, capacity_use=0.90)


class TestRouterCompileEquivalenceFuzz:
    """Fuzz tests for eager-vs-compiled MoSRAHRouter equivalence.

    This suite certifies that compiling the router does not change its forward
    contract. It intentionally compares the full router outputs, not just the
    realized capacity counts, because the compile boundary should preserve
    selection, probabilities, load-balance loss, and monitoring output for the
    same module state and input tensors.

    The tests are CUDA-only because this project already treats uncached
    torch.compile coverage as CUDA-only.
    """

    NUM_TRIALS = 25
    BATCH_SIZE = 4
    TRAINING_SEQUENCE_LENGTH = 256
    MAX_BID_ROUNDS = 64

    SPARSITY_PROFILES = (
        (16, 16),
        (32, 16),
        (64, 16),
    )

    @staticmethod
    def _make_config(
        num_mosrah_heads: int,
        num_selected_heads: int,
    ) -> ShramConfig:
        """Build the router config for one sparsity profile."""

        return small_config(
            embedding_width=64,
            num_mosrah_heads=num_mosrah_heads,
            num_selected_heads=num_selected_heads,
            num_sliding_window_heads=4,
            head_dim=16,
            training_sequence_length=(
                TestRouterCompileEquivalenceFuzz.TRAINING_SEQUENCE_LENGTH
            ),
            inference_sequence_length=(
                TestRouterCompileEquivalenceFuzz.TRAINING_SEQUENCE_LENGTH
            ),
            mosrah_overallocation_factor=1.25,
            max_bid_rounds=TestRouterCompileEquivalenceFuzz.MAX_BID_ROUNDS,
            use_cache=False,
        )

    @staticmethod
    def _runtime_length_for_capacity_use(
        capacity_use: float,
        capacity: int,
        num_mosrah_heads: int,
        num_selected_heads: int,
    ) -> int:
        """Return runtime token count for the requested capacity-use ratio."""

        return int(
            capacity_use
            * capacity
            * num_mosrah_heads
            / num_selected_heads
        )

    def _run_compile_equivalence_fuzz(
        self,
        device: torch.device,
        capacity_use: float,
    ) -> None:
        """Compare eager and compiled router outputs at one capacity pressure."""

        if device.type != "cuda":
            pytest.skip(
                "Router torch.compile equivalence is CUDA-only for this suite."
            )

        torch._dynamo.reset()

        generator = torch.Generator(device=device)
        generator.manual_seed(24103)

        for num_mosrah_heads, num_selected_heads in self.SPARSITY_PROFILES:
            config = self._make_config(num_mosrah_heads, num_selected_heads)

            router = MoSRAHRouter(config).eval().to(device)

            # routing_scale is initialized near-zero (1e-4) to equalize logit
            # magnitude with expert_bias at training start. At that scale, logit
            # differences at the TopK boundary fall within floating-point rounding
            # noise, causing implementation-dependent tie-breaking that differs
            # between eager and compiled execution paths. This test certifies
            # compiled/eager equivalence under a well-separated routing signal,
            # not the load-balance initialization regime. Resetting to 1.0
            # restores logit separability and makes TopK deterministic.
            router.routing_scale.data.fill_(1.0)

            compiled_router = torch.compile(
                router,
                fullgraph=True,
                dynamic=False,
            )

            capacity = config.mosrah_packed_length
            runtime_length = self._runtime_length_for_capacity_use(
                capacity_use,
                capacity,
                num_mosrah_heads,
                num_selected_heads,
            )

            assert runtime_length > 0

            actual_capacity_use = (
                runtime_length
                * num_selected_heads
                / (capacity * num_mosrah_heads)
            )

            for trial in range(self.NUM_TRIALS):
                x = torch.randn(
                    self.BATCH_SIZE,
                    runtime_length,
                    config.embedding_width,
                    generator=generator,
                    device=device,
                )
                active_mask = torch.ones(
                    self.BATCH_SIZE,
                    runtime_length,
                    dtype=torch.bool,
                    device=device,
                )

                with torch.no_grad():
                    eager_selected, eager_probs, eager_diag = router(
                        x,
                        active_mask,
                        used_capacity=None,
                    )

                    compiled_selected, compiled_probs, compiled_diag = (
                        compiled_router(
                            x,
                            active_mask,
                            None,
                        )
                    )
                assert eager_selected.dtype == compiled_selected.dtype
                assert eager_selected.device == compiled_selected.device
                assert torch.equal(eager_selected, compiled_selected), (
                    "compiled router selected different heads: "
                    f"capacity_use={capacity_use}, "
                    f"actual_capacity_use={actual_capacity_use}, "
                    f"trial={trial}, "
                    f"B={self.BATCH_SIZE}, "
                    f"N={runtime_length}, "
                    f"L={num_mosrah_heads}, "
                    f"K={num_selected_heads}, "
                    f"C={capacity}"
                )

                torch.testing.assert_close(
                    compiled_probs,
                    eager_probs,
                    rtol=1e-5,
                    atol=1e-6,
                    msg=(
                        "compiled router produced different routing_probs: "
                        f"capacity_use={capacity_use}, "
                        f"actual_capacity_use={actual_capacity_use}, "
                        f"trial={trial}, "
                        f"L={num_mosrah_heads}, "
                        f"K={num_selected_heads}"
                    ),
                )

                for key in eager_diag:
                    assert compiled_diag[key].device == eager_diag[key].device
                    assert eager_diag[key].dtype == eager_diag[key].dtype
                    torch.testing.assert_close(
                        compiled_diag[key],
                        eager_diag[key],
                        rtol=1e-5,
                        atol=1e-6,
                        msg=(
                            f"compiled router diagnostic '{key}' differs from eager: "
                            f"capacity_use={capacity_use}, "
                            f"actual_capacity_use={actual_capacity_use}, "
                            f"trial={trial}, "
                            f"L={num_mosrah_heads}, "
                            f"K={num_selected_heads}"
                        ),
                    )

        torch._dynamo.reset()

    def test_compile_equivalence_fuzz_50_percent(self, device):
        """Compiled and eager router forward must match at 50% capacity use."""

        self._run_compile_equivalence_fuzz(device, capacity_use=0.50)

    def test_compile_equivalence_fuzz_80_percent(self, device):
        """Compiled and eager router forward must match at 80% capacity use."""

        self._run_compile_equivalence_fuzz(device, capacity_use=0.80)

    def test_compile_equivalence_fuzz_90_percent(self, device):
        """Compiled and eager router forward must match at 90% capacity use."""

        self._run_compile_equivalence_fuzz(device, capacity_use=0.90)

# ---------------------------------------------------------------------------
# get_mask
# ---------------------------------------------------------------------------

# Alias the static method so tests call the production code directly without
# being coupled to the class name in every assertion.
get_mask = MoSRAHRouter.get_best_proposals


def make_tensor(*shape):
    """Deterministic tensor for reproducible tests."""
    return torch.arange(math.prod(shape), dtype=torch.float).reshape(shape)


class TestGetMaskContract:
    """
    Tests verify the contract: mask has exactly min(n_per_slice, dim_length)
    True entries per slice along dim, at the highest-valued positions.
    """

    def test_int_n_selects_top_n_dim_last(self):
        """Exactly n True entries per row along the last dimension."""
        t = torch.randn(3, 8)
        for n in range(1, 8):
            mask = get_mask(t, dim=-1, n=n, capacity_scalar=8)
            count = mask.sum(dim=-1)
            assert (count == n).all(), \
                f"n={n}: expected {n} True per row, got {count}"

    def test_int_n_selects_top_n_dim_second_last(self):
        """Exactly n True entries per column along the second-to-last dimension."""
        t = torch.randn(2, 8, 4)
        for n in range(1, 8):
            mask = get_mask(t, dim=-2, n=n, capacity_scalar=8)
            count = mask.sum(dim=-2)
            assert (count == n).all(), \
                f"n={n}: expected {n} True per column"

    def test_tensor_n_selects_top_n_per_row(self):
        """Each row gets exactly n[row] True entries."""
        t = torch.randn(4, 8)
        n = torch.randint(1, 8, (4,))
        mask = get_mask(t, dim=-1, n=n, capacity_scalar=8)
        for i in range(t.shape[0]):
            count = mask[i].sum()
            assert count.item() == n[i].item(), \
                f"row {i}: expected {n[i]} True, got {count}"

    def test_tensor_n_column_dim(self):
        """Tensor-n contract holds for the second-to-last dimension."""
        t = torch.randn(2, 8, 4)
        n = torch.randint(1, 8, (2, 4))
        mask = get_mask(t, dim=-2, n=n, capacity_scalar=8)
        for b in range(2):
            for j in range(4):
                count = mask[b, :, j].sum()
                assert count.item() == n[b, j].item()

    def test_selected_entries_are_highest_valued(self):
        """True entries must be the n highest-valued positions per row."""
        t = torch.randn(3, 8)
        n = 3
        mask = get_mask(t, dim=-1, n=n, capacity_scalar=8)
        for i in range(t.shape[0]):
            selected_min = t[i][mask[i]].min()
            unselected_max = t[i][~mask[i]].max()
            assert selected_min >= unselected_max, \
                f"row {i}: selected values are not the top {n}"


class TestGetMaskBoundaries:

    def test_int_n_zero_returns_all_false(self):
        """n=0 must return all-False — nothing qualifies."""
        t = torch.randn(4, 8)
        mask = get_mask(t, dim=-1, n=0, capacity_scalar=8)
        assert not mask.any()

    def test_int_n_overflow_returns_all_true(self):
        """n >= dim_length must return all-True — every entry qualifies."""
        t = torch.randn(4, 8)
        mask = get_mask(t, dim=-1, n=9, capacity_scalar=9)   # dim_length == 8
        assert mask.all()

    def test_tensor_n_zero_positions_return_all_false(self):
        """Tensor n=0 per row must produce all-False rows."""
        t = torch.randn(4, 8)
        n = torch.zeros(4, dtype=torch.long)
        mask = get_mask(t, dim=-1, n=n, capacity_scalar=8)
        assert not mask.any()

    def test_tensor_n_overflow_positions_return_all_true(self):
        """Tensor n >= dim_length per row must produce all-True rows."""
        t = torch.randn(4, 8)
        n = torch.full((4,), 9, dtype=torch.long)
        mask = get_mask(t, dim=-1, n=n, capacity_scalar=9)   # dim_length == 8
        assert mask.all()

    def test_tensor_n_mixed_boundaries_and_valid(self):
        """n=0 rows all-False, valid n rows correct count, overflow rows all-True."""
        t = torch.randn(3, 8)
        # row 0: n=0 (all False), row 1: n=4 (4 True), row 2: n=9 (all True)
        n = torch.tensor([0, 4, 9])
        mask = get_mask(t, dim=-1, n=n, capacity_scalar=9)
        assert not mask[0].any()
        assert mask[1].sum().item() == 4
        assert mask[2].all()


class TestGetMaskShape:

    def test_output_shape_matches_input_dim_last(self):
        """Output shape must equal input shape."""
        t = torch.randn(2, 5, 8)
        mask = get_mask(t, dim=-1, n=3, capacity_scalar=8)
        assert tuple(mask.shape) == (2, 5, 8)

    def test_output_shape_matches_input_dim_second_last(self):
        """Output shape must equal input shape for dim=-2."""
        t = torch.randn(2, 8, 4)
        mask = get_mask(t, dim=-2, n=3, capacity_scalar=8)
        assert tuple(mask.shape) == (2, 8, 4)

    def test_output_is_bool(self):
        """Output dtype must be bool."""
        t = torch.randn(4, 8)
        mask = get_mask(t, dim=-1, n=3, capacity_scalar=8)
        assert mask.dtype == torch.bool

    def test_device_preserved(self):
        """Output must be on the same device as the input."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.randn(4, 8, device=device)
        mask = get_mask(t, dim=-1, n=3, capacity_scalar=8)
        assert mask.device.type == device


class TestGetMaskIntTensorAgreement:
    """Int and tensor paths must produce identical masks for the same n values."""

    def test_paths_agree_dim_last(self):
        t = torch.randn(3, 8)
        for n_val in range(1, 9):
            int_mask    = get_mask(t, dim=-1, n=n_val, capacity_scalar=8)
            tensor_mask = get_mask(t, dim=-1, n=torch.full((3,), n_val), capacity_scalar=8)
            assert (int_mask == tensor_mask).all(), \
                f"paths disagree at n={n_val}"

    def test_paths_agree_dim_second_last(self):
        t = torch.randn(2, 8, 4)
        for n_val in range(1, 9):
            int_mask    = get_mask(t, dim=-2, n=n_val, capacity_scalar=8)
            tensor_mask = get_mask(t, dim=-2, n=torch.full((2, 4), n_val), capacity_scalar=8)
            assert (int_mask == tensor_mask).all(), \
                f"paths disagree at n={n_val}"
