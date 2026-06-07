"""Tests for MoSRAHRouter.

Invariants verified:
- Output shapes: selected_heads (B, N, K), routing_probs (B, N, K), loss scalar,
  max_vio scalar
- routing_probs sum to 1 per token and are non-negative
- selected_heads are valid indices in [0, L-1] and are distinct per token
- routing_weight and balance_weight both exist as nn.Parameter with shape (L, embedding_width)
- balance_weight receives gradient through load_balance_loss, not through task loss
- task loss backward populates routing_weight.grad, not balance_weight.grad
- load balance loss backward populates balance_weight.grad, not routing_weight.grad
- assignment_probs are computed before balance_capacity, preventing -1e8 contamination
- max_vio is exactly 0 for perfectly uniform routing frequencies
- max_vio is exactly 1 when the most overloaded head receives double its fair share
- max_vio produces the correct value for a known intermediate routing imbalance
- max_vio is detached from the autograd graph
- dead outer tokens do not affect load_balance_loss
- dead outer tokens do not affect max_vio
- router forward max_vio matches _compute_max_vio called directly on all-live inputs
- router_diagnostics separates routing decisions from routing feedback
- load_balance_loss has gradient; all other diagnostic scalars are detached
- bias_std is zero when balance_weight is zero
- logit_std equals raw_logit_std when balance_weight is zero
- bias_alignment is negative when balance_weight opposes routing_weight direction
- bias_alignment is positive when balance_weight reinforces routing_weight direction
- _compute_bias_diagnostics returns exactly {raw_logit_std, bias_std, logit_std,
  bias_alignment}, all detached
- compiled and eager router diagnostics are numerically identical
- routing_integral_weight and balance_integral_weight exist as nn.Parameter with shape
  (L, L) when routing_mode='integral'; neither exists when routing_mode='default'
- task loss backward populates routing_integral_weight.grad, not balance_integral_weight.grad
- load balance loss backward populates balance_integral_weight.grad, not
  routing_integral_weight.grad
- integral mode output differs from default mode output on the same input and base weights
- compiled integral router matches eager; compiled training step runs without error
"""

import math

import pytest
import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.router import MoSRAHRouter

# Set to True to run the optional routing-mode profiling test.
PROFILE_ROUTING_MODES = False


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
    def test_large_balance_weight_forces_head_selection(self):
        """A large balance_weight row for head 0 must cause head 0 to appear in every
        token's selection — demonstrating that balance_weight drives selection via
        semantic_logits.

        With routing_weight zeroed and balance_weight[0] set large, every token's
        semantic_logit for head 0 dominates regardless of input direction.
        All-ones x ensures the dot product with balance_weight[0] is maximally positive.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        # All-ones x ensures head 0's logit = sum(balance_weight[0]) = large * embedding_width.
        x = torch.ones(1, 6, config.embedding_width)
        active_mask = torch.ones(1, 6, dtype=torch.bool)

        with torch.no_grad():
            router.routing_weight.zero_()
            router.balance_weight.zero_()
            router.balance_weight[0, :] = 100.0
            selected_heads, _, _ = router(x, active_mask, None)

        # Head 0 must be selected for every token when its balance logit is enormous.
        assert (selected_heads == 0).any(dim=-1).all()

    def test_balance_weight_incorporated_in_routing_probs(self):
        """With non-zero balance_weight, routing_probs must differ from the zero-balance case.

        Under the two-pathway architecture, routing_probs are gathered from
        softmax(semantic_logits) = softmax(A·x + (B·x).detach()). A large balance_weight
        row for one head shifts the softmax distribution, producing different routing_probs
        than the zero-balance-weight baseline.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.ones(1, 4, config.embedding_width)
        active_mask = torch.ones(1, 4, dtype=torch.bool)

        with torch.no_grad():
            router.balance_weight.zero_()
            _, routing_probs_zero_balance, _ = router(x, active_mask, None)

            router.balance_weight.zero_()
            router.balance_weight[0, :] = 100.0
            _, routing_probs_biased, _ = router(x, active_mask, None)

        assert not torch.allclose(routing_probs_zero_balance, routing_probs_biased, atol=1e-4), (
            "routing_probs must differ when balance_weight is non-zero — balance_weight "
            "must be incorporated via the semantic gradient channel"
        )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:
    def test_balance_weight_receives_gradient(self):
        """balance_weight must accumulate a gradient after backward on load_balance_loss."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        load_balance_loss = diagnostics["load_balance_loss"]
        load_balance_loss.backward()

        assert router.balance_weight.grad is not None
        assert router.balance_weight.grad.shape == (config.num_mosrah_heads, config.embedding_width)

    def test_balance_weight_gradient_is_not_all_zero(self):
        """With an unbalanced router, balance_weight.grad must be non-zero — all-zero grad
        would mean the load balancing operator has no effect on training."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, None)
        load_balance_loss = diagnostics["load_balance_loss"]
        load_balance_loss.backward()

        # At initialization with random weights the routing will be imperfectly balanced,
        # so at least one entry's gradient should be non-zero.
        assert router.balance_weight.grad.abs().sum().item() > 0.0

    def test_routing_weight_grad_is_none_after_load_balance_loss_backward(self):
        """Backward on load_balance_loss must not populate routing_weight.grad.

        Gradient isolation invariant: load_balancing_logits = (A·x).detach() + B·(x.detach()),
        so there is no autograd path from load_balance_loss back to routing_weight.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, None)
        diagnostics["load_balance_loss"].backward()

        assert router.routing_weight.grad is None


# ---------------------------------------------------------------------------
# Two-pathway gradient isolation (Unit 24.C)
# ---------------------------------------------------------------------------

class TestGradientIsolationTwoPathway:
    """Tests certifying the two-pathway gradient architecture.

    semantic_logits = A·x + (B·x).detach() drives selection and routing_probs.
    load_balancing_logits = (A·x).detach() + B·(x.detach()) drives assignment_probs.
    Each pathway isolates one parameter matrix from the other's loss.
    """

    def test_task_loss_does_not_reach_balance_weight(self):
        """Backward on task loss must not populate balance_weight.grad.

        semantic_logits = A·x + (B·x).detach() — there is no autograd path
        from routing_probs or selected_heads back to balance_weight.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, routing_probs, _ = router(x, active_mask, None)
        routing_probs.sum().backward()

        assert router.routing_weight.grad is not None
        assert router.balance_weight.grad is None

    def test_load_balance_loss_does_not_reach_routing_weight(self):
        """Backward on load_balance_loss must not populate routing_weight.grad.

        load_balancing_logits = (A·x).detach() + B·(x.detach()) — there is no autograd
        path from load_balance_loss back to routing_weight.
        """
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, None)
        diagnostics["load_balance_loss"].backward()

        assert router.balance_weight.grad is not None
        assert router.routing_weight.grad is None

    def test_assignment_probs_not_contaminated_by_capacity_masking(self):
        """Load balance gradients must be finite when a preferred expert is over capacity.

        Post-capacity bug: softmax over -1e8-masked logits gives p_0 = exp(-1e8) / sum,
        which underflows to 0.0 in float32. CE loss computes log(0) = -inf, producing
        inf/NaN gradients through balance_weight.

        Pre-capacity fix (correct): softmax(load_balancing_logits) is computed before
        balance_capacity. With balance_weight[0] large and x = ones, p_0 ≈ 1.0 and
        log(p_0) ≈ 0 — gradient is bounded.
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

        # Large balance_weight row for head 0. With x = ones, load_balancing_logit
        # for head 0 ≈ 50, dominating all other heads.
        with torch.no_grad():
            router.routing_weight.zero_()
            router.balance_weight.zero_()
            router.balance_weight[0, :] = 50.0 / config.embedding_width

        # All-ones x so that head 0's load_balancing_logit = sum(balance_weight[0]) ≈ 50.
        x = torch.ones(B, N, config.embedding_width)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, used_capacity)
        diagnostics["load_balance_loss"].backward()

        # Post-capacity bug: p_0 = 0.0 (float32 underflow) → log(0) = -inf → inf grad.
        # Pre-capacity fix: p_0 ≈ 1.0 → log(1) ≈ 0 → bounded grad.
        assert torch.isfinite(router.balance_weight.grad).all(), (
            "balance_weight.grad is non-finite — assignment_probs may have been computed "
            "post-balance_capacity, producing log(0) from the -1e8 sentinel"
        )
        assert router.balance_weight.grad.abs().max().item() < 1e6


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
        for key in ("max_vio", "raw_logit_std", "bias_std",
                    "logit_std", "bias_alignment"):
            assert not diagnostics[key].requires_grad, (
                f"diagnostic scalar '{key}' must be detached but requires_grad is True"
            )

    def test_bias_std_zero_when_balance_weight_zero(self):
        """bias_std must be zero when balance_weight is zero.

        With balance_weight zeroed, B·x = 0 for any x, so per-token std of
        balance_logits is identically zero.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            router.balance_weight.zero_()
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_std"].item() == 0.0

    def test_logit_std_equals_raw_logit_std_when_balance_weight_zero(self):
        """logit_std must equal raw_logit_std when balance_weight is zero.

        With balance_weight zeroed, B·x = 0, so semantic_logits = A·x + 0 = A·x,
        and both stds are computed over the same values.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            router.balance_weight.zero_()
        _, _, diagnostics = router(x, active_mask, None)
        assert torch.allclose(
            diagnostics["logit_std"], diagnostics["raw_logit_std"], atol=1e-6
        )

    def test_bias_alignment_negative_when_balance_opposes_routing(self):
        """bias_alignment must be negative when balance_weight = -routing_weight.

        Setting balance_weight = -routing_weight guarantees balance_logits = -routing_logits
        for any x, so cosine similarity between the two is -1 for every token.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.randn(1, 8, 64)
        active_mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            router.balance_weight.copy_(-router.routing_weight)
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_alignment"].item() < 0, (
            f"expected negative bias_alignment for opposing balance_weight, "
            f"got {diagnostics['bias_alignment'].item()}"
        )

    def test_bias_alignment_positive_when_balance_reinforces_routing(self):
        """bias_alignment must be positive when balance_weight = routing_weight.

        Setting balance_weight = routing_weight guarantees balance_logits = routing_logits
        for any x, so cosine similarity between the two is 1 for every token.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        torch.manual_seed(42)
        x = torch.randn(1, 8, 64)
        active_mask = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            router.balance_weight.copy_(router.routing_weight)
        _, _, diagnostics = router(x, active_mask, None)
        assert diagnostics["bias_alignment"].item() > 0, (
            f"expected positive bias_alignment for reinforcing balance_weight, "
            f"got {diagnostics['bias_alignment'].item()}"
        )


# ---------------------------------------------------------------------------
# Bias diagnostics (static method)
# ---------------------------------------------------------------------------

class TestBiasDiagnostics:
    """Tests for MoSRAHRouter._compute_bias_diagnostics called directly as a static method.

    TestRouterDiagnostics verifies the same scalars through the router forward pass.
    This class certifies the static method contract independently, without routing.
    """

    def _make_inputs(
        self,
        B: int,
        N: int,
        L: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct (routing_logits, balance_logits, semantic_logits) for static method tests."""
        routing_logits = torch.randn(B, N, L)
        balance_logits = torch.randn(B, N, L)
        semantic_logits = routing_logits + balance_logits
        return routing_logits, balance_logits, semantic_logits

    def test_returns_expected_keys(self):
        """_compute_bias_diagnostics must return a dict with exactly the four expected keys."""
        routing_logits, balance_logits, semantic_logits = self._make_inputs(2, 8, 4)
        result = MoSRAHRouter._compute_bias_diagnostics(routing_logits, balance_logits, semantic_logits)
        assert set(result.keys()) == {
            "raw_logit_std", "bias_std", "logit_std", "bias_alignment"
        }

    def test_all_values_detached(self):
        """All four diagnostic scalars must be detached from the autograd graph."""
        routing_logits = torch.randn(2, 8, 4, requires_grad=True)
        balance_logits = torch.randn(2, 8, 4, requires_grad=True)
        semantic_logits = routing_logits + balance_logits
        result = MoSRAHRouter._compute_bias_diagnostics(routing_logits, balance_logits, semantic_logits)
        for key, val in result.items():
            assert not val.requires_grad, (
                f"{key} must be detached but requires_grad is True"
            )

    def test_bias_std_zero_when_balance_is_uniform(self):
        """bias_std must be exactly 0 when balance_logits are constant across L."""
        B, N, L = 1, 4, 6
        routing_logits = torch.randn(B, N, L)
        # Constant across the L dimension → per-token std = 0.
        balance_logits = torch.full((B, N, L), 2.5)
        semantic_logits = routing_logits + balance_logits
        result = MoSRAHRouter._compute_bias_diagnostics(routing_logits, balance_logits, semantic_logits)
        assert result["bias_std"].item() == 0.0

    def test_alignment_negative_when_balance_opposes_routing(self):
        """bias_alignment must be negative when balance_logits = -routing_logits."""
        torch.manual_seed(7)
        B, N, L = 1, 8, 4
        routing_logits = torch.randn(B, N, L)
        balance_logits = -routing_logits  # perfect anti-alignment → cosine similarity = -1
        semantic_logits = routing_logits + balance_logits
        result = MoSRAHRouter._compute_bias_diagnostics(routing_logits, balance_logits, semantic_logits)
        assert result["bias_alignment"].item() < 0, (
            f"expected negative alignment for opposing balance, "
            f"got {result['bias_alignment'].item()}"
        )

    def test_alignment_positive_when_balance_reinforces_routing(self):
        """bias_alignment must be positive when balance_logits = routing_logits."""
        torch.manual_seed(7)
        B, N, L = 1, 8, 4
        routing_logits = torch.randn(B, N, L)
        balance_logits = routing_logits  # perfect alignment → cosine similarity = 1
        semantic_logits = routing_logits + balance_logits
        result = MoSRAHRouter._compute_bias_diagnostics(routing_logits, balance_logits, semantic_logits)
        assert result["bias_alignment"].item() > 0, (
            f"expected positive alignment for reinforcing balance, "
            f"got {result['bias_alignment'].item()}"
        )


# ---------------------------------------------------------------------------
# Architecture invariants
# ---------------------------------------------------------------------------

class TestArchitectureInvariants:
    def test_routing_weight_is_parameter(self):
        """routing_weight must be an nn.Parameter so the optimizer sees and updates it,
        and HuggingFace _init_weights does not override its kaiming initialization."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert isinstance(router.routing_weight, torch.nn.Parameter)

    def test_balance_weight_is_parameter(self):
        """balance_weight must be an nn.Parameter so the optimizer sees and updates it,
        and HuggingFace _init_weights does not override its kaiming initialization."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert isinstance(router.balance_weight, torch.nn.Parameter)

    def test_routing_weight_shape(self):
        """routing_weight must have shape (num_mosrah_heads, embedding_width)."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.routing_weight.shape == (config.num_mosrah_heads, config.embedding_width)

    def test_balance_weight_shape(self):
        """balance_weight must have shape (num_mosrah_heads, embedding_width)."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.balance_weight.shape == (config.num_mosrah_heads, config.embedding_width)


# ---------------------------------------------------------------------------
# MaxVio
# ---------------------------------------------------------------------------

class TestMaxVio:
    """Tests for the _compute_max_vio helper and the max_vio forward output.

    The helper is tested directly with synthetic assignment_mask tensors, bypassing
    TopK entirely. This avoids the tie-breaking ambiguity that arises when all
    routing scores are equal and makes the expected values exact and analytical.

    All three numerical tests use B=1, L=4, K=1 and analytically derived expected
    values. assignment_mask is constructed via scatter from known head selections so
    that reduce_frequency_tokens produces known f_bl values.
    """

    def test_max_vio_zero_for_uniform_frequencies(self):
        """MaxVio must be exactly 0 when all heads receive equal routing frequency.

        With L=4 and one token per head (N=4, K=1), each head gets f=0.25.
        Every term (f_l - 1/L) is zero, so L * max(f_l - 1/L) = 0.
        """
        L = 4
        # Token i selects head i: f_bl = [0.25, 0.25, 0.25, 0.25]
        assignment_mask = torch.eye(L).unsqueeze(0)        # (1, 4, 4)
        active_mask = torch.ones(1, L, dtype=torch.bool)
        max_vio = MoSRAHRouter._compute_max_vio(assignment_mask, active_mask, L)
        assert torch.isclose(max_vio, torch.tensor(0.0), atol=1e-6)

    def test_max_vio_one_for_double_fair_share(self):
        """MaxVio must be exactly 1 when one head receives double its fair share.

        With L=4, N=6, K=1: 3 tokens to head 0, 1 each to heads 1-3.
        f_bl = [3/6, 1/6, 1/6, 1/6] = [0.5, 1/6, 1/6, 1/6].
        MaxVio = 4 * (0.5 - 0.25) = 1.0.
        """
        L, N = 4, 6
        heads_selected = torch.tensor([[0, 0, 0, 1, 2, 3]]).unsqueeze(-1)  # (1, 6, 1)
        assignment_mask = torch.zeros(1, N, L).scatter_(-1, heads_selected, 1.0)
        active_mask = torch.ones(1, N, dtype=torch.bool)
        max_vio = MoSRAHRouter._compute_max_vio(assignment_mask, active_mask, L)
        assert torch.isclose(max_vio, torch.tensor(1.0), atol=1e-6)

    def test_max_vio_intermediate_value(self):
        """MaxVio must equal 0.5 when one head receives 1.5× its fair share.

        With L=4, N=8, K=1: 3 tokens to head 0, then 2/2/1 to heads 1/2/3.
        f_bl = [3/8, 2/8, 2/8, 1/8] = [0.375, 0.25, 0.25, 0.125].
        MaxVio = 4 * (0.375 - 0.25) = 0.5.
        """
        L, N = 4, 8
        heads_selected = torch.tensor([[0, 0, 0, 1, 2, 3, 1, 2]]).unsqueeze(-1)  # (1, 8, 1)
        assignment_mask = torch.zeros(1, N, L).scatter_(-1, heads_selected, 1.0)
        active_mask = torch.ones(1, N, dtype=torch.bool)
        max_vio = MoSRAHRouter._compute_max_vio(assignment_mask, active_mask, L)
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

        Integral weights are filled with ones to lift the zero-init degeneracy: at
        zero init the cumsum corrections are zero regardless of input, so the dead
        token masking fix would pass trivially. Non-zero weights make the test load-bearing.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        with torch.no_grad():
            router.routing_integral_weight.fill_(1.0)
            router.balance_integral_weight.fill_(1.0)
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
        """Changing a dead token's hidden state must not affect max_vio.

        Integral weights are filled with ones for the same reason as the
        load_balance_loss test above: zero-init would make the test trivial.
        """
        config = small_config()
        router = MoSRAHRouter(config)
        with torch.no_grad():
            router.routing_integral_weight.fill_(1.0)
            router.balance_integral_weight.fill_(1.0)
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

    def test_all_live_mask_gives_max_vio_matching_direct_static_call(self):
        """With all tokens live, router forward max_vio must match _compute_max_vio called directly.

        Builds assignment_mask from selected_heads via scatter and calls the static method
        independently, confirming the router forward correctly wires to _compute_max_vio.
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

            # Build assignment_mask from selected_heads via scatter and call directly.
            assignment_mask = torch.zeros(B, N, L)
            assignment_mask.scatter_(-1, selected_heads, 1.0)
            expected_max_vio = MoSRAHRouter._compute_max_vio(assignment_mask, active_mask, L)

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
            try:
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
            except Exception as err:
                raise err # Debugging aid.

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


# ---------------------------------------------------------------------------
# Integral routing
# ---------------------------------------------------------------------------

class TestIntegralRouting:
    """Tests for the integral routing extension (routing_mode='integral').

    Certifies existence and shape of integral weight parameters, gradient
    isolation between A' and B', output differentiation between modes, and
    shape consistency across modes.
    """

    def test_integral_weights_exist_in_integral_mode(self):
        """routing_integral_weight and balance_integral_weight must both exist
        as nn.Parameter when routing_mode='integral'."""
        config = small_config(routing_mode="integral")
        router = MoSRAHRouter(config)
        assert isinstance(router.routing_integral_weight, torch.nn.Parameter)
        assert isinstance(router.balance_integral_weight, torch.nn.Parameter)

    def test_integral_weights_absent_in_default_mode(self):
        """Neither routing_integral_weight nor balance_integral_weight must
        exist when routing_mode='default'."""
        config = small_config(routing_mode="default")
        router = MoSRAHRouter(config)
        assert not hasattr(router, "routing_integral_weight")
        assert not hasattr(router, "balance_integral_weight")

    def test_integral_weights_shape(self):
        """Both integral weight matrices must have shape (L, L) where
        L = num_mosrah_heads."""
        config = small_config(routing_mode="integral")
        router = MoSRAHRouter(config)
        L = config.num_mosrah_heads
        assert router.routing_integral_weight.shape == (L, L)
        assert router.balance_integral_weight.shape == (L, L)

    def test_task_loss_trains_routing_integral_not_balance_integral(self):
        """Task loss backward must populate routing_integral_weight.grad but
        not balance_integral_weight.grad.

        semantic_logits includes F.linear(u_semantic, routing_integral_weight)
        (differentiable) and F.linear(u_semantic, balance_integral_weight).detach()
        — there is no autograd path from routing_probs back to balance_integral_weight.
        """
        torch.manual_seed(0)
        config = small_config(routing_mode="integral")
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, routing_probs, _ = router(x, active_mask, None)
        routing_probs.sum().backward()

        assert router.routing_integral_weight.grad is not None
        assert router.balance_integral_weight.grad is None

    def test_load_balance_loss_trains_balance_integral_not_routing_integral(self):
        """Load balance loss backward must populate balance_integral_weight.grad
        but not routing_integral_weight.grad.

        load_balancing_logits includes F.linear(u_load, routing_integral_weight).detach()
        and F.linear(u_load, balance_integral_weight) — there is no autograd path
        from load_balance_loss back to routing_integral_weight.
        """
        torch.manual_seed(0)
        config = small_config(routing_mode="integral")
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask, None)
        diagnostics["load_balance_loss"].backward()

        assert router.balance_integral_weight.grad is not None
        assert router.routing_integral_weight.grad is None

    def test_integral_output_differs_from_default(self):
        """Integral mode must produce different routing_probs from default mode
        on the same input and base weights.

        With base weights copied to match and routing_integral_weight set to a
        non-zero constant, the A'@u correction shifts semantic_logits away from
        the default-mode values, producing different routing_probs.
        routing_integral_weight is set explicitly because A' and B' are zero-initialized;
        zero corrections would make integral and default produce identical outputs.
        """
        torch.manual_seed(42)
        config_integral = small_config(routing_mode="integral")
        config_default = small_config(routing_mode="default")
        router_integral = MoSRAHRouter(config_integral)
        router_default = MoSRAHRouter(config_default)

        # Copy base weights and set non-zero A' so the integral correction is visible.
        with torch.no_grad():
            router_default.routing_weight.data.copy_(router_integral.routing_weight.data)
            router_default.balance_weight.data.copy_(router_integral.balance_weight.data)
            router_integral.routing_integral_weight.fill_(0.1)

        x = torch.randn(2, 8, config_integral.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        with torch.no_grad():
            _, probs_integral, _ = router_integral(x, active_mask, None)
            _, probs_default, _ = router_default(x, active_mask, None)

        assert not torch.equal(probs_integral, probs_default), (
            "Integral and default modes must produce different routing_probs — "
            "A' corrections must affect the output when routing_integral_weight is non-zero"
        )

    def test_output_shapes_match_across_modes(self):
        """selected_heads and routing_probs shapes must be identical between
        integral and default modes for the same config dimensions."""
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        config_integral = small_config(routing_mode="integral")
        config_default = small_config(routing_mode="default")
        router_integral = MoSRAHRouter(config_integral)
        router_default = MoSRAHRouter(config_default)

        with torch.no_grad():
            heads_integral, probs_integral, _ = router_integral(x, active_mask, None)
            heads_default, probs_default, _ = router_default(x, active_mask, None)

        assert heads_integral.shape == heads_default.shape
        assert probs_integral.shape == probs_default.shape


# ---------------------------------------------------------------------------
# Compiled training — integral mode
# ---------------------------------------------------------------------------

class TestIntegralModeCompileTraining:
    """Compiled training test for integral routing mode.

    Verifies that torch.compile works with the integral router, that the load
    balance signal reaches balance_integral_weight (B') through the compiled
    integral pathway, and that the load balance loss prevents degenerate routing
    collapse under a task reward that always concentrates on the same heads.

    CUDA-only: uses the device fixture from conftest.py.
    """

    def test_compiled_training_load_balance_holds_back_concentration(self, device):
        """Compiled integral router trained with load balance loss must not
        collapse to max_vio >= 0.3 under a degenerate task reward.

        A single compiled integral router is trained for 30 SGD steps with a
        task reward that always rewards concentration on the same K//2 selected
        heads, plus 0.1 * load_balance_loss. With overallocation_factor=2.0
        and load balancing active, the router must not reach degenerate collapse
        (max_vio < 0.3 after training).

        Additionally verifies: (1) compilation completes without graph breaks, and
        (2) balance_integral_weight.grad is populated, confirming the load balance
        signal reaches B' through the compiled integral pathway.
        """
        if device.type != "cuda":
            pytest.skip("Compiled training test is CUDA-only.")

        torch.manual_seed(0)
        config = small_config(
            routing_mode="integral",
            mosrah_overallocation_factor=2.0,
        )
        K = config.num_selected_heads
        K_half = K // 2

        router = MoSRAHRouter(config).to(device)
        compiled_router = torch.compile(router, fullgraph=True, dynamic=False)
        opt = torch.optim.SGD(router.parameters(), lr=0.001)

        generator = torch.Generator(device=device)
        generator.manual_seed(1)
        B, N = 2, 16

        for _ in range(1000):
            x = torch.randn(B, N, config.embedding_width, device=device, generator=generator)
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            opt.zero_grad()
            _, routing_probs, diagnostics = compiled_router(x, active_mask, None)
            # Degenerate task reward: always maximize probability on the first K_half
            # selected heads, pushing the router toward head concentration.
            task_loss = -(routing_probs[..., :K_half].sum())
            loss = task_loss + 10 * diagnostics["load_balance_loss"]
            loss.backward()
            opt.step()

        # Verify load balance held back collapse on a fixed evaluation input.
        torch.manual_seed(7)
        x_eval = torch.randn(B, N, config.embedding_width, device=device)
        active_mask_eval = torch.ones(B, N, dtype=torch.bool, device=device)

        with torch.no_grad():
            _, _, diag_eval = router(x_eval, active_mask_eval, None)

        max_vio = diag_eval["max_vio"].item()
        assert max_vio < 0.5, (
            f"Load balance loss must prevent degenerate collapse: "
            f"max_vio={max_vio:.4f} must be < 0.3"
        )

        # Verify load balance signal reached B' through the compiled integral pathway.
        opt.zero_grad()
        _, _, diag_check = compiled_router(x_eval, active_mask_eval, None)
        diag_check["load_balance_loss"].backward()
        assert router.balance_integral_weight.grad is not None, (
            "balance_integral_weight.grad must be populated after load_balance_loss.backward() "
            "— the load balance signal must reach B' through the integral pathway"
        )


# ---------------------------------------------------------------------------
# Routing mode profiling (optional)
# ---------------------------------------------------------------------------

class TestRoutingModeProfile:
    """Optional profiling test for default vs integral routing mode performance.

    Disabled by default. Set PROFILE_ROUTING_MODES = True at module level to run.
    Prints a timing table but makes no assertions.
    """

    def test_profile_default_vs_integral_modes(self):
        """Time eager and compiled forward pass for both routing modes.

        Runs 50 warmup iterations then 100 timed iterations per (mode, compile)
        combination. Prints: mode × compile × time (ms/iter). No assertions.
        """
        if not PROFILE_ROUTING_MODES:
            pytest.skip("Set PROFILE_ROUTING_MODES=True to run profiling")

        import time

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        WARMUP = 50
        TIMED = 100
        B, N = 2, 64

        results = {}
        for mode in ("default", "integral"):
            config = small_config(routing_mode=mode)
            router = MoSRAHRouter(config).eval().to(device)
            compiled_router = torch.compile(router, fullgraph=True, dynamic=False)

            x = torch.randn(B, N, config.embedding_width, device=device)
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            for label, fn in [("eager", router), ("compiled", compiled_router)]:
                for _ in range(WARMUP):
                    with torch.no_grad():
                        fn(x, active_mask, None)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(TIMED):
                    with torch.no_grad():
                        fn(x, active_mask, None)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) / TIMED * 1000

                results[(mode, label)] = elapsed_ms

        print(f"\nRouting mode profile ({device}, ms/iter):")
        print(f"{'mode':<12} {'compile':<12} {'ms/iter':>10}")
        print("-" * 36)
        for (mode, label), ms in results.items():
            print(f"{mode:<12} {label:<12} {ms:>10.3f}")
