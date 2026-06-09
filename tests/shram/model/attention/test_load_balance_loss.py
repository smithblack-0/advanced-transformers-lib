"""Tests for the load-balance loss factory and the three loss formulations.

Invariants verified:
- reduce_frequency_tokens produces correct per-batch-item routing frequencies, is detached,
  excludes dead tokens, and returns zeros on the all-dead-tokens edge case
- reduce_probability_tokens produces correct per-batch-item mean softmax probabilities,
  carries gradient from logits, excludes dead tokens, and returns zeros on the
  all-dead-tokens edge case
- Factory returns a callable for each of the four valid type strings
- Factory raises ValueError for an invalid type string
- temporal_overcapacity_loss is exactly zero when no head exceeds its allowed trajectory
- temporal_overcapacity_loss computes the correct correction moment for one or more violating heads
- Inactive tokens are excluded from prior counts and batch reduction
- gshard_loss computes (1/L) * Σ_l f_bl * p_bl per batch item, averaged over B
- ce_loss computes -(1/(L-1)) * Σ_l (1-f_bl) * log(p_bl) per batch item, averaged over B
- bce_loss computes -(1/L) * Σ_l [(1-f_bl)*log(p_bl) + f_bl*log(1-p_bl)] per batch item,
  averaged over B
- All three formulations return a scalar (zero-dimensional tensor)
- Per-batch-item computation: complementary imbalances across batch items are penalised
  independently and do not cancel before the loss signal is formed
- logits receives gradient after backward through any formulation; assignment_mask does not
"""

import math

import pytest
import torch

from src.shram.model.attention.load_balance_loss import (
    make_load_balance_loss,
    reduce_frequency_tokens,
    reduce_probability_tokens,
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def reference_softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax for Python reference calculations."""
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def reference_frequencies(am_row: list[float]) -> list[float]:
    """Routing frequencies for a B=1, N=1 input: f_l = am_l / sum(am)."""
    total = max(sum(am_row), 1.0)
    return [v / total for v in am_row]


def reference_gshard(f: list[float], p: list[float]) -> float:
    """gshard formula: (1/L) * Σ_l f_l * p_l."""
    L = len(f)
    return sum(fi * pi for fi, pi in zip(f, p)) / L


def reference_ce(f: list[float], p: list[float]) -> float:
    """CE formula: -(1/(L-1)) * Σ_l (1-f_l) * log(p_l)."""
    L = len(f)
    return -sum((1.0 - fi) * math.log(pi) for fi, pi in zip(f, p)) / (L - 1)


def reference_bce(f: list[float], p: list[float]) -> float:
    """BCE formula: -(1/L) * Σ_l [(1-f_l)*log(p_l) + f_l*log1p(-p_l)]."""
    L = len(f)
    return -sum(
        (1.0 - fi) * math.log(pi) + fi * math.log1p(-pi)
        for fi, pi in zip(f, p)
    ) / L


# ---------------------------------------------------------------------------
# Input constructors
# ---------------------------------------------------------------------------

# Shared test inputs: L=4, heads 0 and 1 selected, known logits.
_LOGITS_ROW: list[float] = [1.0, 2.0, -1.0, 0.5]
_AM_ROW:     list[float] = [1.0, 1.0, 0.0, 0.0]


def make_single_token_inputs(
    logits_row: list[float] = _LOGITS_ROW,
    am_row: list[float] = _AM_ROW,
    active: bool = True,
    logits_requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct B=1, N=1 tensors for single-token numerical tests.

    Returns:
        logits:          shape (1, 1, L), raw pre-softmax routing values
        assignment_mask: shape (1, 1, L), head-assignment indicators
        active_mask:     shape (1, 1), dtype bool
    """
    logits = torch.tensor(
        [[logits_row]], dtype=torch.float32, requires_grad=logits_requires_grad,
    )
    assignment_mask = torch.tensor([[am_row]], dtype=torch.float32)
    active_mask = torch.tensor([[active]], dtype=torch.bool)
    return logits, assignment_mask, active_mask


# ---------------------------------------------------------------------------
# TestReduceFrequency
# ---------------------------------------------------------------------------

class TestReduceFrequency:
    """Verify reduce_frequency_tokens produces correct per-batch-item routing frequencies."""

    def test_single_active_token(self) -> None:
        """Must compute f_bl as the fraction of active assignments going to each head."""
        _, assignment_mask, active_mask = make_single_token_inputs()

        f_bl = reduce_frequency_tokens(assignment_mask, active_mask)

        # am=[1,1,0,0]: 2 total assignments → f=[0.5, 0.5, 0, 0].
        expected = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        assert f_bl.shape == (1, 4)
        assert torch.allclose(f_bl, expected, atol=1e-6)

    def test_multi_token_pools_across_tokens(self) -> None:
        """Must sum assignments across all active tokens before normalising."""
        # B=1, N=2: token 0 → heads 0,1; token 1 → heads 2,3.
        assignment_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 1.0]]])  # (1, 2, 4)
        active_mask = torch.ones(1, 2, dtype=torch.bool)

        f_bl = reduce_frequency_tokens(assignment_mask, active_mask)

        # 4 total assignments, one each → f=[0.25, 0.25, 0.25, 0.25].
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(f_bl, expected, atol=1e-6)

    def test_dead_tokens_excluded(self) -> None:
        """Must not count inactive tokens' assignments in routing frequency statistics."""
        # B=1, N=2: token 0 is dead, token 1 → heads 0 and 2.
        assignment_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0],
                                         [1.0, 0.0, 1.0, 0.0]]])  # (1, 2, 4)
        active_mask = torch.tensor([[False, True]], dtype=torch.bool)

        f_bl = reduce_frequency_tokens(assignment_mask, active_mask)

        # Only token 1 is active: am=[1,0,1,0] → f=[0.5, 0, 0.5, 0].
        expected = torch.tensor([[0.5, 0.0, 0.5, 0.0]])
        assert torch.allclose(f_bl, expected, atol=1e-6)

    def test_output_is_detached(self) -> None:
        """Must return a detached tensor — routing frequencies carry no gradient."""
        _, assignment_mask, active_mask = make_single_token_inputs()

        f_bl = reduce_frequency_tokens(assignment_mask, active_mask)

        assert not f_bl.requires_grad

    def test_all_dead_tokens_returns_zeros(self) -> None:
        """Must return zeros (not NaN) when all tokens in a batch item are inactive."""
        _, assignment_mask, _ = make_single_token_inputs()
        all_dead = torch.tensor([[False]], dtype=torch.bool)

        f_bl = reduce_frequency_tokens(assignment_mask, all_dead)

        assert f_bl.shape == (1, 4)
        assert torch.all(f_bl == 0.0)
        assert not f_bl.isnan().any()


# ---------------------------------------------------------------------------
# TestReduceProbability
# ---------------------------------------------------------------------------

class TestReduceProbability:
    """Verify reduce_probability_tokens produces correct per-batch-item mean softmax probs."""

    def test_single_active_token(self) -> None:
        """Must equal softmax of the single token's logits when N=1."""
        logits, _, active_mask = make_single_token_inputs()

        p_bl = reduce_probability_tokens(logits, active_mask)

        # With a single active token the mean is just that token's softmax.
        expected_p = reference_softmax(_LOGITS_ROW)
        assert p_bl.shape == (1, 4)
        for l, exp_val in enumerate(expected_p):
            assert p_bl[0, l].item() == pytest.approx(exp_val, abs=1e-6), (
                f"head {l}: expected {exp_val:.8f}, got {p_bl[0, l].item():.8f}"
            )

    def test_multi_token_mean(self) -> None:
        """Must return the mean softmax probability over all active tokens per batch item."""
        # B=1, N=2: two active tokens with known distinct logits.
        logits = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0]]])  # (1, 2, 4)
        active_mask = torch.ones(1, 2, dtype=torch.bool)

        p_bl = reduce_probability_tokens(logits, active_mask)

        p0 = reference_softmax([1.0, 0.0, 0.0, 0.0])
        p1 = reference_softmax([0.0, 1.0, 0.0, 0.0])
        expected = [(p0[l] + p1[l]) / 2.0 for l in range(4)]
        for l, exp_val in enumerate(expected):
            assert p_bl[0, l].item() == pytest.approx(exp_val, abs=1e-6), (
                f"head {l}: expected {exp_val:.8f}, got {p_bl[0, l].item():.8f}"
            )

    def test_dead_tokens_excluded(self) -> None:
        """Must compute the mean over active tokens only, ignoring inactive positions."""
        # B=1, N=2: token 0 is dead with extreme logits; token 1 has the test logits.
        # If token 0 were included it would dominate the mean and make the test fail.
        logits = torch.tensor([[[9.0, -9.0, 0.0, 0.0],
                                 [1.0, 2.0, -1.0, 0.5]]])  # (1, 2, 4)
        active_mask = torch.tensor([[False, True]], dtype=torch.bool)

        p_bl = reduce_probability_tokens(logits, active_mask)

        # Only token 1 is active; its softmax is the full per-item result.
        expected_p = reference_softmax([1.0, 2.0, -1.0, 0.5])
        for l, exp_val in enumerate(expected_p):
            assert p_bl[0, l].item() == pytest.approx(exp_val, abs=1e-6), (
                f"head {l}: expected {exp_val:.8f}, got {p_bl[0, l].item():.8f}"
            )

    def test_gradient_flows_through_output(self) -> None:
        """Must preserve the gradient path from logits through the output tensor."""
        logits, _, active_mask = make_single_token_inputs(logits_requires_grad=True)

        p_bl = reduce_probability_tokens(logits, active_mask)

        assert p_bl.requires_grad

    def test_all_dead_tokens_returns_zeros(self) -> None:
        """Must return zeros (not NaN) when all tokens in a batch item are inactive."""
        logits, _, _ = make_single_token_inputs()
        all_dead = torch.tensor([[False]], dtype=torch.bool)

        p_bl = reduce_probability_tokens(logits, all_dead)

        assert p_bl.shape == (1, 4)
        assert torch.all(p_bl == 0.0)
        assert not p_bl.isnan().any()


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------

class TestFactory:
    """Verify the make_load_balance_loss factory returns the correct callable or raises."""

    def test_returns_callable_for_gshard(self) -> None:
        """Factory must return a callable for loss_type='gshard'."""
        fn = make_load_balance_loss("gshard")
        assert callable(fn)

    def test_returns_callable_for_ce(self) -> None:
        """Factory must return a callable for loss_type='ce'."""
        fn = make_load_balance_loss("ce")
        assert callable(fn)

    def test_returns_callable_for_bce(self) -> None:
        """Factory must return a callable for loss_type='bce'."""
        fn = make_load_balance_loss("bce")
        assert callable(fn)

    def test_raises_for_invalid_type(self) -> None:
        """Factory must raise ValueError for an unrecognised loss_type."""
        with pytest.raises(ValueError, match="load_balance_loss_type"):
            make_load_balance_loss("invalid_type")

    def test_gshard_ignores_temporal_kwargs(self) -> None:
        """gshard factory must accept and silently ignore temporal-specific kwargs."""
        fn = make_load_balance_loss(
            "gshard", num_selected_heads=2, num_total_heads=4, maximum_expert_overclaim=5,
        )
        assert callable(fn)
        logits, assignment_mask, active_mask = make_single_token_inputs()
        loss = fn(logits, assignment_mask, active_mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_ce_ignores_temporal_kwargs(self) -> None:
        """ce factory must accept and silently ignore temporal-specific kwargs."""
        fn = make_load_balance_loss(
            "ce", num_selected_heads=2, num_total_heads=4, maximum_expert_overclaim=5,
        )
        assert callable(fn)
        logits, assignment_mask, active_mask = make_single_token_inputs()
        loss = fn(logits, assignment_mask, active_mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_bce_ignores_temporal_kwargs(self) -> None:
        """bce factory must accept and silently ignore temporal-specific kwargs."""
        fn = make_load_balance_loss(
            "bce", num_selected_heads=2, num_total_heads=4, maximum_expert_overclaim=5,
        )
        assert callable(fn)
        logits, assignment_mask, active_mask = make_single_token_inputs()
        loss = fn(logits, assignment_mask, active_mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_returns_callable_for_temporal_overcapacity(self) -> None:
        """Factory must return a callable for loss_type='temporal_overcapacity'."""
        fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=4, maximum_expert_overclaim=5,
        )
        assert callable(fn)
        logits, assignment_mask, active_mask = make_single_token_inputs()
        loss = fn(logits, assignment_mask, active_mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TestFormulas
# ---------------------------------------------------------------------------

class TestFormulas:
    """Verify each formulation against a Python reference on known B=1, N=1 inputs.

    Uses B=1, N=1 so f and p are determined directly by the single token's assignment_mask
    and the softmax of its logits. Reference functions are direct Python translations of
    the plan-entry formulas.
    """

    def _verify_formula(
        self,
        loss_type: str,
        logits_row: list[float],
        am_row: list[float],
        expected: float,
    ) -> None:
        """Call the loss function and assert the output matches expected to 1e-5."""
        logits, assignment_mask, active_mask = make_single_token_inputs(
            logits_row=logits_row, am_row=am_row,
        )
        loss_fn = make_load_balance_loss(loss_type)
        loss = loss_fn(logits, assignment_mask, active_mask)

        assert loss.shape == (), (
            f"{loss_type}: expected scalar output, got shape {loss.shape}"
        )
        assert loss.item() == pytest.approx(expected, abs=1e-5), (
            f"{loss_type}: expected {expected:.8f}, got {loss.item():.8f} "
            f"with logits={logits_row}, am={am_row}"
        )

    # gshard -----------------------------------------------------------------

    def test_gshard_formula_on_known_inputs(self) -> None:
        """gshard_loss must compute (1/L) * Σ_l f_l * p_l on known inputs."""
        f = reference_frequencies(_AM_ROW)
        p = reference_softmax(_LOGITS_ROW)
        self._verify_formula("gshard", _LOGITS_ROW, _AM_ROW, reference_gshard(f, p))

    def test_gshard_formula_at_balanced_routing(self) -> None:
        """gshard_loss must equal 1/L^2 when all heads have equal frequency and uniform logits."""
        # K=L=4: every head selected; zero logits → uniform softmax → f=p=1/L.
        am  = [1.0, 1.0, 1.0, 1.0]
        lgs = [0.0, 0.0, 0.0, 0.0]
        f = reference_frequencies(am)
        p = reference_softmax(lgs)
        self._verify_formula("gshard", lgs, am, reference_gshard(f, p))

    # ce ---------------------------------------------------------------------

    def test_ce_formula_on_known_inputs(self) -> None:
        """ce_loss must compute -(1/(L-1)) * Σ_l (1-f_l)*log(p_l) on known inputs."""
        f = reference_frequencies(_AM_ROW)
        p = reference_softmax(_LOGITS_ROW)
        self._verify_formula("ce", _LOGITS_ROW, _AM_ROW, reference_ce(f, p))

    def test_ce_formula_at_balanced_routing(self) -> None:
        """ce_loss must produce the correct symmetric value when f_l = p_l = 1/L."""
        am  = [1.0, 1.0, 1.0, 1.0]
        lgs = [0.0, 0.0, 0.0, 0.0]
        f = reference_frequencies(am)
        p = reference_softmax(lgs)
        self._verify_formula("ce", lgs, am, reference_ce(f, p))

    # bce --------------------------------------------------------------------

    def test_bce_formula_on_known_inputs(self) -> None:
        """bce_loss must compute -(1/L)*Σ_l[(1-f_l)*log(p_l)+f_l*log(1-p_l)] on known inputs."""
        f = reference_frequencies(_AM_ROW)
        p = reference_softmax(_LOGITS_ROW)
        self._verify_formula("bce", _LOGITS_ROW, _AM_ROW, reference_bce(f, p))

    def test_bce_formula_at_balanced_routing(self) -> None:
        """bce_loss must produce the correct symmetric value when f_l = p_l = 1/L."""
        am  = [1.0, 1.0, 1.0, 1.0]
        lgs = [0.0, 0.0, 0.0, 0.0]
        f = reference_frequencies(am)
        p = reference_softmax(lgs)
        self._verify_formula("bce", lgs, am, reference_bce(f, p))


# ---------------------------------------------------------------------------
# TestBatchCorrectness
# ---------------------------------------------------------------------------

class TestBatchCorrectness:
    """Verify that loss is computed per batch item so complementary imbalances do not cancel.

    Uses B=2, N=1, L=4, K=1 where item 0 routes to head 0 and item 1 routes to head 3.
    Each item has extreme per-item imbalance. A global-frequency implementation would
    produce f=[0.5,0,0,0.5] and suppress the CE signal; per-item computation preserves
    the imbalance from each item independently.

    The two items are mirror-symmetric (reversed routing + reversed logits), so per-item
    CE values are equal and the expected mean equals either item computed individually.
    """

    def test_ce_per_batch_item_numerical(self) -> None:
        """ce_loss must compute per-item CE and average over B, not flatten B*N first."""
        assignment_mask = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0]],   # item 0: token routed to head 0
            [[0.0, 0.0, 0.0, 1.0]],   # item 1: token routed to head 3
        ])  # (2, 1, 4)
        logits = torch.tensor([
            [[0.4, 0.3, 0.2, 0.1]],
            [[0.1, 0.2, 0.3, 0.4]],   # mirror of item 0
        ])  # (2, 1, 4)
        active_mask = torch.ones(2, 1, dtype=torch.bool)

        loss_fn = make_load_balance_loss("ce")
        loss = loss_fn(logits, assignment_mask, active_mask)

        # Item 0: f=[1,0,0,0], p=softmax([0.4,0.3,0.2,0.1]).
        # Mirror symmetry: CE(item 1) == CE(item 0), so expected = CE(item 0).
        f0 = reference_frequencies([1.0, 0.0, 0.0, 0.0])
        p0 = reference_softmax([0.4, 0.3, 0.2, 0.1])
        expected = reference_ce(f0, p0)

        assert loss.shape == ()
        assert loss.item() == pytest.approx(expected, abs=1e-5), (
            f"ce batch: expected {expected:.8f}, got {loss.item():.8f}"
        )


# ---------------------------------------------------------------------------
# TestGradientIsolation
# ---------------------------------------------------------------------------

class TestGradientIsolation:
    """Verify that gradients flow to logits but not assignment_mask.

    logits must receive gradient; assignment_mask must not. assignment_mask comes
    from discrete TopK selections and carries no gradient.
    """

    def _test_isolation(self, loss_type: str) -> None:
        logits, assignment_mask, active_mask = make_single_token_inputs(
            logits_requires_grad=True,
        )
        loss_fn = make_load_balance_loss(loss_type)
        loss = loss_fn(logits, assignment_mask, active_mask)
        loss.backward()

        assert logits.grad is not None, (
            f"{loss_type}: logits must receive gradient"
        )
        assert torch.isfinite(logits.grad).all(), (
            f"{loss_type}: logits gradient must be finite"
        )
        assert assignment_mask.grad is None, (
            f"{loss_type}: assignment_mask must not receive gradient"
        )

    def test_gshard_gradient_flows_to_logits_only(self) -> None:
        """gshard_loss must propagate gradient to logits; assignment_mask is blocked."""
        self._test_isolation("gshard")

    def test_ce_gradient_flows_to_logits_only(self) -> None:
        """ce_loss must propagate gradient to logits; assignment_mask is blocked."""
        self._test_isolation("ce")

    def test_bce_gradient_flows_to_logits_only(self) -> None:
        """bce_loss must propagate gradient to logits; assignment_mask is blocked."""
        self._test_isolation("bce")



# ---------------------------------------------------------------------------
# TestTemporalOvercapacityFormulas
# ---------------------------------------------------------------------------

class TestTemporalOvercapacityFormulas:
    """Verify the temporal_overcapacity formulation against hand-calculated expected values.

    All tests go through make_load_balance_loss to exercise the full factory path.
    Each test uses a small tensor where imbalance state is fully determined by inspection,
    and the expected loss is derived analytically from the formula.
    """

    def test_no_violations_loss_is_zero(self) -> None:
        """Loss must be exactly zero when no head exceeds its allowed trajectory."""
        # B=1, N=3, L=2, K=1, C=10. Head 0 always selected. All active.
        # Max prior count = 2; threshold ≥ 10 at every position → no imbalance.
        assignment_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        logits = torch.zeros(1, 3, 2)
        active_mask = torch.ones(1, 3, dtype=torch.bool)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=2, maximum_expert_overclaim=10,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)

        assert loss.shape == ()
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_one_violating_expert(self) -> None:
        """Must compute correct loss moment for a single violating head."""
        # B=1, N=3, L=2, K=1, C=0. Head 0 always selected. All active.
        # n=2: prior=[2,0], S=3, threshold=1.5 → head 0 violates (2 > 1.5).
        # violation_count=1, non_overloaded_count=1 (head 1).
        # loss_moment[n=2] = 2.0/1 − 0.0/1 = 2.0; active_count=3.
        # Expected loss = 2.0 / 3.
        assignment_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]])
        active_mask = torch.ones(1, 3, dtype=torch.bool)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=2, maximum_expert_overclaim=0,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)

        assert loss.shape == ()
        assert loss.item() == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_multiple_violating_experts(self) -> None:
        """Must average loss moment across multiple violating heads."""
        # B=1, N=4, L=3, K=2, C=0. Heads 0,1 always selected. All active.
        # n=3: prior=[3,3,0], S=4, threshold=4*(2/3)≈2.667 → heads 0,1 both violate.
        # violation_count=2, non_overloaded_count=1 (head 2).
        # loss_moment[n=3] = (3.0+1.0)/2 − 0.0/1 = 2.0; active_count=4.
        # Expected loss = 2.0 / 4 = 0.5.
        assignment_mask = torch.tensor([[
            [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ]])  # (1, 4, 3)
        logits = torch.tensor([[
            [3.0, 1.0, 0.0], [3.0, 1.0, 0.0],
            [3.0, 1.0, 0.0], [3.0, 1.0, 0.0],
        ]])  # (1, 4, 3)
        active_mask = torch.ones(1, 4, dtype=torch.bool)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=2, num_total_heads=3, maximum_expert_overclaim=0,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)

        assert loss.shape == ()
        assert loss.item() == pytest.approx(0.5, abs=1e-5)

    def test_inactive_tokens_excluded(self) -> None:
        """Dead tokens must not contribute to prior counts or batch reduction."""
        # B=1, N=4, L=2, K=1, C=0. Token n=1 inactive. Head 0 in assignment_mask for all n.
        # With n=1 dead, prior counts at n=3 reflect only n=0 and n=2: prior[n=3,head0]=2.
        # S[n=3]=3 (n=0,2,3 active) → threshold=1.5 → 2>1.5 → violation at n=3.
        # n=1 fires imbalance too but active_float[n=1]=0 zeros its contribution.
        # loss_moment[n=3] = 2.0/1 − 0.0/1 = 2.0; active_count=3.
        # Expected loss = 2.0 / 3.
        assignment_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]])
        active_mask = torch.tensor([[True, False, True, True]])
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=2, maximum_expert_overclaim=0,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)

        assert loss.shape == ()
        assert loss.item() == pytest.approx(2.0 / 3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestTemporalOvercapacityGradient
# ---------------------------------------------------------------------------

class TestTemporalOvercapacityGradient:
    """Verify gradient paths for the temporal_overcapacity formulation."""

    def test_gradient_flows_to_logits(self) -> None:
        """logits must receive a finite gradient after backward."""
        # Uses the one-violating-expert scenario to guarantee non-zero loss.
        logits = torch.tensor(
            [[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]], requires_grad=True,
        )
        assignment_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        active_mask = torch.ones(1, 3, dtype=torch.bool)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=2, maximum_expert_overclaim=0,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)
        loss.backward()

        assert logits.grad is not None, "logits must receive gradient"
        assert torch.isfinite(logits.grad).all(), "logits gradient must be finite"

    def test_assignment_mask_receives_no_gradient(self) -> None:
        """assignment_mask must not receive gradient — it comes from discrete TopK."""
        logits = torch.tensor(
            [[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]], requires_grad=True,
        )
        assignment_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        active_mask = torch.ones(1, 3, dtype=torch.bool)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=2, maximum_expert_overclaim=0,
        )

        loss = loss_fn(logits, assignment_mask, active_mask)
        loss.backward()

        assert assignment_mask.grad is None, "assignment_mask must not receive gradient"


# ---------------------------------------------------------------------------
# TestTemporalOvercapacityCompile
# ---------------------------------------------------------------------------

class TestTemporalOvercapacityCompile:
    """Verify the temporal_overcapacity factory callable is compatible with torch.compile."""

    def test_compile_forward_backward(self, device: str) -> None:
        """torch.compile must run forward and backward without error on CUDA."""
        if device == "cpu":
            pytest.skip("Compile test requires CUDA")

        torch.manual_seed(0)
        B, N, L = 1, 4, 2
        logits = torch.randn(B, N, L, device=device, requires_grad=True)
        assignment_mask = torch.zeros(B, N, L, device=device)
        assignment_mask[:, :, 0] = 1.0
        active_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        loss_fn = make_load_balance_loss(
            "temporal_overcapacity",
            num_selected_heads=1, num_total_heads=L, maximum_expert_overclaim=0,
        )

        compiled = torch.compile(loss_fn, fullgraph=True, dynamic=False)
        loss = compiled(logits, assignment_mask, active_mask)
        loss.backward()

        assert logits.grad is not None
