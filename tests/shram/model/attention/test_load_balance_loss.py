"""Tests for the load-balance loss factory and the three loss formulations.

Invariants verified:
- Factory returns a callable for each of the three valid type strings
- Factory raises ValueError for an invalid type string
- gshard_loss computes (1/L) * Σ_i f_i * p_i on known inputs
- ce_loss computes -(1/(L-1)) * Σ_i (1 - f_i) * log(p_i) on known inputs
- bce_loss computes -(1/L) * Σ_i [(1-f_i)*log(p_i) + f_i*log1p(-p_i)] on known inputs
- All three formulations return a scalar (zero-dimensional tensor)
- assignment_probs receives gradient after backward through any formulation
- routing_freqs (passed detached) receives no gradient
"""

import math

import pytest
import torch

from src.shram.model.attention.load_balance_loss import make_load_balance_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared test vectors. f sums to 1 (valid routing frequencies). p is a
# valid probability distribution (all positive, sums to 1).
_F = [0.4, 0.3, 0.2, 0.1]
_P = [0.25, 0.35, 0.25, 0.15]


def make_tensors(
    f: list[float] = _F,
    p: list[float] = _P,
    p_requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct routing_freqs and assignment_probs tensors for testing."""
    routing_freqs = torch.tensor(f, dtype=torch.float32)
    assignment_probs = torch.tensor(p, dtype=torch.float32, requires_grad=p_requires_grad)
    return routing_freqs, assignment_probs


def expected_gshard(f: list[float], p: list[float]) -> float:
    """Reference implementation of the gshard formula."""
    L = len(f)
    return sum(f[i] * p[i] for i in range(L)) / L


def expected_ce(f: list[float], p: list[float]) -> float:
    """Reference implementation of the ce formula."""
    L = len(f)
    return -sum((1.0 - f[i]) * math.log(p[i]) for i in range(L)) / (L - 1)


def expected_bce(f: list[float], p: list[float]) -> float:
    """Reference implementation of the bce formula using log1p(-p) for safety."""
    L = len(f)
    return -sum(
        (1.0 - f[i]) * math.log(p[i]) + f[i] * math.log1p(-p[i])
        for i in range(L)
    ) / L


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_returns_callable_for_gshard(self):
        """Factory must return a callable for loss_type='gshard'."""
        fn = make_load_balance_loss("gshard")
        assert callable(fn)

    def test_returns_callable_for_ce(self):
        """Factory must return a callable for loss_type='ce'."""
        fn = make_load_balance_loss("ce")
        assert callable(fn)

    def test_returns_callable_for_bce(self):
        """Factory must return a callable for loss_type='bce'."""
        fn = make_load_balance_loss("bce")
        assert callable(fn)

    def test_raises_for_invalid_type(self):
        """Factory must raise ValueError for an unrecognised loss_type."""
        with pytest.raises(ValueError, match="load_balance_loss_type"):
            make_load_balance_loss("invalid_type")


# ---------------------------------------------------------------------------
# Formula correctness and output shape
# ---------------------------------------------------------------------------

class TestFormulas:
    """Verify each formulation against a Python reference on known inputs.

    The reference functions (expected_gshard, expected_ce, expected_bce) are
    direct Python translations of the formulas in the plan entry. Comparing
    the tensor output against these references certifies the PyTorch
    implementation matches the intended formula.
    """

    def _verify_formula(
        self,
        loss_type: str,
        f: list[float],
        p: list[float],
        expected: float,
    ) -> None:
        """Call loss_fn and assert output matches expected to 1e-6."""
        routing_freqs, assignment_probs = make_tensors(f, p)
        loss_fn = make_load_balance_loss(loss_type)
        loss = loss_fn(routing_freqs, assignment_probs)

        assert loss.shape == (), (
            f"{loss_type}: expected scalar output, got shape {loss.shape}"
        )
        assert loss.item() == pytest.approx(expected, abs=1e-6), (
            f"{loss_type}: expected {expected:.8f}, got {loss.item():.8f} "
            f"with f={f}, p={p}"
        )

    # gshard ----------------------------------------------------------------

    def test_gshard_formula_on_known_inputs(self):
        """gshard_loss must compute (1/L) * Σ_i f_i * p_i."""
        f, p = _F, _P
        self._verify_formula("gshard", f, p, expected_gshard(f, p))

    def test_gshard_formula_at_balanced_routing(self):
        """gshard_loss must equal (1/L^2) when f_i = p_i = 1/L for all i."""
        L = 4
        f = [1.0 / L] * L
        p = [1.0 / L] * L
        self._verify_formula("gshard", f, p, expected_gshard(f, p))

    # ce --------------------------------------------------------------------

    def test_ce_formula_on_known_inputs(self):
        """ce_loss must compute -(1/(L-1)) * Σ_i (1 - f_i) * log(p_i)."""
        f, p = _F, _P
        self._verify_formula("ce", f, p, expected_ce(f, p))

    def test_ce_formula_at_balanced_routing(self):
        """ce_loss must equal log(L) when f_i = p_i = 1/L for all i."""
        L = 4
        f = [1.0 / L] * L
        p = [1.0 / L] * L
        self._verify_formula("ce", f, p, expected_ce(f, p))

    # bce -------------------------------------------------------------------

    def test_bce_formula_on_known_inputs(self):
        """bce_loss must compute -(1/L) * Σ_i [(1-f_i)*log(p_i) + f_i*log1p(-p_i)]."""
        f, p = _F, _P
        self._verify_formula("bce", f, p, expected_bce(f, p))

    def test_bce_formula_at_balanced_routing(self):
        """bce_loss must equal the expected symmetric value when f_i = p_i = 1/L."""
        L = 4
        f = [1.0 / L] * L
        p = [1.0 / L] * L
        self._verify_formula("bce", f, p, expected_bce(f, p))


# ---------------------------------------------------------------------------
# Gradient isolation
# ---------------------------------------------------------------------------

class TestGradientIsolation:
    """Verify that gradients flow to assignment_probs but not routing_freqs.

    This is the core isolation property: routing_freqs comes from discrete TopK
    selections (no gradient path), and logits are detached before softmax in the
    caller. So the only differentiable path into the loss is through assignment_probs
    → expert_bias.
    """

    def _test_isolation(self, loss_type: str) -> None:
        routing_freqs, assignment_probs = make_tensors(p_requires_grad=True)
        loss_fn = make_load_balance_loss(loss_type)
        loss = loss_fn(routing_freqs, assignment_probs)
        loss.backward()

        assert assignment_probs.grad is not None, (
            f"{loss_type}: assignment_probs must receive gradient"
        )
        assert torch.isfinite(assignment_probs.grad).all(), (
            f"{loss_type}: assignment_probs gradient must be finite"
        )
        assert routing_freqs.grad is None, (
            f"{loss_type}: routing_freqs must not receive gradient"
        )

    def test_gshard_gradient_flows_to_assignment_probs_only(self):
        """gshard_loss must propagate gradient to assignment_probs; routing_freqs is blocked."""
        self._test_isolation("gshard")

    def test_ce_gradient_flows_to_assignment_probs_only(self):
        """ce_loss must propagate gradient to assignment_probs; routing_freqs is blocked."""
        self._test_isolation("ce")

    def test_bce_gradient_flows_to_assignment_probs_only(self):
        """bce_loss must propagate gradient to assignment_probs; routing_freqs is blocked."""
        self._test_isolation("bce")
