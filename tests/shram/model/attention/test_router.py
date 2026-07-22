"""Tests for MoSRAHRouter.

Invariants verified:
- routing_weight is the only routing projection; balance_weight does not exist
- regret_loss gradient reaches routing_weight; task loss gradient also reaches routing_weight
- regret_loss gradient reaches the router input x
- router_diagnostics exposes exactly {regret_loss, logit_regret, logit_std}
- regret_loss has gradient; logit_regret and logit_std are detached
- compiled and eager router diagnostics are numerically identical
- Output shapes: selected_heads (B, N, K), routing_probs (B, N, K), regret_loss scalar, logit_regret scalar
- routing_probs are non-negative and have total mass 0 or 1 per token
- selected_heads are valid indices in [0, L-1] and are distinct per token
- regret_loss is zero when every expert is assigned at its peak-preference token within the block
- regret_loss matches hand-calculated values for known routing configurations
- dead tokens do not affect regret_loss
- every expert appears exactly once per routing block in selected_heads
- cached decode (router_cache not None, N=1) produces valid (B, 1, K) selections
- W consecutive decode steps cover all L experts exactly once per block
- decode routing respects the used_in_block mask from cache (no expert reuse within block)
"""

import pytest
import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.router import MoSRAHRouter
from src.shram.model.cache.router_cache import RouterCache


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
        selected_heads, _, _ = router(x, active_mask)
        assert selected_heads.shape == (2, 8, config.num_selected_heads)

    def test_routing_probs_shape(self):
        """routing_probs must be (B, N, K)."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask)
        assert routing_probs.shape == (2, 8, config.num_selected_heads)

    def test_regret_loss_is_scalar(self):
        """regret_loss must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask)
        assert diagnostics["regret_loss"].shape == ()

    def test_logit_regret_is_scalar(self):
        """logit_regret must be a zero-dimensional tensor."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask)
        assert diagnostics["logit_regret"].shape == ()


# ---------------------------------------------------------------------------
# Routing probabilities
# ---------------------------------------------------------------------------

class TestRoutingProbabilities:
    def test_routing_probs_have_zero_or_unit_total_mass(self):
        """A token either has a normalized sparse mixture or no sparse contribution."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(3, 10, 64)
        active_mask = torch.ones(3, 10, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask)
        token_sums = routing_probs.sum(dim=-1)
        zero_mass = token_sums == 0
        unit_mass = torch.isclose(
            token_sums,
            torch.ones_like(token_sums),
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.all(zero_mass | unit_mass)

    def test_routing_probs_are_nonnegative(self):
        """Entmax outputs are non-negative; gathering and normalization preserve this."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask)
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
        selected_heads, _, _ = router(x, active_mask)
        assert (selected_heads >= 0).all()
        assert (selected_heads < config.num_mosrah_heads).all()

    def test_selected_heads_are_distinct_per_token(self):
        """The block solver's TopK result must contain K distinct head indices."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        selected_heads, _, _ = router(x, active_mask)
        B, N, K = selected_heads.shape
        for b in range(B):
            for n in range(N):
                assert selected_heads[b, n].unique().shape[0] == K


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:
    """Tests certifying gradient flow in the coupled single-projection architecture.

    routing_weight is the only routing parameter. Both task loss and regret_loss
    must train it directly — there is no gradient isolation between the two signals.
    """

    def test_regret_loss_reaches_routing_weight(self):
        """Backward on regret_loss must populate routing_weight.grad with a
        non-zero, finite gradient — confirming the loss trains the routing projection."""
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask)
        diagnostics["regret_loss"].backward()

        assert router.routing_weight.grad is not None
        assert router.routing_weight.grad.abs().sum().item() > 0.0

    def test_task_loss_reaches_routing_weight(self):
        """A nonconstant weighted reduction must train the routing projection."""
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, routing_probs, _ = router(x, active_mask)
        coefficients = torch.arange(
            config.num_selected_heads,
            dtype=routing_probs.dtype,
            device=routing_probs.device,
        )
        (routing_probs * coefficients).sum().backward()

        assert router.routing_weight.grad is not None
        assert router.routing_weight.grad.abs().sum().item() > 0.0

    def test_regret_loss_reaches_input(self):
        """Backward on regret_loss must populate x.grad — confirming gradient
        flows through the router input, not just to routing_weight."""
        torch.manual_seed(0)
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, config.embedding_width, requires_grad=True)
        active_mask = torch.ones(2, 8, dtype=torch.bool)

        _, _, diagnostics = router(x, active_mask)
        diagnostics["regret_loss"].backward()

        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Router diagnostics
# ---------------------------------------------------------------------------

class TestRouterDiagnostics:
    """Tests for the router_diagnostics dict returned from MoSRAHRouter.forward.

    Verifies the exact three-key contract {regret_loss, logit_regret, logit_std},
    that regret_loss retains gradient, and that monitoring scalars are detached.
    """

    def test_diagnostics_has_exactly_three_keys(self):
        """router_diagnostics must expose exactly {regret_loss, logit_regret, logit_std}."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask)
        assert set(diagnostics.keys()) == {"regret_loss", "logit_regret", "logit_std"}

    def test_regret_loss_has_gradient(self):
        """regret_loss must retain its gradient; it is the training signal."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask)
        assert diagnostics["regret_loss"].requires_grad

    def test_monitoring_scalars_are_detached(self):
        """logit_regret and logit_std must be detached — they are monitoring metrics only."""
        config = small_config()
        router = MoSRAHRouter(config)
        x = torch.randn(2, 8, 64)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask)
        for key in ("logit_regret", "logit_std"):
            assert not diagnostics[key].requires_grad, (
                f"diagnostic scalar '{key}' must be detached but requires_grad is True"
            )


# ---------------------------------------------------------------------------
# Architecture invariants
# ---------------------------------------------------------------------------

class TestArchitectureInvariants:
    """Tests certifying the single-projection coupled architecture.

    routing_weight is the only routing parameter. balance_weight must not exist.
    """

    def test_routing_weight_is_parameter(self):
        """routing_weight must be an nn.Parameter so the optimizer sees and updates it,
        and HuggingFace _init_weights does not override its kaiming initialization."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert isinstance(router.routing_weight, torch.nn.Parameter)

    def test_routing_weight_shape(self):
        """routing_weight must have shape (num_mosrah_heads, embedding_width)."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert router.routing_weight.shape == (config.num_mosrah_heads, config.embedding_width)

    def test_balance_weight_does_not_exist(self):
        """balance_weight must not exist — the router has one coupled projection only."""
        config = small_config()
        router = MoSRAHRouter(config)
        assert not hasattr(router, "balance_weight")


# ---------------------------------------------------------------------------
# Regret loss
# ---------------------------------------------------------------------------

class TestRegretLoss:
    """Hand-calculated verification of _compute_regret.

    All cases use B=1, L=2, K=1, N=2 (num_blocks=1, W=2). The static method
    is called directly with synthetic tensors, bypassing the block solver entirely.
    This makes expected values exact and analytical.
    """

    @staticmethod
    def _make_inputs(
        routing_scores: list,
        selected: list,
        active: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tensors for _compute_regret from plain Python lists.

        Args:
            routing_scores: Shape (N, L) list-of-lists of routing probabilities.
            selected: Shape (N,) list of expert indices (K=1 per token).
            active: Shape (N,) list of bools.

        Returns:
            Tuple of (routing_scores_t, routing_logits_t, selected_heads_blocked, active_mask_t)
            with B=1, num_blocks=1, W=N, K=1.
        """
        # B=1; N and L inferred from routing_scores.
        scores_t  = torch.tensor([routing_scores], dtype=torch.float32)       # (1, N, L)
        # Log of scores as stand-in logits — only relative ordering matters for
        # logit_regret, which is a detached monitoring scalar in these tests.
        logits_t  = torch.log(scores_t.clamp(min=1e-9))                       # (1, N, L)
        N = len(selected)
        # selected_heads_blocked: (B=1, nb=1, W=N, K=1)
        sel_t = torch.tensor([[[[s] for s in selected]]], dtype=torch.long)
        act_t = torch.tensor([active], dtype=torch.bool)                       # (1, N)
        return scores_t, logits_t, sel_t, act_t

    def test_nonzero_regret(self):
        """regret_loss must equal 0.2 when both experts are assigned sub-optimally.

        Setup: B=1, L=2, K=1, N=2 (one block of W=2).
        Token 0 → expert 0, token 1 → expert 1.
        Expert 0: p_chosen=0.6, p_max=max(0.6, 0.8)=0.8 → regret=0.2
        Expert 1: p_chosen=0.2, p_max=max(0.4, 0.2)=0.4 → regret=0.2
        regret_loss = (0.2 + 0.2) / (1 block × 2 experts) = 0.2
        """
        scores, logits, selected, active = self._make_inputs(
            routing_scores=[[0.6, 0.4], [0.8, 0.2]],
            selected=[0, 1],
            active=[True, True],
        )

        regret_loss, _ = MoSRAHRouter._compute_regret(scores, logits, selected, active)

        assert regret_loss.item() == pytest.approx(0.2, abs=1e-6)

    def test_zero_regret(self):
        """regret_loss must be exactly 0 when every expert is assigned at its
        peak-preference token within the block.

        Setup: B=1, L=2, K=1, N=2 (one block of W=2).
        Token 0 → expert 0, token 1 → expert 1.
        Expert 0: p_chosen=0.6, p_max=max(0.6, 0.3)=0.6 → regret=0
        Expert 1: p_chosen=0.7, p_max=max(0.4, 0.7)=0.7 → regret=0
        regret_loss = 0.0
        """
        scores, logits, selected, active = self._make_inputs(
            routing_scores=[[0.6, 0.4], [0.3, 0.7]],
            selected=[0, 1],
            active=[True, True],
        )

        regret_loss, _ = MoSRAHRouter._compute_regret(scores, logits, selected, active)

        assert regret_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_dead_token_excluded_from_max(self):
        """regret_loss must be 0.2 when a dead token is assigned and excluded from p_max.

        Setup: B=1, L=2, K=1, N=2 (one block of W=2).
        Token 0 alive → expert 0, token 1 dead → expert 1.
        Expert 0: p_chosen=0.6 (alive), p_max=max(1.0×0.6, 0.0×0.8)=0.6 → regret=0
        Expert 1: p_chosen=0 (dead, gated), p_max=max(1.0×0.4, 0.0×0.2)=0.4 → regret=0.4
        regret_loss = (0 + 0.4) / (1 block × 2 experts) = 0.2
        """
        scores, logits, selected, active = self._make_inputs(
            routing_scores=[[0.6, 0.4], [0.8, 0.2]],
            selected=[0, 1],
            active=[True, False],
        )

        regret_loss, _ = MoSRAHRouter._compute_regret(scores, logits, selected, active)

        assert regret_loss.item() == pytest.approx(0.2, abs=1e-6)

    def test_regret_loss_has_gradient_logit_regret_detached(self):
        """regret_loss must have requires_grad=True; logit_regret must be detached."""
        scores, logits, selected, active = self._make_inputs(
            routing_scores=[[0.6, 0.4], [0.8, 0.2]],
            selected=[0, 1],
            active=[True, True],
        )
        scores = scores.requires_grad_(True)

        regret_loss, logit_regret = MoSRAHRouter._compute_regret(scores, logits, selected, active)

        assert regret_loss.requires_grad
        assert not logit_regret.requires_grad


# ---------------------------------------------------------------------------
# Masked continuation behavior
# ---------------------------------------------------------------------------

class TestMaskedContinuationBehavior:
    def test_dead_tokens_do_not_affect_regret_loss(self):
        """Changing a dead token's hidden state must not affect regret_loss.

        Verified by marking a token dead, computing regret_loss, then replacing
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
        _, _, diag_a = router(x, active_mask)
        loss_a = diag_a["regret_loss"]

        x_modified = x.clone()
        x_modified[0, 3] = torch.randn(config.embedding_width) * 100.0
        _, _, diag_b = router(x_modified, active_mask)
        loss_b = diag_b["regret_loss"]

        torch.testing.assert_close(loss_a, loss_b)


# ---------------------------------------------------------------------------
# Block balance
# ---------------------------------------------------------------------------

class TestBlockBalance:
    """Tests for block-balanced exact load balance.

    With W = L/K, every expert must appear exactly once per block of W tokens.
    """

    def test_each_expert_used_exactly_once_per_block(self):
        """Every expert must appear exactly once per routing block in selected_heads.

        With L=8, K=4, W=2: each 2-token block covers all 8 experts exactly once.
        """
        torch.manual_seed(0)
        config = small_config(num_mosrah_heads=8, num_selected_heads=4)
        router = MoSRAHRouter(config)
        B, N = 2, 12  # 6 complete blocks of W=2
        x = torch.randn(B, N, config.embedding_width)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            selected_heads, _, _ = router(x, active_mask)

        L, W = config.num_mosrah_heads, router.block_length
        num_blocks = N // W
        assignment = torch.zeros(B, N, L, dtype=torch.long)
        assignment.scatter_(-1, selected_heads, 1)
        expert_counts = assignment.view(B, num_blocks, W, L).sum(dim=2)  # (B, blocks, L)
        expected = torch.ones(B, num_blocks, L, dtype=torch.long)
        assert torch.equal(expert_counts, expected), (
            f"Expert counts per block not exactly 1. "
            f"Max: {expert_counts.max().item()}, Min: {expert_counts.min().item()}"
        )

    def test_block_balance_holds_for_non_multiple_sequence_length(self):
        """Block balance must hold for complete blocks when N is not a multiple of W."""
        torch.manual_seed(3)
        config = small_config(num_mosrah_heads=8, num_selected_heads=4)
        router = MoSRAHRouter(config)
        B, N, W = 1, 13, router.block_length  # W=2, 6 complete blocks + 1 partial

        x = torch.randn(B, N, config.embedding_width)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            selected_heads, _, _ = router(x, active_mask)

        L = config.num_mosrah_heads
        complete_blocks = N // W
        assignment = torch.zeros(B, N, L, dtype=torch.long)
        assignment.scatter_(-1, selected_heads, 1)
        expert_counts = assignment[:, :complete_blocks * W, :].view(B, complete_blocks, W, L).sum(dim=2)
        expected = torch.ones(B, complete_blocks, L, dtype=torch.long)
        assert torch.equal(expert_counts, expected), (
            f"Expert counts in complete blocks not exactly 1. "
            f"Max: {expert_counts.max().item()}, Min: {expert_counts.min().item()}"
        )


class TestRouterCompileEquivalenceFuzz:
    """Fuzz tests for eager-vs-compiled MoSRAHRouter equivalence.

    This suite certifies that compiling the router does not change its forward
    contract. It intentionally compares the full router outputs, not just the
    realized capacity counts, because the compile boundary should preserve
    selection, probabilities, regret_loss, and monitoring output for the
    same module state and input tensors.

    The tests are CUDA-only because this project already treats uncached
    torch.compile coverage as CUDA-only.
    """

    NUM_TRIALS = 25
    BATCH_SIZE = 4
    TRAINING_SEQUENCE_LENGTH = 256

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
                    )

                    compiled_selected, compiled_probs, compiled_diag = (
                        compiled_router(
                            x,
                            active_mask,
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
# Compiled backward
# ---------------------------------------------------------------------------

class TestCompiledBackward:
    """Compiled forward+backward test for the coupled single-projection router.

    CUDA-only via the device fixture from conftest.py.
    """

    def test_compiled_forward_backward_no_error(self, device):
        """Compiled router must run one forward+backward step without error,
        and routing_weight.grad must be populated after backward."""
        if device.type != "cuda":
            pytest.skip("Compiled backward test is CUDA-only.")

        torch.manual_seed(0)
        config = small_config(mosrah_overallocation_factor=2.0)
        router = MoSRAHRouter(config).to(device)
        compiled_router = torch.compile(router, fullgraph=True, dynamic=False)
        opt = torch.optim.SGD(router.parameters(), lr=0.01)

        x = torch.randn(2, 16, config.embedding_width, device=device)
        active_mask = torch.ones(2, 16, dtype=torch.bool, device=device)

        opt.zero_grad()
        _, routing_probs, diagnostics = compiled_router(x, active_mask)
        loss = routing_probs.sum() + diagnostics["regret_loss"]
        loss.backward()

        assert router.routing_weight.grad is not None, (
            "routing_weight.grad must be populated after compiled backward"
        )


# ---------------------------------------------------------------------------
# Cached decode inference
# ---------------------------------------------------------------------------

class TestCachedDecodeInference:
    """Verify the router's decode mode (N=1 with RouterCache).

    These tests exercise the three-mode router forward: training (cache=None),
    prefill (N>1 with cache), and decode (N=1 with cache).
    """

    def _make_router_and_cache(self, **config_kwargs):
        """Helper: construct a router and a matching RouterCache."""
        config = small_config(**config_kwargs)
        torch.manual_seed(42)
        router = MoSRAHRouter(config)
        cache = RouterCache(
            block_length=config.block_length,
            num_mosrah_heads=config.num_mosrah_heads,
            batch_size=2,
            device=torch.device("cpu"),
        )
        return router, cache, config

    def test_decode_output_shapes(self):
        """Decode mode must return (B, 1, K) selected_heads and (B, 1, K) routing_probs."""
        router, cache, config = self._make_router_and_cache()
        B, K = 2, config.num_selected_heads

        x = torch.randn(B, 1, config.embedding_width)
        active_mask = torch.ones(B, 1, dtype=torch.bool)

        selected_heads, routing_probs, diagnostics = router(x, active_mask, cache)

        assert selected_heads.shape == (B, 1, K)
        assert routing_probs.shape == (B, 1, K)

    def test_decode_selected_heads_valid_range(self):
        """Decode mode selected_heads must be in [0, L-1]."""
        router, cache, config = self._make_router_and_cache()
        L = config.num_mosrah_heads

        x = torch.randn(2, 1, config.embedding_width)
        active_mask = torch.ones(2, 1, dtype=torch.bool)
        selected_heads, _, _ = router(x, active_mask, cache)

        assert (selected_heads >= 0).all()
        assert (selected_heads < L).all()

    def test_decode_routing_probs_sum_to_one(self):
        """Decode mode routing_probs must sum to 1 per token."""
        router, cache, config = self._make_router_and_cache()

        x = torch.randn(2, 1, config.embedding_width)
        active_mask = torch.ones(2, 1, dtype=torch.bool)
        _, routing_probs, _ = router(x, active_mask, cache)

        sums = routing_probs.sum(dim=-1)  # (B, 1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_w_decode_steps_cover_all_experts(self):
        """W consecutive decode steps must collectively assign every expert exactly once."""
        # W=2, K=2, L=4 — simplest nontrivial block
        router, cache, config = self._make_router_and_cache(
            num_mosrah_heads=4, num_selected_heads=2
        )
        W = config.block_length  # = 4 // 2 = 2
        L = config.num_mosrah_heads
        B = 2

        all_selected = []
        for _ in range(W):
            x = torch.randn(B, 1, config.embedding_width)
            active_mask = torch.ones(B, 1, dtype=torch.bool)
            selected_heads, _, _ = router(x, active_mask, cache)
            all_selected.append(selected_heads[:, 0, :])  # (B, K)

        # Union of selections across W steps: each expert must appear exactly once
        combined = torch.cat(all_selected, dim=-1)  # (B, W*K = L)
        for b in range(B):
            experts_used = combined[b].tolist()
            assert sorted(experts_used) == list(range(L)), (
                f"Batch item {b}: expected all {L} experts, got {sorted(experts_used)}"
            )

    def test_decode_respects_cached_used_in_block(self):
        """Decode must not select experts already marked as used in the cache."""
        # Manually pre-fill cache with experts 0..K-1 marked used
        router, cache, config = self._make_router_and_cache()
        K = config.num_selected_heads
        L = config.num_mosrah_heads
        B = 2

        # Mark experts 0..K-1 as already used in the current block
        cache._used_in_block[:, :K] = True
        cache._step_in_block[:] = K

        x = torch.randn(B, 1, config.embedding_width)
        active_mask = torch.ones(B, 1, dtype=torch.bool)
        selected_heads, _, _ = router(x, active_mask, cache)

        # Must not select any expert in [0, K-1]
        for b in range(B):
            for e in selected_heads[b, 0].tolist():
                assert e >= K, (
                    f"Batch {b}: selected expert {e} which was already used in block"
                )

    def test_decode_diagnostics_are_scalars(self):
        """Decode mode must return the standard scalar diagnostics dict."""
        router, cache, config = self._make_router_and_cache()

        x = torch.randn(2, 1, config.embedding_width)
        active_mask = torch.ones(2, 1, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, cache)

        assert set(diagnostics.keys()) == {"regret_loss", "logit_regret", "logit_std"}
        for key, val in diagnostics.items():
            assert val.shape == (), f"{key} must be a scalar"

    def test_decode_regret_loss_is_zero(self):
        """Decode mode regret_loss must be exactly 0.0 — regret is undefined over a
        single decode step (not a complete W-token block); returning zero is the
        correct no-op since backward is never called during inference."""
        router, cache, config = self._make_router_and_cache()

        x = torch.randn(2, 1, config.embedding_width)
        active_mask = torch.ones(2, 1, dtype=torch.bool)
        _, _, diagnostics = router(x, active_mask, cache)

        assert diagnostics["regret_loss"].item() == pytest.approx(0.0, abs=1e-6)
        assert diagnostics["logit_regret"].item() == pytest.approx(0.0, abs=1e-6)
