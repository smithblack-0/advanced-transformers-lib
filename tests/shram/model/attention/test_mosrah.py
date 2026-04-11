"""Tests for the full MoSRAH sparse path.

Invariants verified: model-space output shape, propagation of load_balance_loss,
weighted reduction semantics, support for both RoPE position modes, uncached
execution, cached execution with the real layer-local cache, cached/uncached
external-contract equivalence on the sparse output, and preservation of the two
distinct gradient paths exposed by the assembled layer.
"""

import pytest
import torch

from src.shram.model.attention.mosrah import MoSRAHLayer
from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.configuration import ShramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(rope_mode: str = "main_sequence") -> ShramConfig:
    """Construct a small SHRAM config for MoSRAH path tests."""
    return ShramConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_sliding_window_heads=2,
        num_mosrah_heads=3,
        num_selected_heads=2,
        head_dim=4,
        window_size=4,
        rope_mode=rope_mode,
        training_sequence_length=8,
        inference_sequence_length=8,
        attention_dropout=0.0,
    )


def make_inputs(
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a small model-space input and authoritative position tensor."""
    hidden_states = torch.tensor(
        [[
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
        ]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    position_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    return hidden_states, position_ids


def make_cache(config: ShramConfig, batch_size: int) -> MoSRAHCache:
    """Construct a real layer-local MoSRAH cache for cached execution tests."""
    return MoSRAHCache(
        num_mosrah_heads=config.num_mosrah_heads,
        head_dim=config.head_dim,
        batch_size=batch_size,
        device=torch.device("cpu"),
        initial_buffer_size=8,
    )


# ---------------------------------------------------------------------------
# The assembled-path tests belong here because only 10.C can certify that the
# real router, packing path, position layer, BEA, unpacking, and cache combine
# into one coherent sparse path with the correct external contract. Lower-level
# suites already verify local semantics; this section verifies that the real
# pieces run together sanely in both uncached and cached execution.
# ---------------------------------------------------------------------------

class TestRealExecution:
    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_uncached_execution_runs_sanely_and_returns_the_external_contract(self, rope_mode):
        """The assembled uncached path should return model-space outputs and scalar loss."""
        torch.manual_seed(0)
        config = make_config(rope_mode=rope_mode)
        layer = MoSRAHLayer(config)
        hidden_states, position_ids = make_inputs()

        sparse_output, load_balance_loss = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        assert sparse_output.shape == hidden_states.shape
        assert load_balance_loss.ndim == 0
        assert torch.isfinite(sparse_output).all()
        assert torch.isfinite(load_balance_loss)

    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_cached_execution_accumulates_in_the_real_layer_local_cache(self, rope_mode):
        """Repeated cached calls should grow the real sparse cache state."""
        torch.manual_seed(0)
        config = make_config(rope_mode=rope_mode)
        layer = MoSRAHLayer(config)

        hidden_states, position_ids = make_inputs()
        prefix_hidden_states = hidden_states[:, :2]
        prefix_position_ids = position_ids[:, :2]
        current_hidden_states = hidden_states[:, 2:]
        current_position_ids = position_ids[:, 2:]

        cache = make_cache(config, batch_size=hidden_states.shape[0])

        prefix_output, prefix_load_balance_loss = layer(
            hidden_states=prefix_hidden_states,
            position_ids=prefix_position_ids,
            cache=cache,
        )
        lengths_after_prefix = cache.get_heads_lengths().clone()

        current_output, current_load_balance_loss = layer(
            hidden_states=current_hidden_states,
            position_ids=current_position_ids,
            cache=cache,
        )
        lengths_after_current = cache.get_heads_lengths().clone()

        uncached_current_output, uncached_current_load_balance_loss = layer(
            hidden_states=current_hidden_states,
            position_ids=current_position_ids,
            cache=None,
        )

        assert prefix_output.shape == prefix_hidden_states.shape
        assert current_output.shape == current_hidden_states.shape
        assert prefix_load_balance_loss.ndim == 0
        assert current_load_balance_loss.ndim == 0
        assert torch.all(lengths_after_prefix >= 0)
        assert torch.any(lengths_after_prefix > 0)
        assert torch.all(lengths_after_current >= lengths_after_prefix)

        # Cache changes the sparse attention context, so the sparse output may
        # differ from an uncached current-chunk-only call. Routing itself does
        # not depend on cache, so the load-balance loss should remain identical.
        assert not torch.allclose(current_output, uncached_current_output)
        torch.testing.assert_close(
            current_load_balance_loss,
            uncached_current_load_balance_loss,
        )

    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_cached_current_chunk_matches_the_corresponding_suffix_of_full_uncached_execution(
        self,
        rope_mode,
    ):
        """Cached prefix/current execution should match the suffix of a full uncached run."""
        torch.manual_seed(0)
        config = make_config(rope_mode=rope_mode)
        layer = MoSRAHLayer(config)

        hidden_states, position_ids = make_inputs()
        prefix_hidden_states = hidden_states[:, :2]
        prefix_position_ids = position_ids[:, :2]
        current_hidden_states = hidden_states[:, 2:]
        current_position_ids = position_ids[:, 2:]

        full_output, full_load_balance_loss = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        cache = make_cache(config, batch_size=hidden_states.shape[0])

        _, prefix_load_balance_loss = layer(
            hidden_states=prefix_hidden_states,
            position_ids=prefix_position_ids,
            cache=cache,
        )
        current_output, current_load_balance_loss = layer(
            hidden_states=current_hidden_states,
            position_ids=current_position_ids,
            cache=cache,
        )

        torch.testing.assert_close(
            current_output,
            full_output[:, 2:],
            atol=1e-5,
            rtol=1e-5,
        )

        # The sparse output should agree with the corresponding suffix of a full
        # uncached run, while the current-chunk load-balance loss remains a
        # well-formed scalar produced by the current routing step.
        assert current_output.shape == current_hidden_states.shape
        assert prefix_load_balance_loss.ndim == 0
        assert current_load_balance_loss.ndim == 0
        assert full_load_balance_loss.ndim == 0


# ---------------------------------------------------------------------------
# The final section certifies the two gradient responsibilities preserved by
# assembly. The sparse output must still backpropagate through the real output
# path, including the router projection path carried by unbiased routing_probs.
# Separately, the returned load_balance_loss must still expose the router's
# balancing signal through the assembled layer.
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_sparse_output_backward_reaches_the_real_output_path_but_not_expert_bias(self):
        """Sparse-output gradients should flow through the assembled path without using biased scores."""
        torch.manual_seed(0)
        config = make_config("main_sequence")
        layer = MoSRAHLayer(config)
        hidden_states, position_ids = make_inputs(requires_grad=True)

        sparse_output, load_balance_loss = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        del load_balance_loss

        sparse_output.sum().backward()

        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

        router_weight_grad = layer.router.routing_projection.weight.grad
        assert router_weight_grad is not None
        assert torch.isfinite(router_weight_grad).all()

        bea_q_grad = layer.bea.q_proj.grad
        assert bea_q_grad is not None
        assert torch.isfinite(bea_q_grad).all()

        # The sparse output should not backpropagate through the biased selection
        # path into expert_bias. That path is intentionally excluded from the
        # output reduction contract.
        expert_bias_grad = layer.router.expert_bias.grad
        assert expert_bias_grad is None or torch.all(expert_bias_grad == 0)

    def test_load_balance_loss_backward_reaches_expert_bias_but_not_routing_projection(self):
        """The assembled layer should expose the router's load-balance signal to training."""
        torch.manual_seed(0)
        config = make_config("main_sequence")
        layer = MoSRAHLayer(config)
        hidden_states, position_ids = make_inputs(requires_grad=True)

        sparse_output, load_balance_loss = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        del sparse_output

        load_balance_loss.backward()

        expert_bias_grad = layer.router.expert_bias.grad
        assert expert_bias_grad is not None
        assert torch.isfinite(expert_bias_grad).all()

        router_weight_grad = layer.router.routing_projection.weight.grad
        assert router_weight_grad is None or torch.all(router_weight_grad == 0)
