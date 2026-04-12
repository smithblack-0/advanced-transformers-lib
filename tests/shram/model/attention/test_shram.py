"""Integration tests for the SHRAM hybrid attention layer.

Invariants verified: direct hybrid composition in model space, propagation of
the sparse-path load-balance loss, uncached and cached real execution, correct
interaction with the real per-layer cache, preservation of the decoder-facing
(B, N, d) interface, response of the assembled hybrid layer to top-level
experimental configuration changes, and preservation of the two distinct
gradient paths exposed by the hybrid boundary.
"""

import pytest
import torch

from src.shram.model.attention.shram import SHRAMHybridLayer
from src.shram.model.cache.shram_layer_cache import ShramLayerCache
from src.shram.model.configuration import ShramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> ShramConfig:
    """Construct a small SHRAM config for hybrid-layer integration tests."""
    config_kwargs = dict(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_sliding_window_heads=2,
        num_mosrah_heads=5,
        num_selected_heads=2,
        head_dim=4,
        window_size=4,
        rope_mode="main_sequence",
        training_sequence_length=8,
        inference_sequence_length=8,
        attention_dropout=0.0,
        use_cache=True,
        mosrah_rope_theta = 20,
    )
    config_kwargs.update(overrides)
    return ShramConfig(**config_kwargs)


def make_inputs(
    requires_grad: bool = False,
    start_position: int = 0,
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
    )
    hidden_states.requires_grad_(requires_grad)

    position_ids = torch.arange(
        start_position,
        start_position + hidden_states.shape[1],
        dtype=torch.long,
    ).unsqueeze(0)
    return hidden_states, position_ids


def make_continued_decoding_inputs(
    prefix_length: int = 10,
    current_length: int = 4,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a warmed-cache decoding scenario with current positions 10..13."""
    total_length = prefix_length + current_length
    hidden_states = (
        torch.arange(total_length * 8, dtype=torch.float32).view(1, total_length, 8) / 10.0
        + 0.1
    )
    hidden_states.requires_grad_(requires_grad)

    position_ids = torch.arange(total_length, dtype=torch.long).unsqueeze(0)

    prefix_hidden_states = hidden_states[:, :prefix_length]
    prefix_position_ids = position_ids[:, :prefix_length]
    current_hidden_states = hidden_states[:, prefix_length:]
    current_position_ids = position_ids[:, prefix_length:]

    return (
        prefix_hidden_states,
        prefix_position_ids,
        current_hidden_states,
        current_position_ids,
    )


def make_layer_cache(
    config: ShramConfig,
    batch_size: int,
    initial_buffer_size: int = 8,
) -> ShramLayerCache:
    """Construct a real per-layer SHRAM cache."""
    return ShramLayerCache(
        sliding_window=config.window_size,
        num_mosrah_heads=config.num_mosrah_heads,
        mosrah_head_dim=config.head_dim,
        batch_size=batch_size,
        device=torch.device("cpu"),
        initial_buffer_size=initial_buffer_size,
    )


def make_layer(
    config: ShramConfig,
    seed: int = 0,
) -> SHRAMHybridLayer:
    """Construct a deterministically initialized hybrid layer."""
    torch.manual_seed(seed)
    return SHRAMHybridLayer(config)


def zero_module_parameters(module: torch.nn.Module) -> None:
    """Zero all learnable parameters in a module."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def first_parameter(module: torch.nn.Module) -> torch.nn.Parameter:
    """Return the first learnable parameter in a module."""
    return next(module.parameters())


# ---------------------------------------------------------------------------
# The central local responsibility of the hybrid layer is model-space
# composition. These tests therefore use the real local path and the real
# sparse path directly, and then verify that the hybrid layer adds those two
# model-space contributions without introducing extra logic or changing the
# sparse-path load-balance signal.
# ---------------------------------------------------------------------------

class TestHybridComposition:
    def test_hybrid_output_equals_the_sum_of_the_real_local_and_sparse_paths(self):
        """The hybrid layer should compute H(x) = h_l(x) + h_s(x)."""
        hidden_states, position_ids = make_inputs()
        layer = make_layer(make_config(), seed=0)

        local_output = layer.local_attention(
            x=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        sparse_output, sparse_load_balance_loss, _ = layer.sparse_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        hybrid_output, hybrid_load_balance_loss, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        torch.testing.assert_close(
            hybrid_output,
            local_output + sparse_output,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            hybrid_load_balance_loss,
            sparse_load_balance_loss,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_zeroing_local_path_leaves_only_the_real_sparse_path_contribution(self):
        """If the local path is zeroed, the hybrid output should equal the sparse output."""
        hidden_states, position_ids = make_inputs()
        layer = make_layer(make_config(), seed=0)

        zero_module_parameters(layer.local_attention)

        sparse_output, sparse_load_balance_loss, _ = layer.sparse_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        hybrid_output, hybrid_load_balance_loss, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        torch.testing.assert_close(
            hybrid_output,
            sparse_output,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            hybrid_load_balance_loss,
            sparse_load_balance_loss,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_zeroing_sparse_path_leaves_only_the_real_local_path_contribution(self):
        """If the sparse path is zeroed, the hybrid output should equal the local output."""
        hidden_states, position_ids = make_inputs()
        layer = make_layer(make_config(), seed=0)

        zero_module_parameters(layer.sparse_attention)

        local_output = layer.local_attention(
            x=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        hybrid_output, hybrid_load_balance_loss, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        sparse_output, sparse_load_balance_loss, _ = layer.sparse_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        torch.testing.assert_close(
            hybrid_output,
            local_output,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            hybrid_load_balance_loss,
            sparse_load_balance_loss,
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.isfinite(hybrid_load_balance_loss)
        torch.testing.assert_close(
            sparse_output,
            torch.zeros_like(sparse_output),
            atol=1e-6,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# These tests certify that the actual assembled feature stack runs together,
# not merely that the coordinator can be reasoned about in isolation. The real
# local path, real sparse path, and real per-layer cache all participate here.
# The cache tests check both that the real sub-caches are exercised and that
# cached execution preserves the same external hybrid contract as uncached
# execution when compared against the corresponding sequence suffix.
# ---------------------------------------------------------------------------

class TestRealExecution:
    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_uncached_execution_runs_sanely_and_preserves_the_decoder_facing_interface(
        self,
        rope_mode,
    ):
        """The real assembled hybrid layer should return (B, N, d) plus scalar loss."""
        hidden_states, position_ids = make_inputs(start_position=0)
        layer = make_layer(make_config(rope_mode=rope_mode), seed=0)

        hybrid_output, load_balance_loss, max_vio = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        assert hybrid_output.shape == hidden_states.shape
        assert load_balance_loss.ndim == 0
        assert max_vio.ndim == 0
        assert torch.isfinite(hybrid_output).all()
        assert torch.isfinite(load_balance_loss)
        assert torch.isfinite(max_vio)
        assert not max_vio.requires_grad

    def test_uncached_nonzero_starting_positions_fail_explicitly(self):
        """Uncached hybrid execution must not accept nonzero starting positions."""
        hidden_states, position_ids = make_inputs(start_position=10)
        layer = make_layer(make_config(), seed=0)

        with pytest.raises(ValueError, match="nonzero starting positions"):
            layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=None,
            )

    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_cached_execution_runs_sanely_with_the_real_per_layer_cache(self, rope_mode):
        """The real assembled hybrid layer should exercise both owned sub-caches."""
        hidden_states, position_ids = make_inputs(start_position=0)
        config = make_config(rope_mode=rope_mode)
        layer = make_layer(config, seed=0)
        layer_cache = make_layer_cache(
            config,
            batch_size=hidden_states.shape[0],
        )

        prefix_hidden_states = hidden_states[:, :2]
        prefix_position_ids = position_ids[:, :2]
        current_hidden_states = hidden_states[:, 2:]
        current_position_ids = position_ids[:, 2:]

        prefix_output, prefix_load_balance_loss, _ = layer(
            hidden_states=prefix_hidden_states,
            position_ids=prefix_position_ids,
            cache=layer_cache,
        )
        current_output, current_load_balance_loss, _ = layer(
            hidden_states=current_hidden_states,
            position_ids=current_position_ids,
            cache=layer_cache,
        )

        assert prefix_output.shape == prefix_hidden_states.shape
        assert current_output.shape == current_hidden_states.shape
        assert prefix_load_balance_loss.ndim == 0
        assert current_load_balance_loss.ndim == 0
        assert torch.isfinite(prefix_output).all()
        assert torch.isfinite(current_output).all()

        # The local sliding-window cache owns the scalar sequence length, while
        # the MoSRAH cache owns ragged per-head occupancy. Both should reflect
        # real use after cached hybrid execution.
        assert layer_cache.get_seq_length() >= prefix_hidden_states.shape[1]
        assert torch.any(layer_cache.mosrah_cache.get_heads_lengths() > 0)

    @pytest.mark.parametrize("rope_mode", ["main_sequence", "semantic_sequence"])
    def test_cached_current_chunk_matches_the_corresponding_suffix_of_full_uncached_execution(
            self,
            rope_mode,
    ):
        """Cached hybrid execution should match the corresponding suffix of a full uncached run."""
        hidden_states, position_ids = make_inputs(start_position=0)
        config = make_config(rope_mode=rope_mode)
        layer = make_layer(config, seed=0)

        prefix_hidden_states = hidden_states[:, :2]
        prefix_position_ids = position_ids[:, :2]
        current_hidden_states = hidden_states[:, 2:]
        current_position_ids = position_ids[:, 2:]

        # Legal uncached oracle: the entire sequence starts at zero.
        full_output, _, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        # Legal cached continued decoding: warm on the prefix, then decode the suffix.
        layer_cache = make_layer_cache(
            config,
            batch_size=hidden_states.shape[0],
        )
        _, _, _ = layer(
            hidden_states=prefix_hidden_states,
            position_ids=prefix_position_ids,
            cache=layer_cache,
        )
        current_cached_output, current_cached_load_balance_loss, _ = layer(
            hidden_states=current_hidden_states,
            position_ids=current_position_ids,
            cache=layer_cache,
        )

        torch.testing.assert_close(
            current_cached_output,
            full_output[:, 2:],
            atol=1e-5,
            rtol=1e-5,
        )

        assert current_cached_output.shape == current_hidden_states.shape
        assert current_cached_load_balance_loss.ndim == 0
        assert torch.isfinite(current_cached_load_balance_loss)

# ---------------------------------------------------------------------------
# Unit 11 sits at the top of the experimental feature stack, so it is also the
# right place to smoke-test that the real assembled hybrid layer actually
# responds when the experiment-facing configuration knobs change. These are not
# exact-value tests; they certify that the integrated feature stack does not
# silently ignore the configuration.
# ---------------------------------------------------------------------------

class TestConfigurationResponse:
    def test_changing_total_mosrah_head_capacity_changes_the_real_hybrid_output(self):
        """Changing sparsity / total sparse routed capacity should change hybrid behavior."""
        hidden_states, position_ids = make_inputs(start_position=0)

        layer_a = make_layer(
            make_config(num_mosrah_heads=3, num_selected_heads=2),
            seed=0,
        )
        layer_b = make_layer(
            make_config(num_mosrah_heads=5, num_selected_heads=2),
            seed=0,
        )

        output_a, _, _ = layer_a(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        output_b, _, _ = layer_b(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        assert not torch.allclose(output_a, output_b)

    def test_changing_rope_mode_changes_the_real_hybrid_output(self):
        """Switching RoPE mode should matter for a warmed continued-decoding chunk.

        This uses deterministic random inputs rather than arithmetic sequences,
        because contiguous numbered sequences can accidentally preserve the same
        effective RoPE phase differences after packing and therefore fail to
        exercise the intended behavioral distinction.
        """
        main_sequence_config = make_config(rope_mode="main_sequence")
        semantic_sequence_config = make_config(rope_mode="semantic_sequence")

        main_sequence_layer = make_layer(main_sequence_config, seed=0)
        semantic_sequence_layer = make_layer(semantic_sequence_config, seed=0)

        successful_distinctions = 0
        total_input_seeds = 10

        for input_seed in range(total_input_seeds):
            random_generator = torch.Generator(device="cpu")
            random_generator.manual_seed(input_seed)

            total_length = 14
            prefix_length = 10

            hidden_states = torch.randn(
                1,
                total_length,
                main_sequence_config.hidden_size,
                generator=random_generator,
            )
            position_ids = torch.arange(total_length, dtype=torch.long).unsqueeze(0)

            prefix_hidden_states = hidden_states[:, :prefix_length]
            prefix_position_ids = position_ids[:, :prefix_length]
            current_hidden_states = hidden_states[:, prefix_length:]
            current_position_ids = position_ids[:, prefix_length:]

            main_sequence_cache = make_layer_cache(
                main_sequence_config,
                batch_size=current_hidden_states.shape[0],
                initial_buffer_size=16,
            )
            semantic_sequence_cache = make_layer_cache(
                semantic_sequence_config,
                batch_size=current_hidden_states.shape[0],
                initial_buffer_size=16,
            )

            _, _, _ = main_sequence_layer(
                hidden_states=prefix_hidden_states,
                position_ids=prefix_position_ids,
                cache=main_sequence_cache,
            )
            _, _, _ = semantic_sequence_layer(
                hidden_states=prefix_hidden_states,
                position_ids=prefix_position_ids,
                cache=semantic_sequence_cache,
            )

            main_sequence_output, _, _ = main_sequence_layer(
                hidden_states=current_hidden_states,
                position_ids=current_position_ids,
                cache=main_sequence_cache,
            )
            semantic_sequence_output, _, _ = semantic_sequence_layer(
                hidden_states=current_hidden_states,
                position_ids=current_position_ids,
                cache=semantic_sequence_cache,
            )

            outputs_are_distinct = not torch.allclose(
                main_sequence_output,
                semantic_sequence_output,
                atol=1e-5,
                rtol=1e-5,
            )
            successful_distinctions += int(outputs_are_distinct)

        assert successful_distinctions > total_input_seeds // 2, (
            "Changing RoPE mode failed to change the warmed continued-decoding output "
            f"in a majority of deterministic random trials. "
            f"successful_distinctions={successful_distinctions}, "
            f"total_input_seeds={total_input_seeds}"
        )
    def test_changing_yarn_scale_changes_the_real_hybrid_output(self):
        """Changing sparse-path YaRN extrapolation scale should matter."""
        hidden_states, position_ids = make_inputs(start_position=0)

        standard_scale_layer = make_layer(
            make_config(
                training_sequence_length=8,
                inference_sequence_length=8,
            ),
            seed=0,
        )
        yarn_scaled_layer = make_layer(
            make_config(
                training_sequence_length=8,
                inference_sequence_length=16,
            ),
            seed=0,
        )

        standard_scale_output, _, _ = standard_scale_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        yarn_scaled_output, _, _ = yarn_scaled_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )

        assert not torch.allclose(standard_scale_output, yarn_scaled_output)


# ---------------------------------------------------------------------------
# The final responsibility preserved by the hybrid boundary is gradient
# separation. Gradients from the summed hybrid output should reach both real
# model-space paths, while gradients from load_balance_loss should reach the
# sparse balancing path without involving the local path.
# ---------------------------------------------------------------------------

class TestGradientBehavior:
    def test_hybrid_output_backward_reaches_both_real_model_space_paths(self):
        """Gradients from the hybrid output should reach both real subpaths."""
        hidden_states, position_ids = make_inputs(requires_grad=True, start_position=0)
        layer = make_layer(make_config(), seed=0)

        local_param = first_parameter(layer.local_attention)
        sparse_output_param = layer.sparse_attention.bea.q_proj
        sparse_balance_param = layer.sparse_attention.router.expert_bias

        hybrid_output, load_balance_loss, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        del load_balance_loss

        hybrid_output.sum().backward()

        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

        assert local_param.grad is not None
        assert torch.isfinite(local_param.grad).all()

        assert sparse_output_param.grad is not None
        assert torch.isfinite(sparse_output_param.grad).all()

        # The hybrid output should reach the sparse output path, not the sparse
        # balancing-only parameter.
        balance_grad = sparse_balance_param.grad
        assert balance_grad is None or torch.all(balance_grad == 0)

    def test_load_balance_loss_backward_reaches_the_sparse_balancing_path_and_causes_movement(self):
        """The returned load-balance loss should survive the hybrid layer and update expert_bias."""
        hidden_states, position_ids = make_inputs(requires_grad=True, start_position=0)
        layer = make_layer(make_config(), seed=0)

        local_param = first_parameter(layer.local_attention)
        sparse_output_param = layer.sparse_attention.bea.q_proj
        expert_bias = layer.sparse_attention.router.expert_bias
        expert_bias_before = expert_bias.detach().clone()

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        hybrid_output, load_balance_loss, _ = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=None,
        )
        del hybrid_output

        optimizer.zero_grad()
        load_balance_loss.backward()
        optimizer.step()

        assert expert_bias.grad is not None
        assert torch.isfinite(expert_bias.grad).all()
        assert not torch.allclose(expert_bias, expert_bias_before)

        # The hybrid layer should not redirect the sparse balancing signal into
        # the local path or the sparse output-path parameters.
        local_grad = local_param.grad
        assert local_grad is None or torch.all(local_grad == 0)

        sparse_output_grad = sparse_output_param.grad
        assert sparse_output_grad is None or torch.all(sparse_output_grad == 0)