"""Smoke tests for DecoderLayer.

Unit 12 coverage strategy:
- light inspection guards for obvious structural ownership
- real runtime smoke tests at the DecoderLayer public boundary
- no re-proof of Unit 11 attention semantics or brittle symbolic rewrites
"""

import torch

from src.shram.model.attention.shram import SHRAMHybridLayer
from src.shram.model.cache.shram_layer_cache import ShramLayerCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.decoder_layer import DecoderLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_sliding_window_heads=2,
        num_mosrah_heads=5,
        num_selected_heads=2,
        head_dim=4,
        window_size=4,
        rope_mode="main_sequence",
        local_rope_theta=10000.0,
        mosrah_rope_theta=10000.0,
        training_sequence_length=16,
        inference_sequence_length=16,
        alpha=1.0,
        beta=32.0,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        use_cache=True,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def make_layer(
    config: ShramConfig,
    seed: int = 0,
) -> DecoderLayer:
    torch.manual_seed(seed)
    return DecoderLayer(config)


def make_input(
    config: ShramConfig,
    batch: int = 2,
    seq: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(seed)

    x = torch.randn(
        batch,
        seq,
        config.hidden_size,
        generator=random_generator,
    )
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    return x, position_ids


def make_layer_cache(
    config: ShramConfig,
    batch_size: int,
    initial_buffer_size: int = 8,
) -> ShramLayerCache:
    return ShramLayerCache(
        sliding_window=config.window_size,
        num_mosrah_heads=config.num_mosrah_heads,
        mosrah_head_dim=config.head_dim,
        batch_size=batch_size,
        device=torch.device("cpu"),
        initial_buffer_size=initial_buffer_size,
    )


# ---------------------------------------------------------------------------
# Inspection guards
# ---------------------------------------------------------------------------

class TestInspectionGuards:
    def test_decoder_layer_uses_shram_hybrid_layer(self):
        """DecoderLayer must wire in SHRAMHybridLayer, not legacy GQA."""
        config = small_config()
        layer = make_layer(config)

        assert isinstance(layer.attention, SHRAMHybridLayer)

    def test_two_rms_norms_do_not_alias_parameter_storage(self):
        """The two RMSNorm instances must have distinct learnable parameters."""
        config = small_config()
        layer = make_layer(config)

        assert layer.attn_norm is not layer.mlp_norm
        assert layer.attn_norm.weight.data_ptr() != layer.mlp_norm.weight.data_ptr()


# ---------------------------------------------------------------------------
# Runtime smoke tests
# ---------------------------------------------------------------------------

class TestRuntimeSmoke:
    def test_real_forward_returns_valid_output_and_scalar_load_balance_loss(self):
        """DecoderLayer should preserve (B, N, d) and return finite scalar loss."""
        config = small_config()
        layer = make_layer(config, seed=0)
        x, position_ids = make_input(config, batch=2, seq=4, seed=1)

        output, load_balance_loss = layer(
            x,
            position_ids,
            cache=None,
        )

        assert output.shape == x.shape
        assert load_balance_loss.ndim == 0
        assert torch.isfinite(output).all()
        assert torch.isfinite(load_balance_loss)

    def test_output_responds_to_input_perturbation(self):
        """A real DecoderLayer should not be dead or bypassed with respect to x."""
        config = small_config()
        layer = make_layer(config, seed=0)
        x, position_ids = make_input(config, batch=1, seq=4, seed=2)

        baseline_output, baseline_load_balance_loss = layer(
            x,
            position_ids,
            cache=None,
        )

        perturbed_x = x.clone()
        perturbed_x[:, 2, :] += 0.5

        perturbed_output, perturbed_load_balance_loss = layer(
            perturbed_x,
            position_ids,
            cache=None,
        )

        assert not torch.allclose(
            baseline_output,
            perturbed_output,
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.isfinite(baseline_load_balance_loss)
        assert torch.isfinite(perturbed_load_balance_loss)

    def test_output_responds_to_position_change_on_at_least_one_deterministic_seed(self):
        """Changing positions should affect the real DecoderLayer on at least one fixed input seed."""
        config = small_config()
        layer = make_layer(config, seed=0)

        position_ids_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        position_ids_b = torch.tensor([[0, 3, 6, 9]], dtype=torch.long)

        successful_distinctions = 0
        input_seeds = list(range(10))

        for input_seed in input_seeds:
            x, _ = make_input(config, batch=1, seq=4, seed=input_seed)

            output_a, _ = layer(
                x,
                position_ids_a,
                cache=None,
            )
            output_b, _ = layer(
                x,
                position_ids_b,
                cache=None,
            )

            outputs_are_distinct = not torch.allclose(
                output_a,
                output_b,
                atol=1e-5,
                rtol=1e-5,
            )
            successful_distinctions += int(outputs_are_distinct)

        assert successful_distinctions >= 1, (
            "Changing position_ids never changed the DecoderLayer output across the fixed seed set. "
            f"successful_distinctions={successful_distinctions}, total_trials={len(input_seeds)}"
        )

    def test_real_cache_passthrough_smoke(self):
        """A real per-layer SHRAM cache should pass cleanly through DecoderLayer."""
        config = small_config()
        layer = make_layer(config, seed=0)

        x, position_ids = make_input(config, batch=1, seq=4, seed=3)
        prefix_x = x[:, :2]
        prefix_position_ids = position_ids[:, :2]
        current_x = x[:, 2:]
        current_position_ids = position_ids[:, 2:]

        layer_cache = make_layer_cache(
            config,
            batch_size=1,
            initial_buffer_size=8,
        )

        prefix_output, prefix_load_balance_loss = layer(
            prefix_x,
            prefix_position_ids,
            cache=layer_cache,
        )
        current_output, current_load_balance_loss = layer(
            current_x,
            current_position_ids,
            cache=layer_cache,
        )

        assert prefix_output.shape == prefix_x.shape
        assert current_output.shape == current_x.shape
        assert prefix_load_balance_loss.ndim == 0
        assert current_load_balance_loss.ndim == 0
        assert torch.isfinite(prefix_output).all()
        assert torch.isfinite(current_output).all()
        assert torch.isfinite(prefix_load_balance_loss)
        assert torch.isfinite(current_load_balance_loss)