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
        embedding_width=8,
        mlp_width=16,
        num_decoder_layers=2,
        num_sliding_window_heads=2,
        num_mosrah_heads=4,
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
    device: torch.device,
    seed: int = 0,
) -> DecoderLayer:
    torch.manual_seed(seed)
    return DecoderLayer(config).to(device)


def make_input(
    config: ShramConfig,
    device: torch.device,
    batch: int = 2,
    seq: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    random_generator = torch.Generator(device=str(device))
    random_generator.manual_seed(seed)

    x = torch.randn(
        batch,
        seq,
        config.embedding_width,
        generator=random_generator,
        device=device,
    )
    position_ids = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
    active_mask = torch.ones(batch, seq, dtype=torch.bool, device=device)
    return x, position_ids, active_mask


def make_layer_cache(
    config: ShramConfig,
    batch_size: int,
    device: torch.device,
) -> ShramLayerCache:
    return ShramLayerCache(
        config=config,
        batch_size=batch_size,
        device=device,
    )


# ---------------------------------------------------------------------------
# Inspection guards
# ---------------------------------------------------------------------------

class TestInspectionGuards:
    def test_decoder_layer_uses_shram_hybrid_layer(self, device):
        """DecoderLayer must wire in SHRAMHybridLayer, not legacy GQA."""
        config = small_config()
        layer = make_layer(config, device)

        assert isinstance(layer.attention, SHRAMHybridLayer)

    def test_two_rms_norms_do_not_alias_parameter_storage(self, device):
        """The two RMSNorm instances must have distinct learnable parameters."""
        config = small_config()
        layer = make_layer(config, device)

        assert layer.attn_norm is not layer.mlp_norm
        assert layer.attn_norm.weight.data_ptr() != layer.mlp_norm.weight.data_ptr()


# ---------------------------------------------------------------------------
# Runtime smoke tests
# ---------------------------------------------------------------------------

class TestRuntimeSmoke:
    def test_real_forward_returns_valid_output_and_scalar_regret_loss(self, device):
        """DecoderLayer should preserve (B, N, d) and return finite scalar loss."""
        config = small_config()
        layer = make_layer(config, device, seed=0)
        x, position_ids, active_mask = make_input(config, device, batch=2, seq=4, seed=1)

        output, router_diagnostics = layer(
            x,
            position_ids,
            active_mask,
            cache=None,
        )
        regret_loss = router_diagnostics["regret_loss"]

        assert output.shape == x.shape
        assert regret_loss.ndim == 0
        assert torch.isfinite(output).all()
        assert torch.isfinite(regret_loss)

    def test_output_responds_to_input_perturbation(self, device):
        """A real DecoderLayer should not be dead or bypassed with respect to x."""
        config = small_config()
        layer = make_layer(config, device, seed=0)
        x, position_ids, active_mask = make_input(config, device, batch=1, seq=4, seed=2)

        baseline_output, baseline_diagnostics = layer(
            x,
            position_ids,
            active_mask,
            cache=None,
        )
        baseline_regret_loss = baseline_diagnostics["regret_loss"]

        perturbed_x = x.clone()
        perturbed_x[:, 2, :] += 0.5

        perturbed_output, perturbed_diagnostics = layer(
            perturbed_x,
            position_ids,
            active_mask,
            cache=None,
        )
        perturbed_regret_loss = perturbed_diagnostics["regret_loss"]

        assert not torch.allclose(
            baseline_output,
            perturbed_output,
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.isfinite(baseline_regret_loss)
        assert torch.isfinite(perturbed_regret_loss)

    def test_output_responds_to_position_change_on_at_least_one_deterministic_seed(self, device):
        """Changing positions should affect the real DecoderLayer on at least one fixed input seed."""
        config = small_config()
        layer = make_layer(config, device, seed=0)
        # Open the residual gate so sublayer outputs are visible; the zero-init
        # default would make every output identical regardless of position.
        with torch.no_grad():
            layer.attn_residual_gate.fill_(1.0)
            layer.mlp_residual_gate.fill_(1.0)

        position_ids_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        position_ids_b = torch.tensor([[0, 3, 6, 9]], dtype=torch.long, device=device)

        successful_distinctions = 0
        input_seeds = list(range(10))

        for input_seed in input_seeds:
            x, _, active_mask = make_input(config, device, batch=1, seq=4, seed=input_seed)

            output_a, _ = layer(
                x,
                position_ids_a,
                active_mask,
                cache=None,
            )
            output_b, _ = layer(
                x,
                position_ids_b,
                active_mask,
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

    def test_real_cache_passthrough_smoke(self, device):
        """A real per-layer SHRAM cache should pass cleanly through DecoderLayer."""
        config = small_config()
        layer = make_layer(config, device, seed=0)

        x, position_ids, active_mask = make_input(config, device, batch=1, seq=4, seed=3)
        prefix_x = x[:, :2]
        prefix_position_ids = position_ids[:, :2]
        prefix_active_mask = active_mask[:, :2]
        current_x = x[:, 2:]
        current_position_ids = position_ids[:, 2:]
        current_active_mask = active_mask[:, 2:]

        layer_cache = make_layer_cache(
            config,
            batch_size=1,
            device=device,
        )

        prefix_output, prefix_diagnostics = layer(
            prefix_x,
            prefix_position_ids,
            prefix_active_mask,
            cache=layer_cache,
        )
        prefix_regret_loss = prefix_diagnostics["regret_loss"]
        current_output, current_diagnostics = layer(
            current_x,
            current_position_ids,
            current_active_mask,
            cache=layer_cache,
        )
        current_regret_loss = current_diagnostics["regret_loss"]

        assert prefix_output.shape == prefix_x.shape
        assert current_output.shape == current_x.shape
        assert prefix_regret_loss.ndim == 0
        assert current_regret_loss.ndim == 0
        assert torch.isfinite(prefix_output).all()
        assert torch.isfinite(current_output).all()
        assert torch.isfinite(prefix_regret_loss)
        assert torch.isfinite(current_regret_loss)