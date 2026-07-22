"""Trainability contracts for SHRAM initialization and router numerics."""

import math

import pytest
import torch

from src.shram.model.attention.bottlenecked_ensemble_attention import (
    BottleneckedEnsembleAttention,
)
from src.shram.model.attention.router import MoSRAHRouter
from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.cache.router_cache import RouterCache
from src.shram.model.cache.shram_layer_cache import ShramLayerCache
from src.shram.model.cache.slow_mosrah_cache import SlowMoSRAHCache
from src.shram.model.cache.sliding_window_cache import LocalSlidingWindowLayerCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.initialization import ROUTER_INIT_STD


def small_config(**overrides) -> ShramConfig:
    values = {
        "vocab_size": 128,
        "embedding_width": 64,
        "mlp_width": 128,
        "num_decoder_layers": 2,
        "num_sliding_window_heads": 4,
        "num_mosrah_heads": 8,
        "num_selected_heads": 4,
        "head_dim": 16,
        "window_size": 16,
        "training_sequence_length": 16,
        "inference_sequence_length": 16,
        "use_cache": False,
    }
    values.update(overrides)
    return ShramConfig(**values)


def assert_expert_bank_uses_per_matrix_xavier(bank: torch.Tensor) -> None:
    """Verify every stored expert matrix follows its own fan geometry."""
    for matrix in bank.detach().float().unbind(dim=0):
        fan_in, fan_out = nn_fans(matrix)
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        assert matrix.std().item() == pytest.approx(expected_std, rel=0.12)


def nn_fans(matrix: torch.Tensor) -> tuple[int, int]:
    """Return fan-in/fan-out for one two-dimensional linear matrix."""
    assert matrix.ndim == 2
    return matrix.shape[0], matrix.shape[1]


def test_bea_projection_banks_use_each_experts_fan_geometry() -> None:
    torch.manual_seed(0)
    layer = BottleneckedEnsembleAttention(small_config())

    assert_expert_bank_uses_per_matrix_xavier(layer.q_proj)
    assert_expert_bank_uses_per_matrix_xavier(layer.k_proj)
    assert_expert_bank_uses_per_matrix_xavier(layer.v_proj)
    assert_expert_bank_uses_per_matrix_xavier(layer.o_proj)


def test_router_projection_uses_balance_initialization_scale() -> None:
    torch.manual_seed(0)
    router = MoSRAHRouter(small_config())
    observed = router.routing_weight.detach().float().std().item()
    assert observed == pytest.approx(ROUTER_INIT_STD, rel=0.20)


def test_router_entmax_outputs_are_finite_and_have_valid_total_mass() -> None:
    torch.manual_seed(0)
    config = small_config()
    router = MoSRAHRouter(config)
    hidden = torch.randn(2, 16, config.embedding_width)
    active = torch.ones(2, 16, dtype=torch.bool)

    selected, probabilities, diagnostics = router(hidden, active)

    assert selected.shape == (2, 16, config.num_selected_heads)
    assert probabilities.shape == selected.shape
    assert probabilities.dtype == hidden.dtype
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()

    probability_mass = probabilities.sum(dim=-1)
    has_no_sparse_contribution = probability_mass == 0
    has_normalized_sparse_contribution = torch.isclose(
        probability_mass,
        torch.ones_like(probability_mass),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(has_no_sparse_contribution | has_normalized_sparse_contribution)

    assert diagnostics["regret_loss"].dtype == torch.float32
    assert torch.isfinite(diagnostics["regret_loss"])
    assert torch.isfinite(diagnostics["logit_regret"])
    assert torch.isfinite(diagnostics["logit_std"])


def test_forced_zero_support_assignment_skips_sparse_contribution(monkeypatch) -> None:
    """A token with no selected Entmax mass should contribute nothing sparsely."""
    config = small_config()
    router = MoSRAHRouter(config)
    hidden = torch.zeros(1, config.block_length, config.embedding_width)
    active = torch.ones(1, config.block_length, dtype=torch.bool)

    # Both tokens strongly prefer experts 0..3. The first token claims them,
    # forcing the second token onto experts 4..7, all outside its Entmax support.
    forced_logits = torch.tensor(
        [[
            [100.0, 99.0, 98.0, 97.0, -100.0, -101.0, -102.0, -103.0],
            [100.0, 99.0, 98.0, 97.0, -100.0, -101.0, -102.0, -103.0],
        ]],
        dtype=hidden.dtype,
    )
    monkeypatch.setattr(router, "_compute_routing_logits", lambda _x: forced_logits)

    selected, probabilities, _ = router(hidden, active)

    assert torch.equal(selected[0, 1].sort().values, torch.tensor([4, 5, 6, 7]))
    torch.testing.assert_close(
        probabilities[0, 1],
        torch.zeros(config.num_selected_heads),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        probabilities[0, 0].sum(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_router_entmax_task_gradient_reaches_projection_and_input() -> None:
    torch.manual_seed(0)
    config = small_config()
    router = MoSRAHRouter(config)
    hidden = torch.randn(
        2,
        16,
        config.embedding_width,
        requires_grad=True,
    )
    active = torch.ones(2, 16, dtype=torch.bool)

    _, probabilities, _ = router(hidden, active)
    coefficient = torch.arange(
        config.num_selected_heads,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    loss = (probabilities * coefficient).sum()
    loss.backward()

    assert router.routing_weight.grad is not None
    assert torch.isfinite(router.routing_weight.grad).all()
    assert router.routing_weight.grad.abs().sum().item() > 0.0
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum().item() > 0.0


def test_cache_layers_implement_current_max_length_contract() -> None:
    """Every SHRAM CacheLayerMixin owner must expose truthful maximum length."""
    device = torch.device("cpu")
    config = small_config(use_cache=True)

    local_cache = LocalSlidingWindowLayerCache(
        sliding_window=config.window_size,
        num_heads=config.num_sliding_window_heads,
        head_dim=config.head_dim,
        batch_size=2,
        device=device,
    )
    mosrah_cache = MoSRAHCache(
        num_mosrah_heads=config.num_mosrah_heads,
        head_dim=config.head_dim,
        batch_size=2,
        device=device,
        mosrah_cache_length=config.mosrah_cache_length,
    )
    slow_mosrah_cache = SlowMoSRAHCache(
        num_mosrah_heads=config.num_mosrah_heads,
        head_dim=config.head_dim,
        batch_size=2,
        device=device,
        mosrah_cache_length=config.mosrah_cache_length,
    )
    router_cache = RouterCache(
        block_length=config.block_length,
        num_mosrah_heads=config.num_mosrah_heads,
        batch_size=2,
        device=device,
    )
    layer_cache = ShramLayerCache(
        config=config,
        batch_size=2,
        device=device,
    )

    expected_lengths = {
        local_cache: config.window_size,
        mosrah_cache: config.mosrah_cache_length,
        slow_mosrah_cache: config.mosrah_cache_length,
        router_cache: -1,
        layer_cache: config.inference_sequence_length,
    }
    for cache, expected in expected_lengths.items():
        assert cache.get_max_length() == expected
        assert cache.get_max_cache_shape() == expected
