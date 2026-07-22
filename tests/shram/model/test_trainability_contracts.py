"""Trainability contracts for SHRAM initialization and router numerics."""

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
from src.shram.model.initialization import PROJECTION_INIT_STD


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


def assert_projection_scale(parameter: torch.Tensor, *, relative_tolerance: float) -> None:
    observed = parameter.detach().float().std().item()
    assert observed == pytest.approx(
        PROJECTION_INIT_STD,
        rel=relative_tolerance,
    )


def test_bea_projection_banks_use_rank_independent_scale() -> None:
    torch.manual_seed(0)
    layer = BottleneckedEnsembleAttention(small_config())

    assert_projection_scale(layer.q_proj, relative_tolerance=0.08)
    assert_projection_scale(layer.k_proj, relative_tolerance=0.08)
    assert_projection_scale(layer.v_proj, relative_tolerance=0.08)
    assert_projection_scale(layer.o_proj, relative_tolerance=0.08)


def test_router_projection_uses_model_projection_scale() -> None:
    torch.manual_seed(0)
    router = MoSRAHRouter(small_config())
    assert_projection_scale(router.routing_weight, relative_tolerance=0.20)


def test_router_entmax_outputs_are_normalized_and_finite() -> None:
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
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-6,
    )
    assert diagnostics["regret_loss"].dtype == torch.float32
    assert torch.isfinite(diagnostics["regret_loss"])
    assert torch.isfinite(diagnostics["logit_regret"])
    assert torch.isfinite(diagnostics["logit_std"])


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
