"""Tests for ShramCache — Unit 6.C / Unit 14.D.

Invariants verified in this file:

HF Cache protocol and construction:
- ShramCache subclasses Cache
- ShramCache owns exactly one ShramLayerCache per decoder layer
- All owned layer caches are ShramLayerCache instances
- is_initialized is True immediately after construction
- update() raises NotImplementedError — no composite update interface
- get_seq_length() raises NotImplementedError — no scalar sequence length available
- crop() raises NotImplementedError
- max_batch_size raises NotImplementedError
- max_cache_len raises NotImplementedError

Composite behaviors:
- reset() clears all layer caches through the ShramCache boundary
- reorder_cache() permutes all layer caches consistently through the ShramCache boundary
- len(cache) == num_hidden_layers
"""

import torch
import pytest
from transformers.cache_utils import Cache

from src.shram.model.cache.shram_cache import ShramCache
from src.shram.model.cache.shram_layer_cache import ShramLayerCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 3
SLIDING_WINDOW = 16
NUM_MOSRAH_HEADS = 4
MOSRAH_HEAD_DIM = 8
BATCH = 2
INITIAL_BUFFER_SIZE = 8

LOCAL_HEADS = 3
LOCAL_HEAD_DIM = 8


def make_cache(num_layers: int = NUM_LAYERS, batch: int = BATCH) -> ShramCache:
    return ShramCache(
        num_hidden_layers=num_layers,
        sliding_window=SLIDING_WINDOW,
        num_local_heads=LOCAL_HEADS,
        local_head_dim=LOCAL_HEAD_DIM,
        num_mosrah_heads=NUM_MOSRAH_HEADS,
        mosrah_head_dim=MOSRAH_HEAD_DIM,
        batch_size=batch,
        device=torch.device("cpu"),
        initial_buffer_size=INITIAL_BUFFER_SIZE,
    )


def sw_update(layer_cache: ShramLayerCache, batch: int, num_tokens: int) -> None:
    """Drive a layer's sliding_window_cache.update() with all-active random data."""
    k = torch.randn(batch, LOCAL_HEADS, num_tokens, LOCAL_HEAD_DIM)
    v = torch.randn(batch, LOCAL_HEADS, num_tokens, LOCAL_HEAD_DIM)
    mask = torch.ones(batch, num_tokens, dtype=torch.bool)
    layer_cache.sliding_window_cache.update(k, v, mask)


def mosrah_update(layer_cache: ShramLayerCache, batch: int, num_tokens: int) -> None:
    """Drive a layer's mosrah_cache.update() with all-active mask."""
    B, L, T, u = batch, NUM_MOSRAH_HEADS, num_tokens, MOSRAH_HEAD_DIM
    k = torch.randn(B, L, T, u)
    v = torch.randn(B, L, T, u)
    mask = torch.ones(B, L, T, dtype=torch.bool)
    layer_cache.mosrah_cache.update(k, v, mask)


# ---------------------------------------------------------------------------
# HF Cache protocol
# ---------------------------------------------------------------------------

def test_shram_cache_is_cache_subclass():
    """ShramCache satisfies the HuggingFace top-level Cache role."""
    assert issubclass(ShramCache, Cache)


def test_update_raises():
    """update() raises NotImplementedError — no composite update interface exists."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.update(torch.zeros(1), torch.zeros(1), layer_idx=0)


def test_crop_raises():
    """crop() raises NotImplementedError — ShramLayerCache does not implement crop."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.crop(max_length=4)


def test_batch_repeat_interleave_expands_all_mosrah_caches():
    """batch_repeat_interleave() expands the MoSRAH batch dimension in every layer."""
    cache = make_cache(batch=2)
    repeats = 3
    cache.batch_repeat_interleave(repeats)
    for layer in cache.layers:
        assert layer.mosrah_cache.batch_size == 2 * repeats
        assert layer.mosrah_cache.keys.shape[0] == 2 * repeats


def test_batch_select_indices_trims_all_mosrah_caches():
    """batch_select_indices() trims the MoSRAH batch dimension in every layer."""
    cache = make_cache(batch=4)
    indices = torch.tensor([0, 2])
    cache.batch_select_indices(indices)
    for layer in cache.layers:
        assert layer.mosrah_cache.batch_size == 2
        assert layer.mosrah_cache.keys.shape[0] == 2


def test_max_batch_size_raises():
    """max_batch_size raises NotImplementedError."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        _ = cache.max_batch_size


def test_max_cache_len_raises():
    """max_cache_len raises NotImplementedError — composite has no single maximum."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        _ = cache.max_cache_len


# ---------------------------------------------------------------------------
# Construction — ownership and shape
# ---------------------------------------------------------------------------

def test_len_equals_num_layers():
    """len(cache) == num_hidden_layers."""
    cache = make_cache(num_layers=5)
    assert len(cache) == 5


def test_all_layers_are_shram_layer_cache():
    """Every entry in cache.layers is a ShramLayerCache instance."""
    cache = make_cache()
    assert all(isinstance(layer, ShramLayerCache) for layer in cache.layers)


def test_is_initialized_true_at_construction():
    """is_initialized is True at construction — both sub-caches pre-allocate storage."""
    cache = make_cache()
    assert cache.is_initialized is True


# ---------------------------------------------------------------------------
# get_seq_length — not supported
# ---------------------------------------------------------------------------

def test_get_seq_length_returns_cumulative_token_count():
    """get_seq_length() returns the cumulative processed token count from layer 0's local path."""
    cache = make_cache()
    sw_update(cache.layers[0], BATCH, 4)
    assert cache.get_seq_length() == 4


def test_get_seq_length_uses_specified_layer():
    """get_seq_length(layer_idx) delegates to the correct layer."""
    cache = make_cache(num_layers=3)
    sw_update(cache.layers[1], BATCH, 7)
    assert cache.get_seq_length(layer_idx=1) == 7


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_all_sliding_window_caches():
    """reset() clears the sliding-window cache active mask in every layer."""
    cache = make_cache()
    for layer in cache.layers:
        sw_update(layer, BATCH, 3)
    cache.reset()
    for layer in cache.layers:
        assert not layer.sliding_window_cache.active_mask.any()


def test_reset_clears_all_mosrah_caches():
    """reset() clears the MoSRAH cache in every layer."""
    cache = make_cache()
    for layer in cache.layers:
        mosrah_update(layer, BATCH, 2)
    cache.reset()
    for layer in cache.layers:
        assert layer.mosrah_cache.get_heads_lengths().sum() == 0


def test_reset_on_fresh_cache_is_idempotent():
    """reset() on a fresh cache does not raise and leaves all layers in clean state."""
    cache = make_cache()
    cache.reset()
    for layer in cache.layers:
        assert not layer.sliding_window_cache.active_mask.any()
    for layer in cache.layers:
        assert layer.mosrah_cache.get_heads_lengths().sum() == 0


# ---------------------------------------------------------------------------
# reorder_cache()
# ---------------------------------------------------------------------------

def test_reorder_cache_permutes_all_sliding_window_caches():
    """reorder_cache() permutes the batch dimension of every layer's sliding-window cache."""
    cache = make_cache(batch=3)
    for layer in cache.layers:
        sw_update(layer, 3, 2)

    originals = [layer.sliding_window_cache.keys.clone() for layer in cache.layers]
    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    for i, layer in enumerate(cache.layers):
        assert torch.allclose(layer.sliding_window_cache.keys[0], originals[i][2])
        assert torch.allclose(layer.sliding_window_cache.keys[1], originals[i][0])
        assert torch.allclose(layer.sliding_window_cache.keys[2], originals[i][1])


def test_reorder_cache_permutes_all_mosrah_caches():
    """reorder_cache() permutes the batch dimension of every layer's MoSRAH cache."""
    cache = make_cache(batch=3)
    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    for layer in cache.layers:
        layer.mosrah_cache._counts[:] = counts.clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    for layer in cache.layers:
        assert torch.equal(layer.mosrah_cache._counts[0], counts[2])
        assert torch.equal(layer.mosrah_cache._counts[1], counts[0])
        assert torch.equal(layer.mosrah_cache._counts[2], counts[1])
