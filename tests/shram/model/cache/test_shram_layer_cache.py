"""Tests for ShramLayerCache — Unit 6.B / Unit 14.D / Unit 19.G.4.

Invariants verified in this file:

HF CacheLayerMixin protocol and construction:
- ShramLayerCache subclasses CacheLayerMixin
- is_compileable is True
- is_initialized is True at construction
- sliding_window_cache is a LocalSlidingWindowLayerCache with the correct window
- mosrah_cache is a MoSRAHCache with the correct buffer shape (from config.mosrah_cache_length)
- lazy_initialization() is a no-op; is_initialized remains True
- update() raises NotImplementedError — no composite update semantics
- get_seq_length() delegates to sliding_window_cache — cumulative local token count
- get_max_cache_shape() returns config.inference_sequence_length
- get_mask_sizes() raises NotImplementedError — two paths, different mask semantics

Composite behaviors:
- reset() clears both sub-caches atomically through the ShramLayerCache boundary
- reorder_cache() permutes both sub-caches atomically through the ShramLayerCache boundary
"""

import torch
import pytest
from transformers.cache_utils import CacheLayerMixin

from src.shram.model.cache.shram_layer_cache import ShramLayerCache
from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.cache.sliding_window_cache import LocalSlidingWindowLayerCache
from src.shram.model.configuration import ShramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLIDING_WINDOW = 16
LOCAL_HEADS = 3
NUM_MOSRAH_HEADS = 4
NUM_SELECTED_HEADS = 2
HEAD_DIM = 8
TRAINING_SEQ_LEN = 64
INFERENCE_SEQ_LEN = 64
BATCH = 2


def make_config() -> ShramConfig:
    return ShramConfig(
        window_size=SLIDING_WINDOW,
        num_sliding_window_heads=LOCAL_HEADS,
        num_mosrah_heads=NUM_MOSRAH_HEADS,
        num_selected_heads=NUM_SELECTED_HEADS,
        head_dim=HEAD_DIM,
        training_sequence_length=TRAINING_SEQ_LEN,
        inference_sequence_length=INFERENCE_SEQ_LEN,
    )


def make_cache(batch: int = BATCH) -> ShramLayerCache:
    return ShramLayerCache(
        config=make_config(),
        batch_size=batch,
        device=torch.device("cpu"),
    )


def sw_update(cache: ShramLayerCache, batch: int, num_tokens: int) -> None:
    """Drive sliding_window_cache.update() with num_tokens of all-active random data."""
    k = torch.randn(batch, LOCAL_HEADS, num_tokens, HEAD_DIM)
    v = torch.randn(batch, LOCAL_HEADS, num_tokens, HEAD_DIM)
    mask = torch.ones(batch, num_tokens, dtype=torch.bool)
    positions = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cache.sliding_window_cache.update(k, v, mask, positions)


def mosrah_update(cache: ShramLayerCache, batch: int, num_tokens: int) -> None:
    """Drive mosrah_cache.update() with num_tokens active per head."""
    B, L, T, u = batch, NUM_MOSRAH_HEADS, num_tokens, HEAD_DIM
    k = torch.randn(B, L, T, u)
    v = torch.randn(B, L, T, u)
    mask = torch.ones(B, L, T, dtype=torch.bool)
    cache.mosrah_cache.update(k, v, mask)


# ---------------------------------------------------------------------------
# HF CacheLayerMixin protocol
# ---------------------------------------------------------------------------

def test_shram_layer_cache_is_cachelayermixin_subclass():
    """ShramLayerCache satisfies the HuggingFace per-layer cache role."""
    assert issubclass(ShramLayerCache, CacheLayerMixin)


def test_is_compileable_true():
    """is_compileable is True — ShramLayerCache participates in the static cache contract."""
    assert ShramLayerCache.is_compileable is True


def test_is_initialized_true_at_construction():
    """is_initialized is True at construction — both sub-caches pre-allocate storage."""
    cache = make_cache()
    assert cache.is_initialized is True


def test_lazy_initialization_is_noop():
    """lazy_initialization() completes without error and is_initialized remains True."""
    cache = make_cache()
    cache.lazy_initialization(torch.zeros(1), torch.zeros(1))
    assert cache.is_initialized is True


def test_update_raises():
    """update() raises NotImplementedError — no composite update interface exists."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.update(torch.zeros(1), torch.zeros(1))


def test_get_max_cache_shape_returns_inference_sequence_length():
    """get_max_cache_shape() returns config.inference_sequence_length."""
    cache = make_cache()
    assert cache.get_max_cache_shape() == INFERENCE_SEQ_LEN


def test_get_mask_sizes_raises():
    """get_mask_sizes() raises NotImplementedError — two paths have different mask semantics."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_mask_sizes(torch.tensor([0]))


# ---------------------------------------------------------------------------
# Construction — sub-cache ownership and shape
# ---------------------------------------------------------------------------

def test_owns_sliding_window_cache():
    """sliding_window_cache is a LocalSlidingWindowLayerCache."""
    cache = make_cache()
    assert isinstance(cache.sliding_window_cache, LocalSlidingWindowLayerCache)


def test_owns_mosrah_cache():
    """mosrah_cache is a MoSRAHCache."""
    cache = make_cache()
    assert isinstance(cache.mosrah_cache, MoSRAHCache)


def test_sliding_window_cache_has_correct_window():
    """sliding_window_cache.sliding_window matches config.window_size."""
    cache = make_cache()
    assert cache.sliding_window_cache.sliding_window == SLIDING_WINDOW


def test_get_seq_length_delegates_to_sliding_window_cache():
    """get_seq_length() returns the cumulative token count from the local sliding-window path."""
    cache = make_cache()
    sw_update(cache, BATCH, 5)
    assert cache.get_seq_length() == 5


def test_mosrah_cache_has_correct_buffer_shape():
    """mosrah_cache buffers have shape (B, L, mosrah_cache_length, u) at construction."""
    config = make_config()
    cache = make_cache()
    expected_len = config.mosrah_cache_length
    assert cache.mosrah_cache.keys.shape == (BATCH, NUM_MOSRAH_HEADS, expected_len, HEAD_DIM)
    assert cache.mosrah_cache.values.shape == (BATCH, NUM_MOSRAH_HEADS, expected_len, HEAD_DIM)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_mosrah_counts():
    """reset() clears the mosrah_cache so get_heads_lengths() returns all zeros."""
    cache = make_cache()
    mosrah_update(cache, BATCH, 2)
    assert cache.mosrah_cache.get_heads_lengths().sum() > 0
    cache.reset()
    assert cache.mosrah_cache.get_heads_lengths().sum() == 0


def test_reset_clears_sliding_window_active_mask():
    """reset() clears the sliding_window_cache active mask to all-False."""
    cache = make_cache()
    sw_update(cache, BATCH, 4)
    assert cache.sliding_window_cache.active_mask.any()
    cache.reset()
    assert not cache.sliding_window_cache.active_mask.any()


def test_reset_on_fresh_cache_is_idempotent():
    """reset() on a fresh cache does not raise and leaves both sub-caches in clean state."""
    cache = make_cache()
    cache.reset()
    assert not cache.sliding_window_cache.active_mask.any()
    assert cache.mosrah_cache.get_heads_lengths().sum() == 0


# ---------------------------------------------------------------------------
# reorder_cache()
# ---------------------------------------------------------------------------

def test_batch_repeat_interleave_expands_both_sub_caches():
    """batch_repeat_interleave() expands the batch dimension of both sub-caches atomically."""
    cache = make_cache(batch=2)
    sw_update(cache, 2, 1)
    mosrah_update(cache, 2, 1)
    cache.batch_repeat_interleave(3)
    assert cache.sliding_window_cache.keys.shape[0] == 6
    assert cache.mosrah_cache.batch_size == 6
    assert cache.mosrah_cache.keys.shape[0] == 6


def test_batch_select_indices_trims_both_sub_caches():
    """batch_select_indices() trims the batch dimension of both sub-caches atomically."""
    cache = make_cache(batch=4)
    sw_update(cache, 4, 1)
    mosrah_update(cache, 4, 1)
    indices = torch.tensor([0, 3])
    cache.batch_select_indices(indices)
    assert cache.sliding_window_cache.keys.shape[0] == 2
    assert cache.mosrah_cache.batch_size == 2
    assert cache.mosrah_cache.keys.shape[0] == 2


def test_reorder_cache_permutes_sliding_window_keys():
    """reorder_cache() permutes the batch dimension of sliding_window_cache.keys."""
    cache = make_cache(batch=3)
    sw_update(cache, 3, 2)
    original_keys = cache.sliding_window_cache.keys.clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.allclose(cache.sliding_window_cache.keys[0], original_keys[2])
    assert torch.allclose(cache.sliding_window_cache.keys[1], original_keys[0])
    assert torch.allclose(cache.sliding_window_cache.keys[2], original_keys[1])


def test_reorder_cache_permutes_mosrah_counts():
    """reorder_cache() permutes the batch dimension of mosrah_cache._counts."""
    cache = make_cache(batch=3)
    # Seed different counts per batch item by writing directly.
    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    cache.mosrah_cache._counts[:] = counts.clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.equal(cache.mosrah_cache._counts[0], counts[2])
    assert torch.equal(cache.mosrah_cache._counts[1], counts[0])
    assert torch.equal(cache.mosrah_cache._counts[2], counts[1])
