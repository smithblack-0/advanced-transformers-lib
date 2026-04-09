"""Tests for MoSRAHCache — Unit 6.A.

Invariants verified in this file:

HF CacheLayerMixin protocol and construction:
- MoSRAHCache subclasses CacheLayerMixin
- get_seq_length() raises NotImplementedError
- lazy_initialization() is a no-op; is_initialized remains True
- get_max_cache_shape() raises NotImplementedError
- get_mask_sizes() raises NotImplementedError
- get_heads_lengths() returns the (B, L) count tensor
- get_heads_lengths() returns zeros for a fresh cache
- get_heads_lengths() returns correct values after update() calls
- get_heads_lengths() returns correct values after repeated update() calls
- is_initialized is True at construction (storage is pre-allocated)
- is_initialized remains True after reset() (storage is still allocated)
- reset() zeroes all counts and buffers
- reorder_cache() permutes dim 0 of both buffers and counts atomically

update()/return value interface:
- update() returns a three-tensor tuple (keys, values, active_mask)
- returned shapes and dtypes are correct
- active_mask is True exactly for slots that have been written
- roundtrip: values written are retrievable in the returned tuple

Oracle agreement:
- update() produces identical keys, values, _counts to SlowMoSRAHCache on all test
  inputs. SlowMoSRAHCache is independently verified in test_slow_mosrah_cache.py;
  agreement with it on all cases licenses trust in the vectorized implementation.
  Cases covered: single active position; multiple heads; multiple tokens in one head;
  sparse mask; multi-call accumulation; uneven per-batch routing; buffer expansion.
- The returned (keys, values, active_mask) tuple also agrees between implementations.
"""

import torch
import pytest
from transformers.cache_utils import CacheLayerMixin

from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.cache.slow_mosrah_cache import SlowMoSRAHCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 4
HEAD_DIM = 16
BATCH = 2
INITIAL_BUFFER_SIZE = 8


def make_cache(
    batch: int = BATCH,
    initial_buffer_size: int = INITIAL_BUFFER_SIZE,
) -> MoSRAHCache:
    return MoSRAHCache(
        num_mosrah_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        batch_size=batch,
        device=torch.device("cpu"),
        initial_buffer_size=initial_buffer_size,
    )


def make_slow_cache(
    batch: int = BATCH,
    initial_buffer_size: int = INITIAL_BUFFER_SIZE,
) -> SlowMoSRAHCache:
    return SlowMoSRAHCache(
        num_mosrah_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        batch_size=batch,
        device=torch.device("cpu"),
        initial_buffer_size=initial_buffer_size,
    )


def seed_cache(cache: MoSRAHCache, counts: torch.Tensor) -> None:
    """Directly write known counts and matching key/value data into cache storage.

    Bypasses update() to allow reset(), reorder_cache(), and get_heads_lengths() to be
    tested against known non-zero state.
    """
    cache._counts[:] = counts.clone()
    for b in range(cache.batch_size):
        cache.keys[b] = float(b + 1)
        cache.values[b] = float(-(b + 1))


def _run_both(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    mask: torch.Tensor,
    batch: int,
    initial_buffer_size: int = INITIAL_BUFFER_SIZE,
) -> tuple[MoSRAHCache, SlowMoSRAHCache, tuple, tuple]:
    """Run the same update() call on both implementations and return both caches and results."""
    fast = make_cache(batch=batch, initial_buffer_size=initial_buffer_size)
    slow = make_slow_cache(batch=batch, initial_buffer_size=initial_buffer_size)
    fast_result = fast.update(key_states, value_states, mask)
    slow_result = slow.update(key_states, value_states, mask)
    return fast, slow, fast_result, slow_result


def _assert_agree(
    fast: MoSRAHCache,
    slow: SlowMoSRAHCache,
    fast_result: tuple,
    slow_result: tuple,
) -> None:
    """Assert that internal state and returned tuple agree exactly between implementations."""
    assert torch.equal(fast.keys, slow.keys), "keys differ"
    assert torch.equal(fast.values, slow.values), "values differ"
    assert torch.equal(fast._counts, slow._counts), "counts differ"

    fast_keys, fast_vals, fast_mask = fast_result
    slow_keys, slow_vals, slow_mask = slow_result
    assert torch.equal(fast_keys, slow_keys), "returned keys differ"
    assert torch.equal(fast_vals, slow_vals), "returned values differ"
    assert torch.equal(fast_mask, slow_mask), "returned active_mask differs"


# ---------------------------------------------------------------------------
# HF CacheLayerMixin protocol
# ---------------------------------------------------------------------------

def test_mosrah_cache_is_cachelayermixin_subclass():
    """MoSRAHCache satisfies the HuggingFace per-layer cache role."""
    assert issubclass(MoSRAHCache, CacheLayerMixin)


def test_get_seq_length_raises():
    """get_seq_length() raises NotImplementedError — no single length represents state."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_seq_length()


def test_lazy_initialization_is_noop():
    """lazy_initialization() completes without error and leaves is_initialized True.

    The cache is pre-allocated at construction time, so lazy_initialization() has
    no work to do. It must not raise: HF orchestration paths call it before first
    use and expect it to succeed.
    """
    cache = make_cache()
    cache.lazy_initialization(torch.zeros(1), torch.zeros(1))
    assert cache.is_initialized is True


def test_get_max_cache_shape_raises():
    """get_max_cache_shape() raises NotImplementedError — cache is unbounded."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_max_cache_shape()


def test_get_mask_sizes_raises():
    """get_mask_sizes() raises NotImplementedError — not used by MoSRAH path."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_mask_sizes(torch.tensor([0]))


# ---------------------------------------------------------------------------
# Construction and shape invariants
# ---------------------------------------------------------------------------

def test_buffer_shapes_at_construction():
    """Key and value buffers have shape (B, L, T_max, u); counts have shape (B, L)."""
    cache = make_cache()
    assert cache.keys.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert cache.values.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert cache._counts.shape == (BATCH, NUM_HEADS)


def test_buffer_capacity_at_construction():
    """buffer_capacity reflects the allocated slot count immediately after construction."""
    cache = make_cache()
    assert cache.buffer_capacity == INITIAL_BUFFER_SIZE


def test_counts_zero_at_construction():
    """All counts are zero immediately after construction."""
    cache = make_cache()
    assert cache._counts.sum() == 0


# ---------------------------------------------------------------------------
# is_initialized
# ---------------------------------------------------------------------------

def test_is_initialized_true_at_construction():
    """is_initialized is True immediately after construction — storage is pre-allocated."""
    cache = make_cache()
    assert cache.is_initialized is True


# ---------------------------------------------------------------------------
# get_heads_lengths
# ---------------------------------------------------------------------------

def test_get_heads_lengths_shape_on_fresh_cache():
    """get_heads_lengths() returns shape (B, L) even for a fresh cache."""
    cache = make_cache()
    lengths = cache.get_heads_lengths()
    assert lengths.shape == (BATCH, NUM_HEADS)


def test_get_heads_lengths_zeros_on_fresh_cache():
    """get_heads_lengths() returns all zeros before any update()."""
    cache = make_cache()
    assert cache.get_heads_lengths().sum() == 0


def test_get_heads_lengths_correct_after_update():
    """get_heads_lengths() returns the correct counts after update()."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 2, 0] = True
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    lengths = cache.get_heads_lengths()
    assert lengths[0, 0].item() == 1
    assert lengths[0, 2].item() == 1
    assert lengths[0, 1].item() == 0
    assert lengths[0, 3].item() == 0


def test_get_heads_lengths_correct_after_repeated_updates():
    """get_heads_lengths() accumulates correctly across multiple update() calls."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    assert cache.get_heads_lengths()[0, 0].item() == 2


def test_get_heads_lengths_returns_correct_seeded_values():
    """get_heads_lengths() returns the exact counts for a seeded cache."""
    cache = make_cache()
    expected = torch.tensor([[3, 1, 4, 2], [0, 5, 2, 3]])
    seed_cache(cache, counts=expected)
    assert torch.equal(cache.get_heads_lengths(), expected)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_zeroes_all_counts():
    """reset() sets all counts to zero."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.get_heads_lengths().sum() == 0


def test_reset_zeroes_key_buffers():
    """reset() zeroes the key buffer data."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.keys.sum() == 0


def test_reset_is_initialized_remains_true():
    """is_initialized remains True after reset — storage is still allocated."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.is_initialized is True


def test_reset_on_fresh_cache_is_idempotent():
    """reset() on a fresh cache does not raise and leaves state unchanged."""
    cache = make_cache()
    cache.reset()
    assert cache.is_initialized is True


def test_get_heads_lengths_zeros_after_reset():
    """get_heads_lengths() returns all zeros after reset()."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.get_heads_lengths().sum() == 0


def test_reset_allows_reuse():
    """After reset(), counts can be re-seeded and read back correctly."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()

    new_counts = torch.tensor([[1, 0, 2, 0], [0, 3, 1, 2]])
    seed_cache(cache, counts=new_counts)
    assert torch.equal(cache.get_heads_lengths(), new_counts)


# ---------------------------------------------------------------------------
# reorder_cache()
# ---------------------------------------------------------------------------

def test_reorder_cache_permutes_keys():
    """reorder_cache() permutes dim 0 of the key buffer."""
    cache = make_cache(batch=3)
    seed_cache(cache, counts=torch.zeros(3, NUM_HEADS, dtype=torch.long))
    original_keys = cache.keys.clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.allclose(cache.keys[0], original_keys[2])
    assert torch.allclose(cache.keys[1], original_keys[0])
    assert torch.allclose(cache.keys[2], original_keys[1])


def test_reorder_cache_permutes_values():
    """reorder_cache() permutes dim 0 of the value buffer."""
    cache = make_cache(batch=3)
    seed_cache(cache, counts=torch.zeros(3, NUM_HEADS, dtype=torch.long))
    original_values = cache.values.clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.allclose(cache.values[0], original_values[2])
    assert torch.allclose(cache.values[1], original_values[0])
    assert torch.allclose(cache.values[2], original_values[1])


def test_reorder_cache_permutes_counts():
    """reorder_cache() permutes dim 0 of the count tensor."""
    cache = make_cache(batch=3)
    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    seed_cache(cache, counts=counts)

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.equal(cache._counts[0], counts[2])
    assert torch.equal(cache._counts[1], counts[0])
    assert torch.equal(cache._counts[2], counts[1])


# ---------------------------------------------------------------------------
# batch_repeat_interleave()
# ---------------------------------------------------------------------------

def test_batch_repeat_interleave_expands_keys():
    """batch_repeat_interleave() expands dim 0 of the key buffer."""
    cache = make_cache(batch=2)
    seed_cache(cache, counts=torch.ones(2, NUM_HEADS, dtype=torch.long))
    original_keys = cache.keys.clone()

    cache.batch_repeat_interleave(3)

    assert cache.keys.shape[0] == 6
    assert torch.equal(cache.keys[0], original_keys[0])
    assert torch.equal(cache.keys[1], original_keys[0])
    assert torch.equal(cache.keys[2], original_keys[0])
    assert torch.equal(cache.keys[3], original_keys[1])


def test_batch_repeat_interleave_expands_counts():
    """batch_repeat_interleave() expands dim 0 of the count tensor."""
    cache = make_cache(batch=2)
    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    seed_cache(cache, counts=counts)

    cache.batch_repeat_interleave(2)

    assert torch.equal(cache._counts[0], counts[0])
    assert torch.equal(cache._counts[1], counts[0])
    assert torch.equal(cache._counts[2], counts[1])
    assert torch.equal(cache._counts[3], counts[1])


def test_batch_repeat_interleave_updates_batch_size():
    """batch_repeat_interleave() updates batch_size to B * repeats."""
    cache = make_cache(batch=2)
    cache.batch_repeat_interleave(4)
    assert cache.batch_size == 8


# ---------------------------------------------------------------------------
# batch_select_indices()
# ---------------------------------------------------------------------------

def test_batch_select_indices_selects_keys():
    """batch_select_indices() selects the correct rows from the key buffer."""
    cache = make_cache(batch=4)
    seed_cache(cache, counts=torch.ones(4, NUM_HEADS, dtype=torch.long))
    original_keys = cache.keys.clone()

    indices = torch.tensor([0, 3])
    cache.batch_select_indices(indices)

    assert cache.keys.shape[0] == 2
    assert torch.equal(cache.keys[0], original_keys[0])
    assert torch.equal(cache.keys[1], original_keys[3])


def test_batch_select_indices_selects_counts():
    """batch_select_indices() selects the correct rows from the count tensor."""
    cache = make_cache(batch=4)
    counts = torch.tensor([[1, 0, 2, 0], [3, 1, 0, 2], [0, 0, 1, 0], [4, 2, 1, 3]])
    seed_cache(cache, counts=counts)

    indices = torch.tensor([1, 3])
    cache.batch_select_indices(indices)

    assert torch.equal(cache._counts[0], counts[1])
    assert torch.equal(cache._counts[1], counts[3])


def test_batch_select_indices_updates_batch_size():
    """batch_select_indices() updates batch_size to the number of retained indices."""
    cache = make_cache(batch=4)
    cache.batch_select_indices(torch.tensor([0, 2]))
    assert cache.batch_size == 2


# ---------------------------------------------------------------------------
# update() return value
# ---------------------------------------------------------------------------

def test_update_returns_correct_shapes():
    """update() returns (keys, values, active_mask) with the correct shapes."""
    cache = make_cache()
    B, L, T = BATCH, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    keys, values, active_mask = cache.update(
        torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask
    )
    assert keys.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert values.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert active_mask.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE)


def test_update_active_mask_dtype():
    """update() returns active_mask as a bool tensor."""
    cache = make_cache()
    B, L, T = BATCH, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    _, _, active_mask = cache.update(
        torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask
    )
    assert active_mask.dtype == torch.bool


def test_update_active_mask_all_false_on_empty_write():
    """update() with all-False mask returns an all-False active_mask."""
    cache = make_cache()
    B, L, T = BATCH, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    _, _, active_mask = cache.update(
        torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask
    )
    assert not active_mask.any()


def test_update_active_mask_reflects_written_slots():
    """Returned active_mask is True exactly for slots that have been written."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 3
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True

    _, _, active_mask = cache.update(
        torch.randn(B, L, T, HEAD_DIM), torch.randn(B, L, T, HEAD_DIM), mask
    )

    assert active_mask[0, 0, 0].item() is True
    assert active_mask[0, 0, 1].item() is True
    assert active_mask[0, 0, 2].item() is False
    assert active_mask[0, 1, 0].item() is True
    assert active_mask[0, 1, 1].item() is False
    for h in [2, 3]:
        assert not active_mask[0, h, :].any()


def test_update_roundtrip():
    """Keys and values written via update() are retrievable in the returned tuple."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 2, 0] = True

    keys_out, vals_out, _ = cache.update(key, val, mask)

    assert torch.equal(keys_out[0, 2, 0, :], key[0, 2, 0, :])
    assert torch.equal(vals_out[0, 2, 0, :], val[0, 2, 0, :])


# ---------------------------------------------------------------------------
# Oracle agreement — MoSRAHCache vs SlowMoSRAHCache
#
# Each test runs identical inputs through both implementations and asserts that
# the resulting key buffers, value buffers, count tensors, and returned tuples
# are exactly equal. SlowMoSRAHCache was independently verified in
# test_slow_mosrah_cache.py; agreement with it licenses trust in the vectorized
# implementation.
# ---------------------------------------------------------------------------

def test_oracle_single_active_position():
    """Single active position in one head: vectorized matches oracle."""
    B, L, T = 1, NUM_HEADS, 1
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 2, 0] = True
    fast, slow, fr, sr = _run_both(key, val, mask, batch=B)
    _assert_agree(fast, slow, fr, sr)


def test_oracle_multiple_heads_same_step():
    """Multiple heads active in the same update() call: vectorized matches oracle."""
    B, L, T = 1, NUM_HEADS, 1
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 3, 0] = True
    fast, slow, fr, sr = _run_both(key, val, mask, batch=B)
    _assert_agree(fast, slow, fr, sr)


def test_oracle_multiple_tokens_one_head():
    """Multiple active positions in the same head: causal order preserved."""
    B, L, T = 1, NUM_HEADS, 4
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, :] = True
    fast, slow, fr, sr = _run_both(key, val, mask, batch=B)
    _assert_agree(fast, slow, fr, sr)


def test_oracle_sparse_mask():
    """Not all T positions active — only True positions written."""
    B, L, T = 1, NUM_HEADS, 3
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 2] = True
    mask[0, 1, 1] = True
    fast, slow, fr, sr = _run_both(key, val, mask, batch=B)
    _assert_agree(fast, slow, fr, sr)


def test_oracle_multi_call_accumulation():
    """Two sequential update() calls accumulate correctly."""
    B, L, T = 1, NUM_HEADS, 1
    fast = make_cache(batch=B)
    slow = make_slow_cache(batch=B)
    last_fr = last_sr = None
    for _ in range(2):
        key = torch.randn(B, L, T, HEAD_DIM)
        val = torch.randn(B, L, T, HEAD_DIM)
        mask = torch.zeros(B, L, T, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 1, 0] = True
        last_fr = fast.update(key, val, mask)
        last_sr = slow.update(key, val, mask)
    _assert_agree(fast, slow, last_fr, last_sr)


def test_oracle_uneven_batch():
    """Different batch items have different active heads — independence preserved."""
    B, L, T = 2, NUM_HEADS, 1
    key = torch.randn(B, L, T, HEAD_DIM)
    val = torch.randn(B, L, T, HEAD_DIM)
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 1, 0] = True
    mask[1, 2, 0] = True
    mask[1, 3, 0] = True
    fast, slow, fr, sr = _run_both(key, val, mask, batch=B)
    _assert_agree(fast, slow, fr, sr)


def test_oracle_buffer_expansion():
    """Vectorized expansion produces the same result as the oracle expansion."""
    B, L, T = 1, NUM_HEADS, 2
    fast = make_cache(batch=B, initial_buffer_size=2)
    slow = make_slow_cache(batch=B, initial_buffer_size=2)
    last_fr = last_sr = None
    for _ in range(2):
        key = torch.randn(B, L, T, HEAD_DIM)
        val = torch.randn(B, L, T, HEAD_DIM)
        mask = torch.zeros(B, L, T, dtype=torch.bool)
        mask[0, 0, :] = True
        last_fr = fast.update(key, val, mask)
        last_sr = slow.update(key, val, mask)
    assert fast.buffer_capacity == 4
    _assert_agree(fast, slow, last_fr, last_sr)
