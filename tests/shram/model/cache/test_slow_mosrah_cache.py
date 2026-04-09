"""Tests for SlowMoSRAHCache — Unit 6.A oracle establishment.

SlowMoSRAHCache is the correctness oracle against which MoSRAHCache.update() is
validated. Its tests therefore carry extra weight: they must establish independent
trust in the oracle before it can license trust in the production implementation.
These tests verify SlowMoSRAHCache directly against known expected values — not
against another implementation — so that the trust chain has a solid foundation.

Invariants verified in this file:
- SlowMoSRAHCache subclasses CacheLayerMixin
- is_initialized is True immediately after construction
- get_seq_length() raises NotImplementedError
- lazy_initialization() is a no-op
- update() is the primary interface — stores and returns in one call
- get_max_cache_shape() raises NotImplementedError
- get_mask_sizes() raises NotImplementedError
- get_heads_lengths() returns the (B, L) count tensor
- get_heads_lengths() returns zeros for a fresh cache
- get_heads_lengths() returns correct values after update() calls
- get_heads_lengths() returns correct values after repeated update() calls
- get_heads_lengths() returns zeros after reset()
- buffer_capacity reflects the current allocated slot count
- reset() zeroes all counts and buffers; is_initialized remains True
- reorder_cache() permutes dim 0 of both buffers and counts atomically
- update() stores key/value at the correct (batch, head, slot) position
- update() increments counts correctly per (batch, head)
- update() preserves causal ordering: tokens appear in T order within each slot
- update() accumulates across calls: second call writes after first call's data
- update() is sparse: inactive positions (mask False) are not written
- update() is batch-independent: different batch items accumulate without cross-contamination
- update() triggers buffer expansion when any slot would overflow; existing data preserved
- update() returns (keys, values, active_mask) with correct shapes and dtypes
- returned active_mask is True exactly for slots that have been written
- update() roundtrip: keys and values written are retrievable in the returned tuple
"""

import torch
import pytest
from transformers.cache_utils import CacheLayerMixin

from src.shram.model.cache.slow_mosrah_cache import SlowMoSRAHCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 4
HEAD_DIM = 4   # small so expected values are easy to read in assertions
BATCH = 2
INITIAL_BUFFER_SIZE = 8


def make_cache(
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


def seed_cache(cache: SlowMoSRAHCache, counts: torch.Tensor) -> None:
    """Directly write known counts and matching key/value data into cache storage.

    Used for reset() and reorder_cache() tests that need known non-zero state without
    going through update().
    """
    cache._counts[:] = counts.clone()
    for b in range(cache.batch_size):
        cache.keys[b] = float(b + 1)
        cache.values[b] = float(-(b + 1))


# ---------------------------------------------------------------------------
# HF CacheLayerMixin protocol
# ---------------------------------------------------------------------------

def test_slow_mosrah_cache_is_cachelayermixin_subclass():
    """SlowMoSRAHCache satisfies the HuggingFace per-layer cache role."""
    assert issubclass(SlowMoSRAHCache, CacheLayerMixin)


def test_is_initialized_true_at_construction():
    """is_initialized is True immediately after construction — storage is pre-allocated."""
    cache = make_cache()
    assert cache.is_initialized is True


def test_get_seq_length_raises():
    """get_seq_length() raises NotImplementedError — no single length represents state."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_seq_length()


def test_lazy_initialization_is_noop():
    """lazy_initialization() completes without error — storage is already allocated."""
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
    """Key and value buffers have shape (B, L, T, u); counts have shape (B, L)."""
    cache = make_cache()
    assert cache.keys.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert cache.values.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert cache.get_heads_lengths().shape == (BATCH, NUM_HEADS)


def test_counts_zero_at_construction():
    """All counts are zero immediately after construction."""
    cache = make_cache()
    assert cache.get_heads_lengths().sum() == 0


def test_buffer_capacity_at_construction():
    """buffer_capacity reflects the initial allocated slot count."""
    cache = make_cache()
    assert cache.buffer_capacity == INITIAL_BUFFER_SIZE


# ---------------------------------------------------------------------------
# get_heads_lengths
# ---------------------------------------------------------------------------

def test_get_heads_lengths_shape_on_fresh_cache():
    """get_heads_lengths() returns shape (B, L) for a fresh cache."""
    cache = make_cache()
    assert cache.get_heads_lengths().shape == (BATCH, NUM_HEADS)


def test_get_heads_lengths_zeros_on_fresh_cache():
    """get_heads_lengths() returns all zeros before any update()."""
    cache = make_cache()
    assert cache.get_heads_lengths().sum() == 0


def test_get_heads_lengths_correct_after_update():
    """get_heads_lengths() returns the correct counts immediately after update()."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 3, 0] = True
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    lengths = cache.get_heads_lengths()
    assert lengths[0, 0].item() == 1
    assert lengths[0, 3].item() == 1
    assert lengths[0, 1].item() == 0
    assert lengths[0, 2].item() == 0


def test_get_heads_lengths_correct_after_repeated_updates():
    """get_heads_lengths() accumulates correctly across multiple update() calls."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    assert cache.get_heads_lengths()[0, 0].item() == 2


def test_get_heads_lengths_seeded_values():
    """get_heads_lengths() returns the exact counts for a seeded cache."""
    cache = make_cache()
    expected = torch.tensor([[3, 1, 4, 2], [0, 5, 2, 3]])
    seed_cache(cache, counts=expected)
    assert torch.equal(cache.get_heads_lengths(), expected)


def test_get_heads_lengths_zeros_after_reset():
    """get_heads_lengths() returns all zeros after reset()."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.get_heads_lengths().sum() == 0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_zeroes_key_buffers():
    """reset() zeroes the key buffer data."""
    cache = make_cache()
    seed_cache(cache, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.keys.sum() == 0


def test_reset_is_initialized_remains_true():
    """is_initialized remains True after reset — storage is still allocated."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)
    cache.reset()
    assert cache.is_initialized is True


def test_reset_on_fresh_cache_is_idempotent():
    """reset() on a fresh cache does not raise and leaves state unchanged."""
    cache = make_cache()
    cache.reset()
    assert cache.is_initialized is True


def test_reset_allows_reuse():
    """After reset(), update() writes new data starting from position 0."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True

    cache.update(torch.full((B, L, T, HEAD_DIM), 9.0), torch.ones(B, L, T, HEAD_DIM), mask)
    cache.reset()

    new_keys = torch.full((B, L, T, HEAD_DIM), 7.0)
    cache.update(new_keys, torch.ones(B, L, T, HEAD_DIM), mask)

    assert torch.equal(cache.keys[0, 0, 0, :], new_keys[0, 0, 0, :])
    assert cache.get_heads_lengths()[0, 0].item() == 1


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

    assert torch.equal(cache.get_heads_lengths()[0], counts[2])
    assert torch.equal(cache.get_heads_lengths()[1], counts[0])
    assert torch.equal(cache.get_heads_lengths()[2], counts[1])


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

    assert torch.equal(cache.get_heads_lengths()[0], counts[0])
    assert torch.equal(cache.get_heads_lengths()[1], counts[0])
    assert torch.equal(cache.get_heads_lengths()[2], counts[1])
    assert torch.equal(cache.get_heads_lengths()[3], counts[1])


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

    assert torch.equal(cache.get_heads_lengths()[0], counts[1])
    assert torch.equal(cache.get_heads_lengths()[1], counts[3])


def test_batch_select_indices_updates_batch_size():
    """batch_select_indices() updates batch_size to the number of retained indices."""
    cache = make_cache(batch=4)
    cache.batch_select_indices(torch.tensor([0, 2]))
    assert cache.batch_size == 2


# ---------------------------------------------------------------------------
# update() — single step correctness
#
# Each test builds a small known input, calls update(), and checks the internal
# buffer directly against expected values. These exact-value assertions establish
# the oracle's trustworthiness without depending on another implementation.
# ---------------------------------------------------------------------------

def test_update_key_to_correct_slot():
    """update() stores the key at the correct (batch, head, slot) position."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    key_states = torch.zeros(B, L, T, HEAD_DIM)
    key_states[0, 2, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 2, 0] = True

    cache.update(key_states, torch.zeros(B, L, T, HEAD_DIM), mask)

    assert torch.equal(cache.keys[0, 2, 0, :], key_states[0, 2, 0, :])


def test_update_value_to_correct_slot():
    """update() stores the value at the correct (batch, head, slot) position."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    val_states = torch.zeros(B, L, T, HEAD_DIM)
    val_states[0, 1, 0, :] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 1, 0] = True

    cache.update(torch.zeros(B, L, T, HEAD_DIM), val_states, mask)

    assert torch.equal(cache.values[0, 1, 0, :], val_states[0, 1, 0, :])


def test_update_increments_counts():
    """update() increments counts for each (batch, head) pair that received a token."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 3, 0] = True

    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)

    lengths = cache.get_heads_lengths()[0]
    assert lengths[0].item() == 1
    assert lengths[3].item() == 1
    assert lengths[1].item() == 0
    assert lengths[2].item() == 0


def test_update_multiple_tokens_same_head_counts():
    """update() with T tokens all in the same head increments count by T."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 3
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, :] = True

    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)

    assert cache.get_heads_lengths()[0, 0].item() == 3


def test_update_causal_ordering_within_slot():
    """Tokens appear in the buffer in the order they occupy the T dimension.

    This is the central correctness invariant for the cache: BEA relies on causal
    order being preserved within each head's slot. Three positions each with a distinct
    recognisable key value are active for head 0; the test verifies they land at buffer
    positions 0, 1, 2 in T order.
    """
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 3
    key_states = torch.zeros(B, L, T, HEAD_DIM)
    key_states[0, 0, 0, :] = 1.0
    key_states[0, 0, 1, :] = 2.0
    key_states[0, 0, 2, :] = 3.0
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, :] = True

    cache.update(key_states, torch.zeros(B, L, T, HEAD_DIM), mask)

    assert torch.equal(cache.keys[0, 0, 0, :], key_states[0, 0, 0, :])
    assert torch.equal(cache.keys[0, 0, 1, :], key_states[0, 0, 1, :])
    assert torch.equal(cache.keys[0, 0, 2, :], key_states[0, 0, 2, :])


def test_update_accumulates_across_calls():
    """Second update() call writes tokens after the first call's data."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True

    key_first = torch.full((B, L, T, HEAD_DIM), 1.0)
    key_second = torch.full((B, L, T, HEAD_DIM), 2.0)

    cache.update(key_first, torch.zeros(B, L, T, HEAD_DIM), mask)
    cache.update(key_second, torch.zeros(B, L, T, HEAD_DIM), mask)

    assert torch.equal(cache.keys[0, 0, 0, :], key_first[0, 0, 0, :])
    assert torch.equal(cache.keys[0, 0, 1, :], key_second[0, 0, 0, :])
    assert cache.get_heads_lengths()[0, 0].item() == 2


def test_update_inactive_heads_untouched():
    """Heads with no active positions in mask are not written and remain zero."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 1, 0] = True

    cache.update(torch.ones(B, L, T, HEAD_DIM), torch.ones(B, L, T, HEAD_DIM), mask)

    for h in [0, 2, 3]:
        assert cache.keys[0, h].sum().item() == 0.0
        assert cache.get_heads_lengths()[0, h].item() == 0


def test_update_sparse_mask_skips_inactive_positions():
    """Positions where mask is False are not written; active positions after them are."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 3
    key_states = torch.zeros(B, L, T, HEAD_DIM)
    key_states[0, 0, 0, :] = 1.0
    key_states[0, 0, 2, :] = 2.0
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 2] = True  # position 1 is skipped

    cache.update(key_states, torch.zeros(B, L, T, HEAD_DIM), mask)

    # Position 0 lands at slot 0, position 2 lands at slot 1.
    assert torch.equal(cache.keys[0, 0, 0, :], key_states[0, 0, 0, :])
    assert torch.equal(cache.keys[0, 0, 1, :], key_states[0, 0, 2, :])
    assert cache.get_heads_lengths()[0, 0].item() == 2


def test_update_batch_items_accumulate_independently():
    """Different batch items route independently — no cross-contamination."""
    cache = make_cache(batch=2)
    B, L, T = 2, NUM_HEADS, 1
    key_states = torch.zeros(B, L, T, HEAD_DIM)
    key_states[0, 0, 0, :] = 1.0
    key_states[1, 1, 0, :] = 2.0
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 1, 0] = True

    cache.update(key_states, torch.zeros(B, L, T, HEAD_DIM), mask)

    assert cache.get_heads_lengths()[0, 0].item() == 1
    assert cache.get_heads_lengths()[0, 1].item() == 0
    assert torch.equal(cache.keys[0, 0, 0, :], key_states[0, 0, 0, :])

    assert cache.get_heads_lengths()[1, 1].item() == 1
    assert cache.get_heads_lengths()[1, 0].item() == 0
    assert torch.equal(cache.keys[1, 1, 0, :], key_states[1, 1, 0, :])


def test_update_expansion_triggers_on_overflow():
    """buffer_capacity doubles when any slot would overflow."""
    cache = make_cache(batch=1, initial_buffer_size=2)
    assert cache.buffer_capacity == 2

    B, L = 1, NUM_HEADS
    mask = torch.zeros(B, L, 2, dtype=torch.bool)
    mask[0, 0, :] = True  # fill head 0 to capacity

    cache.update(torch.ones(B, L, 2, HEAD_DIM), torch.ones(B, L, 2, HEAD_DIM), mask)
    assert cache.buffer_capacity == 2

    # One more token to the same head triggers expansion.
    mask2 = torch.zeros(B, L, 1, dtype=torch.bool)
    mask2[0, 0, 0] = True
    cache.update(torch.ones(B, L, 1, HEAD_DIM), torch.ones(B, L, 1, HEAD_DIM), mask2)
    assert cache.buffer_capacity == 4


def test_update_expansion_preserves_existing_data():
    """Existing tokens are intact after buffer expansion."""
    cache = make_cache(batch=1, initial_buffer_size=2)
    B, L = 1, NUM_HEADS

    key_first = torch.zeros(B, L, 2, HEAD_DIM)
    key_first[0, 0, 0, :] = 1.0
    key_first[0, 0, 1, :] = 2.0
    mask = torch.zeros(B, L, 2, dtype=torch.bool)
    mask[0, 0, :] = True

    cache.update(key_first, torch.zeros(B, L, 2, HEAD_DIM), mask)

    key_third = torch.full((B, L, 1, HEAD_DIM), 3.0)
    mask2 = torch.zeros(B, L, 1, dtype=torch.bool)
    mask2[0, 0, 0] = True
    cache.update(key_third, torch.zeros(B, L, 1, HEAD_DIM), mask2)

    assert torch.equal(cache.keys[0, 0, 0, :], key_first[0, 0, 0, :])
    assert torch.equal(cache.keys[0, 0, 1, :], key_first[0, 0, 1, :])
    assert torch.equal(cache.keys[0, 0, 2, :], key_third[0, 0, 0, :])
    assert cache.get_heads_lengths()[0, 0].item() == 3


# ---------------------------------------------------------------------------
# update() — return value
# ---------------------------------------------------------------------------

def test_update_returns_three_tensor_tuple():
    """update() returns a tuple of exactly three tensors."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    result = cache.update(torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask)
    assert len(result) == 3
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_update_returns_correct_shapes():
    """update() returns (keys, values, active_mask) with the correct shapes."""
    cache = make_cache()
    B, L, T = BATCH, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    keys_out, vals_out, active_mask = cache.update(
        torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask
    )
    assert keys_out.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert vals_out.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE, HEAD_DIM)
    assert active_mask.shape == (BATCH, NUM_HEADS, INITIAL_BUFFER_SIZE)


def test_update_returns_bool_active_mask():
    """update() returns active_mask as a bool tensor."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
    mask = torch.zeros(B, L, T, dtype=torch.bool)
    _, _, active_mask = cache.update(
        torch.zeros(B, L, T, HEAD_DIM), torch.zeros(B, L, T, HEAD_DIM), mask
    )
    assert active_mask.dtype == torch.bool


def test_update_active_mask_all_false_on_empty_write():
    """update() with all-False mask returns an all-False active_mask."""
    cache = make_cache(batch=1)
    B, L, T = 1, NUM_HEADS, 1
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
    mask[0, 0, 0] = True   # 2 tokens to head 0
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True   # 1 token to head 1

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
