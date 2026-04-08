"""Tests for MoSRAHCache — Units 6.A.A and 6.A.B.

Invariants verified in this file:

Unit 6.A.A — storage and HF Cache protocol:
- MoSRAHCache subclasses transformers.cache_utils.Cache
- get_seq_length() raises NotImplementedError
- get_expert_lengths() returns the (B, L) count tensor for any layer
- get_expert_lengths() returns zeros for layers with no updates yet
- is_initialized is False when all counts are zero, True after any update
- reset() zeroes all counts and buffers across all layers
- reorder_cache() permutes dim 0 of both buffers and counts atomically across all layers

Unit 6.A.B — vectorized scatter update:
- update() produces identical (keys, counts) to SlowMoSRAHCache on all test inputs.
  SlowMoSRAHCache is the correctness oracle: it was independently verified in
  test_slow_mosrah_cache.py before being used here. Agreement with it on all cases
  licenses trust in the vectorized implementation.
  Cases covered: single token; multi-token same head; multi-call accumulation;
  sparse routing; uneven per-head counts across batch items; buffer expansion.
"""

import torch
import pytest
from transformers.cache_utils import Cache

from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.cache.slow_mosrah_cache import SlowMoSRAHCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 3
NUM_HEADS = 4
HEAD_DIM = 16
BATCH = 2
INITIAL_T_MAX = 8


def make_cache(
    num_layers: int = NUM_LAYERS,
    batch: int = BATCH,
    initial_t_max: int = INITIAL_T_MAX,
) -> MoSRAHCache:
    return MoSRAHCache(
        num_hidden_layers=num_layers,
        num_mosrah_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        batch_size=batch,
        device=torch.device("cpu"),
        initial_t_max=initial_t_max,
    )


def make_slow_cache(
    num_layers: int = NUM_LAYERS,
    batch: int = BATCH,
    initial_t_max: int = INITIAL_T_MAX,
) -> SlowMoSRAHCache:
    return SlowMoSRAHCache(
        num_hidden_layers=num_layers,
        num_mosrah_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        batch_size=batch,
        device=torch.device("cpu"),
        initial_t_max=initial_t_max,
    )


def _run_both(
    head_idx: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    batch: int,
    initial_t_max: int = INITIAL_T_MAX,
) -> tuple[MoSRAHCache, SlowMoSRAHCache]:
    """Run the same update() call on both implementations and return both caches."""
    fast = make_cache(num_layers=1, batch=batch, initial_t_max=initial_t_max)
    slow = make_slow_cache(num_layers=1, batch=batch, initial_t_max=initial_t_max)
    fast.update(layer_idx=0, head_idx=head_idx, key_states=key_states, value_states=value_states)
    slow.update(layer_idx=0, head_idx=head_idx, key_states=key_states, value_states=value_states)
    return fast, slow


def _assert_agree(fast: MoSRAHCache, slow: SlowMoSRAHCache) -> None:
    """Assert that keys, values, and counts agree exactly across all layers."""
    for i in range(fast.num_hidden_layers):
        assert torch.equal(fast._keys[i], slow._keys[i]), f"keys differ at layer {i}"
        assert torch.equal(fast._values[i], slow._values[i]), f"values differ at layer {i}"
        assert torch.equal(fast._counts[i], slow._counts[i]), f"counts differ at layer {i}"


def seed_layer(cache: MoSRAHCache, layer_idx: int, counts: torch.Tensor) -> None:
    """Directly write known counts and matching key/value data into a layer's storage.

    Bypasses update() (which raises NotImplementedError in 6.A.A) to allow
    reset(), reorder_cache(), and get_expert_lengths() to be tested against
    known non-zero state.
    """
    cache._counts[layer_idx] = counts.clone()
    # Fill keys and values with recognisable per-batch-item data so reorder
    # tests can verify the correct rows were permuted.
    for b in range(cache.batch_size):
        cache._keys[layer_idx][b] = float(b + 1)
        cache._values[layer_idx][b] = float(-(b + 1))


# ---------------------------------------------------------------------------
# HF Cache protocol
# ---------------------------------------------------------------------------

def test_mosrah_cache_is_cache_subclass():
    """MoSRAHCache satisfies the HF Cache protocol."""
    assert issubclass(MoSRAHCache, Cache)


def test_get_seq_length_raises():
    """get_seq_length() raises NotImplementedError — no single length represents state."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_seq_length()


def test_get_seq_length_raises_with_layer_arg():
    """get_seq_length() raises regardless of the layer_idx argument."""
    cache = make_cache()
    with pytest.raises(NotImplementedError):
        cache.get_seq_length(layer_idx=2)



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

def test_buffers_allocated_at_construction():
    """All per-layer buffers exist immediately after construction."""
    cache = make_cache()
    assert len(cache._keys) == NUM_LAYERS
    assert len(cache._values) == NUM_LAYERS
    assert len(cache._counts) == NUM_LAYERS


def test_buffer_shapes_at_construction():
    """Key and value buffers have shape (B, L, T_max, u); counts have shape (B, L)."""
    cache = make_cache()
    for layer_idx in range(NUM_LAYERS):
        assert cache._keys[layer_idx].shape == (BATCH, NUM_HEADS, INITIAL_T_MAX, HEAD_DIM)
        assert cache._values[layer_idx].shape == (BATCH, NUM_HEADS, INITIAL_T_MAX, HEAD_DIM)
        assert cache._counts[layer_idx].shape == (BATCH, NUM_HEADS)


def test_counts_zero_at_construction():
    """All count tensors are zero immediately after construction."""
    cache = make_cache()
    for layer_idx in range(NUM_LAYERS):
        assert cache._counts[layer_idx].sum() == 0


# ---------------------------------------------------------------------------
# is_initialized
# ---------------------------------------------------------------------------

def test_is_initialized_false_on_fresh_cache():
    """is_initialized is False when all counts are zero."""
    cache = make_cache()
    assert cache.is_initialized is False


def test_is_initialized_true_after_update():
    """is_initialized is True after any update() call."""
    cache = make_cache(num_layers=1, batch=1)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=torch.ones(1, 1, 1, HEAD_DIM),
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    assert cache.is_initialized is True


# ---------------------------------------------------------------------------
# get_expert_lengths
# ---------------------------------------------------------------------------

def test_get_expert_lengths_shape_on_fresh_cache():
    """get_expert_lengths() returns shape (B, L) even for an unwritten layer."""
    cache = make_cache()
    lengths = cache.get_expert_lengths(0)
    assert lengths.shape == (BATCH, NUM_HEADS)


def test_get_expert_lengths_zeros_on_fresh_cache():
    """get_expert_lengths() returns all zeros for a layer that has no updates."""
    cache = make_cache()
    lengths = cache.get_expert_lengths(0)
    assert lengths.sum() == 0


def test_get_expert_lengths_returns_correct_values():
    """get_expert_lengths() returns the exact counts for a seeded layer."""
    cache = make_cache()
    expected = torch.tensor([[3, 1, 4, 2], [0, 5, 2, 3]])
    seed_layer(cache, layer_idx=0, counts=expected)
    assert torch.equal(cache.get_expert_lengths(0), expected)


def test_get_expert_lengths_independent_per_layer():
    """get_expert_lengths() returns the correct counts for each layer independently."""
    cache = make_cache()
    counts_0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    counts_2 = torch.tensor([[8, 7, 6, 5], [4, 3, 2, 1]])
    seed_layer(cache, layer_idx=0, counts=counts_0)
    seed_layer(cache, layer_idx=2, counts=counts_2)
    assert torch.equal(cache.get_expert_lengths(0), counts_0)
    assert torch.equal(cache.get_expert_lengths(2), counts_2)
    # Layer 1 was never written — should still be zero.
    assert cache.get_expert_lengths(1).sum() == 0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_zeroes_all_counts():
    """reset() sets all count tensors to zero across all layers."""
    cache = make_cache()
    seed_layer(cache, layer_idx=0, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    seed_layer(cache, layer_idx=2, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))

    cache.reset()

    for layer_idx in range(NUM_LAYERS):
        assert cache.get_expert_lengths(layer_idx).sum() == 0


def test_reset_zeroes_key_buffers():
    """reset() zeroes the key buffer data across all layers."""
    cache = make_cache()
    seed_layer(cache, layer_idx=1, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))

    cache.reset()

    for layer_idx in range(NUM_LAYERS):
        assert cache._keys[layer_idx].sum() == 0


def test_reset_is_initialized_false_afterward():
    """is_initialized returns False after reset."""
    cache = make_cache()
    seed_layer(cache, layer_idx=0, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()
    assert cache.is_initialized is False


def test_reset_on_fresh_cache_is_idempotent():
    """reset() on a fresh cache does not raise and leaves state unchanged."""
    cache = make_cache()
    cache.reset()
    assert cache.is_initialized is False


def test_reset_allows_reuse():
    """After reset(), counts can be re-seeded and read back correctly."""
    cache = make_cache()
    seed_layer(cache, layer_idx=0, counts=torch.ones(BATCH, NUM_HEADS, dtype=torch.long))
    cache.reset()

    new_counts = torch.tensor([[1, 0, 2, 0], [0, 3, 1, 2]])
    seed_layer(cache, layer_idx=0, counts=new_counts)
    assert torch.equal(cache.get_expert_lengths(0), new_counts)


# ---------------------------------------------------------------------------
# reorder_cache()
# ---------------------------------------------------------------------------

def test_reorder_cache_permutes_keys():
    """reorder_cache() permutes dim 0 of the key buffer for all layers."""
    cache = make_cache(num_layers=1, batch=3)
    seed_layer(cache, layer_idx=0, counts=torch.zeros(3, NUM_HEADS, dtype=torch.long))
    original_keys = cache._keys[0].clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.allclose(cache._keys[0][0], original_keys[2])
    assert torch.allclose(cache._keys[0][1], original_keys[0])
    assert torch.allclose(cache._keys[0][2], original_keys[1])


def test_reorder_cache_permutes_values():
    """reorder_cache() permutes dim 0 of the value buffer for all layers."""
    cache = make_cache(num_layers=1, batch=3)
    seed_layer(cache, layer_idx=0, counts=torch.zeros(3, NUM_HEADS, dtype=torch.long))
    original_values = cache._values[0].clone()

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.allclose(cache._values[0][0], original_values[2])
    assert torch.allclose(cache._values[0][1], original_values[0])
    assert torch.allclose(cache._values[0][2], original_values[1])


def test_reorder_cache_permutes_counts():
    """reorder_cache() permutes dim 0 of the count tensor for all layers."""
    cache = make_cache(num_layers=1, batch=3)
    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    seed_layer(cache, layer_idx=0, counts=counts)

    beam_idx = torch.tensor([2, 0, 1])
    cache.reorder_cache(beam_idx)

    assert torch.equal(cache._counts[0][0], counts[2])
    assert torch.equal(cache._counts[0][1], counts[0])
    assert torch.equal(cache._counts[0][2], counts[1])


def test_reorder_cache_applies_to_all_layers():
    """reorder_cache() reorders every allocated layer, not just the first."""
    cache = make_cache(batch=2)
    for layer_idx in range(NUM_LAYERS):
        seed_layer(
            cache,
            layer_idx=layer_idx,
            counts=torch.zeros(2, NUM_HEADS, dtype=torch.long),
        )
    original_keys = [cache._keys[i].clone() for i in range(NUM_LAYERS)]

    beam_idx = torch.tensor([1, 0])
    cache.reorder_cache(beam_idx)

    for layer_idx in range(NUM_LAYERS):
        assert torch.allclose(cache._keys[layer_idx][0], original_keys[layer_idx][1])
        assert torch.allclose(cache._keys[layer_idx][1], original_keys[layer_idx][0])


# ---------------------------------------------------------------------------
# update() — oracle comparison against SlowMoSRAHCache
#
# Each test runs identical inputs through both MoSRAHCache and SlowMoSRAHCache and
# asserts that the resulting key buffers and count tensors are exactly equal.
# SlowMoSRAHCache was independently verified in test_slow_mosrah_cache.py; agreement
# with it licenses trust in the vectorized implementation.
# ---------------------------------------------------------------------------

def test_oracle_single_token_single_head():
    """Single token routed to one head: vectorized matches oracle."""
    fast, slow = _run_both(
        head_idx=torch.tensor([[[2]]]),
        key_states=torch.randn(1, 1, 1, HEAD_DIM),
        value_states=torch.randn(1, 1, 1, HEAD_DIM),
        batch=1,
    )
    _assert_agree(fast, slow)


def test_oracle_single_token_multiple_heads():
    """Single token routed to K=2 heads: vectorized matches oracle."""
    fast, slow = _run_both(
        head_idx=torch.tensor([[[0, 3]]]),
        key_states=torch.randn(1, 1, 2, HEAD_DIM),
        value_states=torch.randn(1, 1, 2, HEAD_DIM),
        batch=1,
    )
    _assert_agree(fast, slow)


def test_oracle_multiple_tokens_same_head():
    """Multiple tokens all routed to the same head: causal order preserved."""
    fast, slow = _run_both(
        head_idx=torch.tensor([[[0], [0], [0], [0]]]),
        key_states=torch.randn(1, 4, 1, HEAD_DIM),
        value_states=torch.randn(1, 4, 1, HEAD_DIM),
        batch=1,
    )
    _assert_agree(fast, slow)


def test_oracle_multiple_tokens_different_heads():
    """Multiple tokens routed to different heads: each slot correct."""
    fast, slow = _run_both(
        head_idx=torch.tensor([[[0, 1], [2, 3], [1, 0]]]),
        key_states=torch.randn(1, 3, 2, HEAD_DIM),
        value_states=torch.randn(1, 3, 2, HEAD_DIM),
        batch=1,
    )
    _assert_agree(fast, slow)


def test_oracle_multi_call_accumulation():
    """Two sequential update() calls accumulate correctly."""
    fast = make_cache(num_layers=1, batch=1)
    slow = make_slow_cache(num_layers=1, batch=1)
    for _ in range(2):
        h = torch.tensor([[[0, 1]]])
        k = torch.randn(1, 1, 2, HEAD_DIM)
        v = torch.randn(1, 1, 2, HEAD_DIM)
        fast.update(layer_idx=0, head_idx=h, key_states=k, value_states=v)
        slow.update(layer_idx=0, head_idx=h, key_states=k, value_states=v)
    _assert_agree(fast, slow)


def test_oracle_uneven_batch():
    """Different batch items route to different heads: independence preserved."""
    fast, slow = _run_both(
        head_idx=torch.tensor([[[0, 1]], [[2, 3]]]),  # (2, 1, 2)
        key_states=torch.randn(2, 1, 2, HEAD_DIM),
        value_states=torch.randn(2, 1, 2, HEAD_DIM),
        batch=2,
    )
    _assert_agree(fast, slow)


def test_oracle_buffer_expansion():
    """Vectorized expansion produces the same result as the oracle expansion."""
    fast = make_cache(num_layers=1, batch=1, initial_t_max=2)
    slow = make_slow_cache(num_layers=1, batch=1, initial_t_max=2)
    # Two calls — first fills capacity, second triggers expansion.
    for _ in range(2):
        h = torch.tensor([[[0], [0]]])
        k = torch.randn(1, 2, 1, HEAD_DIM)
        v = torch.randn(1, 2, 1, HEAD_DIM)
        fast.update(layer_idx=0, head_idx=h, key_states=k, value_states=v)
        slow.update(layer_idx=0, head_idx=h, key_states=k, value_states=v)
    assert fast._t_max[0] == 4
    _assert_agree(fast, slow)
