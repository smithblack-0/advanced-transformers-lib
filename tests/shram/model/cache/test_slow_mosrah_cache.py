"""Tests for SlowMoSRAHCache — Unit 6.A.B (oracle establishment).

SlowMoSRAHCache is the correctness oracle against which MoSRAHCache.update() is
validated. Its tests therefore carry extra weight: they must establish independent
trust in the oracle before it can license trust in the production implementation.
These tests verify SlowMoSRAHCache directly against known expected values — not
against another implementation — so that the trust chain has a solid foundation.

Invariants verified in this unit:
- SlowMoSRAHCache subclasses transformers.cache_utils.Cache
- get_seq_length() raises NotImplementedError
- get_expert_lengths() returns the (B, L) count tensor for any layer
- get_expert_lengths() returns zeros for layers with no updates yet
- is_initialized is False before any update, True after
- reset() zeroes all counts and buffers across all layers
- reorder_cache() permutes dim 0 of both buffers and counts atomically across all layers
- update() writes key/value to the correct (batch, head, slot) position
- update() increments counts correctly per (batch, head)
- update() preserves causal ordering: tokens appear in sequence order within each slot
- update() accumulates across calls: second call writes after first call's data
- update() is sparse: heads not selected in a step remain untouched
- update() is batch-independent: different batch items accumulate without cross-contamination
- update() triggers buffer expansion when any slot would overflow; existing data preserved
- update() returns the full (B, L, T_max, u) key and value buffers after the update
"""

import torch
import pytest
from transformers.cache_utils import Cache

from src.shram.model.cache.slow_mosrah_cache import SlowMoSRAHCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 3
NUM_HEADS = 4
HEAD_DIM = 4   # small so expected values are easy to read in assertions
BATCH = 2
INITIAL_T_MAX = 8


def make_cache(
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


def seed_layer(cache: SlowMoSRAHCache, layer_idx: int, counts: torch.Tensor) -> None:
    """Directly write known counts and matching key/value data into a layer's storage.

    Used for reset() and reorder_cache() tests that need known non-zero state without
    going through update().
    """
    cache._counts[layer_idx] = counts.clone()
    for b in range(cache.batch_size):
        cache._keys[layer_idx][b] = float(b + 1)
        cache._values[layer_idx][b] = float(-(b + 1))


# ---------------------------------------------------------------------------
# HF Cache protocol
# ---------------------------------------------------------------------------

def test_slow_mosrah_cache_is_cache_subclass():
    """SlowMoSRAHCache satisfies the HF Cache protocol."""
    assert issubclass(SlowMoSRAHCache, Cache)


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
    """After reset(), update() can write new data from position 0."""
    cache = make_cache(num_layers=1, batch=1)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=torch.full((1, 1, 1, HEAD_DIM), 9.0),
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    cache.reset()

    new_keys = torch.full((1, 1, 1, HEAD_DIM), 7.0)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=new_keys,
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    # After reset and re-update, token lands at position 0 with the new value.
    assert torch.equal(cache._keys[0][0, 0, 0, :], new_keys[0, 0, 0, :])
    assert cache._counts[0][0, 0].item() == 1


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
# update() — single step correctness
# ---------------------------------------------------------------------------

def test_update_writes_key_to_correct_slot():
    """update() writes key_states to the correct (batch, head, slot) position."""
    cache = make_cache(num_layers=1, batch=1)
    key = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[2]]]),  # head 2
        key_states=key,
        value_states=torch.zeros(1, 1, 1, HEAD_DIM),
    )
    assert torch.equal(cache._keys[0][0, 2, 0, :], key[0, 0, 0, :])


def test_update_writes_value_to_correct_slot():
    """update() writes value_states to the correct (batch, head, slot) position."""
    cache = make_cache(num_layers=1, batch=1)
    val = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[1]]]),  # head 1
        key_states=torch.zeros(1, 1, 1, HEAD_DIM),
        value_states=val,
    )
    assert torch.equal(cache._values[0][0, 1, 0, :], val[0, 0, 0, :])


def test_update_increments_counts():
    """update() increments counts for each (batch, head) pair that received a token."""
    cache = make_cache(num_layers=1, batch=1)
    # Select heads 0 and 3 (K=2).
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0, 3]]]),
        key_states=torch.ones(1, 1, 2, HEAD_DIM),
        value_states=torch.ones(1, 1, 2, HEAD_DIM),
    )
    counts = cache._counts[0][0]  # shape (L,)
    assert counts[0].item() == 1
    assert counts[3].item() == 1
    assert counts[1].item() == 0
    assert counts[2].item() == 0


def test_update_multiple_tokens_same_head_counts():
    """update() with N tokens all selecting the same head increments count by N."""
    cache = make_cache(num_layers=1, batch=1)
    # 3 tokens, K=1, all selecting head 0.
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0], [0], [0]]]),
        key_states=torch.ones(1, 3, 1, HEAD_DIM),
        value_states=torch.ones(1, 3, 1, HEAD_DIM),
    )
    assert cache._counts[0][0, 0].item() == 3


# ---------------------------------------------------------------------------
# update() — causal ordering
# ---------------------------------------------------------------------------

def test_update_causal_ordering_within_slot():
    """Tokens appear in the buffer in the order they occupied the sequence dimension.

    This is the central correctness invariant for the cache: BEA relies on causal
    order being preserved within each head's slot. Three tokens each with a distinct
    recognisable key value are routed to the same head; the test verifies they land
    at positions 0, 1, 2 in sequence order.
    """
    cache = make_cache(num_layers=1, batch=1)
    # 3 tokens, K=1, all selecting head 0. Keys are 1.0, 2.0, 3.0 so order is visible.
    key_states = torch.tensor([[
        [[1.0, 1.0, 1.0, 1.0]],  # token 0
        [[2.0, 2.0, 2.0, 2.0]],  # token 1
        [[3.0, 3.0, 3.0, 3.0]],  # token 2
    ]])  # (1, 3, 1, 4)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0], [0], [0]]]),
        key_states=key_states,
        value_states=torch.zeros(1, 3, 1, HEAD_DIM),
    )
    assert torch.equal(cache._keys[0][0, 0, 0, :], key_states[0, 0, 0, :])
    assert torch.equal(cache._keys[0][0, 0, 1, :], key_states[0, 1, 0, :])
    assert torch.equal(cache._keys[0][0, 0, 2, :], key_states[0, 2, 0, :])


# ---------------------------------------------------------------------------
# update() — multi-call accumulation
# ---------------------------------------------------------------------------

def test_update_accumulates_across_calls():
    """Second update() call writes tokens after the first call's data."""
    cache = make_cache(num_layers=1, batch=1)
    key_first = torch.full((1, 1, 1, HEAD_DIM), 1.0)
    key_second = torch.full((1, 1, 1, HEAD_DIM), 2.0)

    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=key_first,
        value_states=torch.zeros(1, 1, 1, HEAD_DIM),
    )
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=key_second,
        value_states=torch.zeros(1, 1, 1, HEAD_DIM),
    )

    assert torch.equal(cache._keys[0][0, 0, 0, :], key_first[0, 0, 0, :])
    assert torch.equal(cache._keys[0][0, 0, 1, :], key_second[0, 0, 0, :])
    assert cache._counts[0][0, 0].item() == 2


# ---------------------------------------------------------------------------
# update() — sparse routing
# ---------------------------------------------------------------------------

def test_update_sparse_heads_not_selected_are_untouched():
    """Heads not selected by the router in a given step remain unchanged."""
    cache = make_cache(num_layers=1, batch=1)
    # Select only head 1 — heads 0, 2, 3 must be untouched.
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[1]]]),
        key_states=torch.ones(1, 1, 1, HEAD_DIM),
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    for h in [0, 2, 3]:
        assert cache._keys[0][0, h].sum().item() == 0.0
        assert cache._counts[0][0, h].item() == 0


# ---------------------------------------------------------------------------
# update() — batch independence
# ---------------------------------------------------------------------------

def test_update_batch_items_accumulate_independently():
    """Different batch items route independently — no cross-contamination."""
    cache = make_cache(num_layers=1, batch=2)
    # Batch 0 selects head 0; batch 1 selects head 1. Different heads, different items.
    key_b0 = torch.full((1, HEAD_DIM), 1.0)
    key_b1 = torch.full((1, HEAD_DIM), 2.0)
    key_states = torch.stack([key_b0, key_b1]).unsqueeze(1).unsqueeze(1)  # (2,1,1,4)

    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]], [[1]]]),  # (2, 1, 1)
        key_states=key_states,
        value_states=torch.zeros(2, 1, 1, HEAD_DIM),
    )

    # Batch 0: head 0 has 1 token; head 1 is empty.
    assert cache._counts[0][0, 0].item() == 1
    assert cache._counts[0][0, 1].item() == 0
    assert torch.equal(cache._keys[0][0, 0, 0, :], key_b0[0])

    # Batch 1: head 1 has 1 token; head 0 is empty.
    assert cache._counts[0][1, 1].item() == 1
    assert cache._counts[0][1, 0].item() == 0
    assert torch.equal(cache._keys[0][1, 1, 0, :], key_b1[0])


# ---------------------------------------------------------------------------
# update() — buffer expansion
# ---------------------------------------------------------------------------

def test_update_expansion_triggers_on_overflow():
    """T_max doubles when any slot would overflow the current capacity."""
    cache = make_cache(num_layers=1, batch=1, initial_t_max=2)
    assert cache._t_max[0] == 2

    # Fill to capacity with 2 tokens.
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0], [0]]]),
        key_states=torch.ones(1, 2, 1, HEAD_DIM),
        value_states=torch.ones(1, 2, 1, HEAD_DIM),
    )
    assert cache._t_max[0] == 2

    # One more token — should trigger expansion.
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=torch.ones(1, 1, 1, HEAD_DIM),
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    assert cache._t_max[0] == 4


def test_update_expansion_preserves_existing_data():
    """Existing tokens are intact after buffer expansion."""
    cache = make_cache(num_layers=1, batch=1, initial_t_max=2)
    key_first = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]]])  # (1,2,1,4)

    # Fill to capacity.
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0], [0]]]),
        key_states=key_first,
        value_states=torch.zeros(1, 2, 1, HEAD_DIM),
    )

    # Trigger expansion with a third token.
    key_third = torch.full((1, 1, 1, HEAD_DIM), 3.0)
    cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=key_third,
        value_states=torch.zeros(1, 1, 1, HEAD_DIM),
    )

    assert torch.equal(cache._keys[0][0, 0, 0, :], key_first[0, 0, 0, :])
    assert torch.equal(cache._keys[0][0, 0, 1, :], key_first[0, 1, 0, :])
    assert torch.equal(cache._keys[0][0, 0, 2, :], key_third[0, 0, 0, :])
    assert cache._counts[0][0, 0].item() == 3


# ---------------------------------------------------------------------------
# update() — return value
# ---------------------------------------------------------------------------

def test_update_returns_key_and_value_buffers():
    """update() returns (keys, values) — the full (B, L, T_max, u) buffers."""
    cache = make_cache(num_layers=1, batch=1)
    keys, values = cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=torch.ones(1, 1, 1, HEAD_DIM),
        value_states=torch.ones(1, 1, 1, HEAD_DIM),
    )
    assert keys.shape == (1, NUM_HEADS, INITIAL_T_MAX, HEAD_DIM)
    assert values.shape == (1, NUM_HEADS, INITIAL_T_MAX, HEAD_DIM)


def test_update_returned_buffers_contain_written_data():
    """Returned key buffer contains the token just written."""
    cache = make_cache(num_layers=1, batch=1)
    key = torch.full((1, 1, 1, HEAD_DIM), 5.0)
    keys, _ = cache.update(
        layer_idx=0,
        head_idx=torch.tensor([[[0]]]),
        key_states=key,
        value_states=torch.zeros(1, 1, 1, HEAD_DIM),
    )
    assert torch.equal(keys[0, 0, 0, :], key[0, 0, 0, :])
