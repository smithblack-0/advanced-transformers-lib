# tests/shram/model/cache/test_sliding_window_cache.py

"""Tests for LocalSlidingWindowLayerCache — Unit 14.C."""

import pytest
import torch
from transformers.cache_utils import CacheLayerMixin

from src.shram.model.cache.sliding_window_cache import LocalSlidingWindowLayerCache


def k(rows: list[list[float]]) -> torch.Tensor:
    """Build `(B, H=1, T, D=1)` key tensors from simple row-wise data."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)


def v(rows: list[list[float]]) -> torch.Tensor:
    """Build `(B, H=1, T, D=1)` value tensors from simple row-wise data."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)


def m(rows: list[list[bool]]) -> torch.Tensor:
    """Build `(B, T)` active masks."""
    return torch.tensor(rows, dtype=torch.bool)


def make_cache(
    *,
    sliding_window: int,
    batch_size: int,
) -> LocalSlidingWindowLayerCache:
    return LocalSlidingWindowLayerCache(
        sliding_window=sliding_window,
        num_heads=1,
        head_dim=1,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Basic cache-boundary protocol
# ---------------------------------------------------------------------------


def test_local_sliding_window_layer_cache_is_cache_layer_mixin():
    assert issubclass(LocalSlidingWindowLayerCache, CacheLayerMixin)


def test_cache_is_initialized_at_construction():
    cache = make_cache(sliding_window=3, batch_size=1)
    assert cache.is_initialized is True


def test_get_max_cache_shape_returns_configured_window():
    cache = make_cache(sliding_window=7, batch_size=1)
    assert cache.get_max_cache_shape() == 7


def test_get_seq_length_is_zero_at_construction():
    cache = make_cache(sliding_window=3, batch_size=1)
    assert cache.get_seq_length() == 0


def test_get_seq_length_counts_first_chunk():
    cache = make_cache(sliding_window=3, batch_size=1)
    cache.update(k([[1.0, 2.0, 3.0]]), v([[1.0, 2.0, 3.0]]), m([[True, True, True]]))
    assert cache.get_seq_length() == 3


def test_get_seq_length_accumulates_across_updates():
    cache = make_cache(sliding_window=3, batch_size=1)
    cache.update(k([[1.0, 2.0]]), v([[1.0, 2.0]]), m([[True, True]]))
    cache.update(k([[3.0]]), v([[3.0]]), m([[True]]))
    assert cache.get_seq_length() == 3


def test_get_seq_length_counts_all_positions_not_just_active():
    # Dead positions still count toward processed sequence length.
    cache = make_cache(sliding_window=3, batch_size=1)
    cache.update(k([[1.0, 2.0]]), v([[1.0, 2.0]]), m([[False, True]]))
    assert cache.get_seq_length() == 2


def test_get_seq_length_resets_to_zero_after_reset():
    cache = make_cache(sliding_window=3, batch_size=1)
    cache.update(k([[1.0, 2.0]]), v([[1.0, 2.0]]), m([[True, True]]))
    cache.reset()
    assert cache.get_seq_length() == 0


def test_get_mask_sizes_raises_not_implemented():
    cache = make_cache(sliding_window=3, batch_size=1)
    with pytest.raises(NotImplementedError):
        cache.get_mask_sizes(torch.tensor([0]))


# ---------------------------------------------------------------------------
# update() contract
# ---------------------------------------------------------------------------


def test_first_update_returns_retained_buffer_concatenated_with_current_chunk():
    cache = make_cache(sliding_window=3, batch_size=1)

    out_k, out_v, out_m = cache.update(
        k([[10.0, 11.0]]),
        v([[110.0, 111.0]]),
        m([[True, False]]),
    )

    expected_k = k([[0.0, 0.0, 0.0, 10.0, 11.0]])
    expected_v = v([[0.0, 0.0, 0.0, 110.0, 111.0]])
    expected_m = m([[False, False, False, True, False]])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_second_update_observationally_shows_only_last_window_is_retained():
    cache = make_cache(sliding_window=3, batch_size=1)

    _ = cache.update(
        k([[10.0, 11.0]]),
        v([[110.0, 111.0]]),
        m([[True, False]]),
    )

    out_k, out_v, out_m = cache.update(
        k([[12.0]]),
        v([[112.0]]),
        m([[True]]),
    )

    # After the first update, the retained next-step cache is the last three raw
    # positions of [0, 0, 0, 10, 11], i.e. [0, 10, 11], with mask [F, T, F].
    # The second update should therefore return [0, 10, 11] + [12].
    expected_k = k([[0.0, 10.0, 11.0, 12.0]])
    expected_v = v([[0.0, 110.0, 111.0, 112.0]])
    expected_m = m([[False, True, False, True]])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_ragged_batch_masks_are_supported():
    cache = make_cache(sliding_window=3, batch_size=2)

    _ = cache.update(
        k([[10.0, 11.0], [20.0, 21.0]]),
        v([[110.0, 111.0], [120.0, 121.0]]),
        m([[False, True], [True, True]]),
    )

    out_k, out_v, out_m = cache.update(
        k([[12.0], [22.0]]),
        v([[112.0], [122.0]]),
        m([[True], [False]]),
    )

    expected_k = k([
        [0.0, 10.0, 11.0, 12.0],
        [0.0, 20.0, 21.0, 22.0],
    ])
    expected_v = v([
        [0.0, 110.0, 111.0, 112.0],
        [0.0, 120.0, 121.0, 122.0],
    ])
    expected_m = m([
        [False, False, True, True],
        [False, True, True, False],
    ])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_dead_positions_may_remain_present_but_are_marked_dead():
    cache = make_cache(sliding_window=3, batch_size=1)

    out_k, out_v, out_m = cache.update(
        k([[7.0, 8.0]]),
        v([[107.0, 108.0]]),
        m([[False, True]]),
    )

    expected_k = k([[0.0, 0.0, 0.0, 7.0, 8.0]])
    expected_v = v([[0.0, 0.0, 0.0, 107.0, 108.0]])
    expected_m = m([[False, False, False, False, True]])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_all_live_case_behaves_like_simple_positional_concat_and_trim():
    cache = make_cache(sliding_window=3, batch_size=1)

    _ = cache.update(
        k([[1.0, 2.0]]),
        v([[101.0, 102.0]]),
        m([[True, True]]),
    )

    out_k, out_v, out_m = cache.update(
        k([[3.0]]),
        v([[103.0]]),
        m([[True]]),
    )

    expected_k = k([[0.0, 1.0, 2.0, 3.0]])
    expected_v = v([[0.0, 101.0, 102.0, 103.0]])
    expected_m = m([[False, True, True, True]])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


# ---------------------------------------------------------------------------
# reset / reorder / repeat / select
# ---------------------------------------------------------------------------


def test_reset_restores_fresh_cache_behavior_observationally():
    cache = make_cache(sliding_window=3, batch_size=1)

    _ = cache.update(
        k([[10.0, 11.0]]),
        v([[110.0, 111.0]]),
        m([[True, True]]),
    )

    cache.reset()

    out_k, out_v, out_m = cache.update(
        k([[5.0]]),
        v([[105.0]]),
        m([[True]]),
    )

    expected_k = k([[0.0, 0.0, 0.0, 5.0]])
    expected_v = v([[0.0, 0.0, 0.0, 105.0]])
    expected_m = m([[False, False, False, True]])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_reorder_cache_preserves_alignment_observationally():
    cache = make_cache(sliding_window=2, batch_size=2)

    _ = cache.update(
        k([[10.0], [20.0]]),
        v([[110.0], [120.0]]),
        m([[True], [False]]),
    )

    cache.reorder_cache(torch.tensor([1, 0]))

    out_k, out_v, out_m = cache.update(
        k([[21.0], [11.0]]),
        v([[121.0], [111.0]]),
        m([[True], [True]]),
    )

    expected_k = k([
        [0.0, 20.0, 21.0],
        [0.0, 10.0, 11.0],
    ])
    expected_v = v([
        [0.0, 120.0, 121.0],
        [0.0, 110.0, 111.0],
    ])
    expected_m = m([
        [False, False, True],
        [False, True, True],
    ])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_batch_repeat_interleave_preserves_alignment_observationally():
    cache = make_cache(sliding_window=2, batch_size=2)

    _ = cache.update(
        k([[10.0], [20.0]]),
        v([[110.0], [120.0]]),
        m([[True], [False]]),
    )

    cache.batch_repeat_interleave(2)

    out_k, out_v, out_m = cache.update(
        k([[1.0], [2.0], [3.0], [4.0]]),
        v([[101.0], [102.0], [103.0], [104.0]]),
        m([[True], [True], [True], [True]]),
    )

    expected_k = k([
        [0.0, 10.0, 1.0],
        [0.0, 10.0, 2.0],
        [0.0, 20.0, 3.0],
        [0.0, 20.0, 4.0],
    ])
    expected_v = v([
        [0.0, 110.0, 101.0],
        [0.0, 110.0, 102.0],
        [0.0, 120.0, 103.0],
        [0.0, 120.0, 104.0],
    ])
    expected_m = m([
        [False, True, True],
        [False, True, True],
        [False, False, True],
        [False, False, True],
    ])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)


def test_batch_select_indices_preserves_alignment_observationally():
    cache = make_cache(sliding_window=2, batch_size=3)

    _ = cache.update(
        k([[10.0], [20.0], [30.0]]),
        v([[110.0], [120.0], [130.0]]),
        m([[True], [False], [True]]),
    )

    cache.batch_select_indices(torch.tensor([2, 0]))

    out_k, out_v, out_m = cache.update(
        k([[31.0], [11.0]]),
        v([[131.0], [111.0]]),
        m([[True], [True]]),
    )

    expected_k = k([
        [0.0, 30.0, 31.0],
        [0.0, 10.0, 11.0],
    ])
    expected_v = v([
        [0.0, 130.0, 131.0],
        [0.0, 110.0, 111.0],
    ])
    expected_m = m([
        [False, True, True],
        [False, True, True],
    ])

    assert torch.equal(out_k, expected_k)
    assert torch.equal(out_v, expected_v)
    assert torch.equal(out_m, expected_m)