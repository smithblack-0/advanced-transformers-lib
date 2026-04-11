"""Tests for SparseMoSRAHPositions.

Invariants verified: main-sequence passthrough behavior, semantic-sequence local
position construction, semantic cached offset behavior, contiguous per-expert
positions for bulk T > 1 updates, BEA-compatible output shape, explicit cache
occupancy usage in cached semantic mode, and config-selected behavior across the
two supported RoPE modes.
"""

import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.positions_converter import SparseMoSRAHPositions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_packed_positions() -> torch.Tensor:
    """Construct a small packed position tensor with nontrivial values."""
    return torch.tensor(
        [
            [
                [100, 101, 103, 0],
                [100, 102, 103, 0],
                [101, 102, 0, 0],
            ],
            [
                [200, 201, 202, 203],
                [200, 202, 203, 0],
                [201, 203, 0, 0],
            ],
        ],
        dtype=torch.long,
    )


def make_config(rope_mode: str) -> ShramConfig:
    """Construct a minimal SHRAM config for position-layer tests."""
    return ShramConfig(
        rope_mode=rope_mode,
    )


class SpyMoSRAHCache:
    """Minimal cache spy for semantic-sequence position tests."""

    def __init__(self, head_lengths: torch.Tensor) -> None:
        self.head_lengths = head_lengths
        self.get_heads_lengths_calls = 0

    def get_heads_lengths(self) -> torch.Tensor:
        self.get_heads_lengths_calls += 1
        return self.head_lengths

    def update(self, *args, **kwargs):
        raise AssertionError(
            "SparseMoSRAHPositions must not update the cache."
        )


# ---------------------------------------------------------------------------
# Main-sequence behavior
# ---------------------------------------------------------------------------

class TestMainSequenceBehavior:
    def test_main_sequence_returns_packed_positions_unchanged(self):
        """Main-sequence mode should forward packed positions J unchanged."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("main_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        torch.testing.assert_close(positions, packed_positions)

    def test_main_sequence_output_shape_is_bea_compatible(self):
        """Main-sequence output should preserve the packed (B, L, T) shape."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("main_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        assert positions.shape == packed_positions.shape


# ---------------------------------------------------------------------------
# Semantic-sequence behavior
# ---------------------------------------------------------------------------

class TestSemanticSequenceBehavior:
    def test_semantic_sequence_without_cache_returns_local_positions(self):
        """Semantic-sequence mode without cache should return 0, 1, 2, ... over T."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        expected = torch.arange(
            packed_positions.shape[-1],
            dtype=packed_positions.dtype,
            device=packed_positions.device,
        ).view(1, 1, -1).expand_as(packed_positions)

        torch.testing.assert_close(positions, expected)

    def test_semantic_sequence_with_cache_returns_occupancy_offset_positions(self):
        """Cached semantic-sequence mode should offset local positions by head occupancies."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))
        cache = SpyMoSRAHCache(
            head_lengths=torch.tensor(
                [
                    [5, 2, 0],
                    [1, 3, 4],
                ],
                dtype=torch.long,
            )
        )

        positions = layer(
            packed_positions=packed_positions,
            cache=cache,
        )

        local_positions = torch.arange(
            packed_positions.shape[-1],
            dtype=packed_positions.dtype,
            device=packed_positions.device,
        ).view(1, 1, -1).expand_as(packed_positions)
        expected = local_positions + cache.head_lengths.unsqueeze(-1)

        torch.testing.assert_close(positions, expected)

    def test_bulk_semantic_updates_with_t_greater_than_one_are_contiguous(self):
        """Semantic-sequence positions should be contiguous across the packed T dimension."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))
        cache = SpyMoSRAHCache(
            head_lengths=torch.tensor(
                [
                    [7, 1, 0],
                    [4, 9, 2],
                ],
                dtype=torch.long,
            )
        )

        positions = layer(
            packed_positions=packed_positions,
            cache=cache,
        )

        deltas = positions[..., 1:] - positions[..., :-1]
        expected_deltas = torch.ones_like(deltas)
        torch.testing.assert_close(deltas, expected_deltas)

    def test_semantic_sequence_reads_cache_occupancies_and_does_not_mutate_cache(self):
        """Cached semantic-sequence computation should consult get_heads_lengths() only."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))
        cache = SpyMoSRAHCache(
            head_lengths=torch.tensor(
                [
                    [2, 0, 3],
                    [1, 4, 5],
                ],
                dtype=torch.long,
            )
        )

        _ = layer(
            packed_positions=packed_positions,
            cache=cache,
        )

        assert cache.get_heads_lengths_calls == 1

    def test_semantic_sequence_output_shape_is_bea_compatible(self):
        """Semantic-sequence output should preserve the packed (B, L, T) shape."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        assert positions.shape == packed_positions.shape


# ---------------------------------------------------------------------------
# Config-selected behavior
# ---------------------------------------------------------------------------

class TestConfiguredBehavior:
    def test_configured_layer_selects_main_sequence_behavior(self):
        """Configured main-sequence mode should preserve packed original-token positions."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("main_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        torch.testing.assert_close(positions, packed_positions)

    def test_configured_layer_selects_semantic_sequence_behavior(self):
        """Configured semantic-sequence mode should synthesize packed local-slot positions."""
        packed_positions = make_packed_positions()
        layer = SparseMoSRAHPositions(make_config("semantic_sequence"))

        positions = layer(
            packed_positions=packed_positions,
            cache=None,
        )

        expected = torch.arange(
            packed_positions.shape[-1],
            dtype=packed_positions.dtype,
            device=packed_positions.device,
        ).view(1, 1, -1).expand_as(packed_positions)

        torch.testing.assert_close(positions, expected)
