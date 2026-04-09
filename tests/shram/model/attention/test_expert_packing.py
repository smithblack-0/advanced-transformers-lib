"""Tests for expert packing and unpacking.

Invariants verified: paper-specified packing behavior, stable-sort-based causal
order preservation within expert buckets, failure of a deliberately unstable
alternative, active-mask cardinality and padding identification, left-justified
active packing, packed original-token positions, restoration of token-choice
shape on unpacking, round-trip identity on active entries, and padding
inactivity through unpacking.
"""

import torch

from src.shram.model.attention.expert_packing import (
    pack_experts,
    setup_packing,
    unpack_experts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_example() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Construct a small hand-checkable packing example.

    selected_heads is arranged so that stable expert-major packing has a clear
    expected result:

      token 0 -> experts 1, 0
      token 1 -> experts 0, 2
      token 2 -> experts 1, 2
      token 3 -> experts 0, 1

    This yields packed expert buckets:

      expert 0: tokens [0, 1, 3]
      expert 1: tokens [0, 2, 3]
      expert 2: tokens [1, 2]
    """
    hidden_states = torch.tensor(
        [[
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ]]
    )
    selected_heads = torch.tensor(
        [[
            [1, 0],
            [0, 2],
            [1, 2],
            [0, 1],
        ]],
        dtype=torch.long,
    )
    num_experts = 3
    return hidden_states, selected_heads, num_experts


def make_batch_example() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Construct a batch-sized example for mask and round-trip checks."""
    hidden_states = torch.tensor(
        [
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
            ],
            [
                [50.0, 51.0],
                [60.0, 61.0],
                [70.0, 71.0],
                [80.0, 81.0],
            ],
        ]
    )
    selected_heads = torch.tensor(
        [
            [
                [1, 0],
                [0, 2],
                [1, 2],
                [0, 1],
            ],
            [
                [2, 1],
                [2, 0],
                [1, 0],
                [0, 2],
            ],
        ],
        dtype=torch.long,
    )
    num_experts = 3
    return hidden_states, selected_heads, num_experts


def make_unstable_permutation(flattened_selected_heads: torch.Tensor) -> torch.Tensor:
    """Construct a deliberately unstable expert-major permutation.

    This keeps expert-major grouping but reverses original order within each
    equal-expert group, which is enough to violate the causal-order invariant.
    """
    batch_size, num_routed_copies = flattened_selected_heads.shape
    reverse_indices = torch.arange(
        num_routed_copies - 1,
        -1,
        -1,
        device=flattened_selected_heads.device,
        dtype=flattened_selected_heads.dtype,
    ).view(1, num_routed_copies).expand(batch_size, -1)

    unstable_keys = flattened_selected_heads * num_routed_copies + reverse_indices
    return torch.argsort(unstable_keys, dim=-1, stable=True)


def assert_left_justified(active_mask: torch.Tensor) -> None:
    """Assert each (B, expert) row is all True then all False."""
    _, _, max_tokens = active_mask.shape
    for row in active_mask.reshape(-1, max_tokens).tolist():
        seen_inactive = False
        for entry in row:
            if not entry:
                seen_inactive = True
            else:
                assert not seen_inactive, "Found an active slot after padding."


# ---------------------------------------------------------------------------
# Paper-algorithm behavior
# ---------------------------------------------------------------------------

class TestPaperAlgorithmBehavior:
    def test_pack_matches_hand_computed_example(self):
        """Packing should match the paper-specified expert-major layout on a known example."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        expected_positions = torch.tensor(
            [[
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 0],
            ]],
            dtype=torch.long,
        )
        expected_active_mask = torch.tensor(
            [[
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]]
        )
        expected_hidden_states = torch.tensor(
            [[
                [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0]],
                [[10.0, 11.0], [30.0, 31.0], [40.0, 41.0]],
                [[20.0, 21.0], [30.0, 31.0], [0.0, 0.0]],
            ]]
        )

        assert packed_hidden_states.shape == expected_hidden_states.shape
        assert packed_positions.shape == expected_positions.shape
        assert active_mask.shape == expected_active_mask.shape

        torch.testing.assert_close(packed_hidden_states, expected_hidden_states)
        torch.testing.assert_close(packed_positions, expected_positions)
        torch.testing.assert_close(active_mask, expected_active_mask)

    def test_packed_positions_match_original_sequence_positions_in_packed_order(self):
        """Packed positions J should record original token positions in expert-major order."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        expected_active_positions = [
            [0, 1, 3],  # expert 0
            [0, 2, 3],  # expert 1
            [1, 2],     # expert 2
        ]

        for expert_idx, expected in enumerate(expected_active_positions):
            actual = packed_positions[0, expert_idx][active_mask[0, expert_idx]].tolist()
            assert actual == expected


# ---------------------------------------------------------------------------
# Stable-sort causal behavior
# ---------------------------------------------------------------------------

class TestStableSortBehavior:
    def test_stable_sort_preserves_causal_order_within_each_expert_bucket(self):
        """Stable expert-major packing must preserve original token order within each expert."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][active_mask[0, expert_idx]]
            assert torch.all(active_positions[1:] >= active_positions[:-1])

    def test_deliberately_unstable_alternative_breaks_causal_order(self):
        """A non-stable expert-major ordering should fail the causal-order invariant."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, _, _ = setup_packing(selected_heads)
        unstable_permutation = make_unstable_permutation(flattened_selected_heads)

        _, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=unstable_permutation,
        )

        # At least one expert bucket should now be out of original token order.
        found_causal_violation = False
        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][active_mask[0, expert_idx]]
            if torch.any(active_positions[1:] < active_positions[:-1]):
                found_causal_violation = True
                break

        assert found_causal_violation


# ---------------------------------------------------------------------------
# Active mask and left-justified packing
# ---------------------------------------------------------------------------

class TestActiveMask:
    def test_active_mask_has_correct_cardinality(self):
        """The active mask should contain exactly B * N * K true entries."""
        hidden_states, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert active_mask.sum().item() == batch_size * sequence_length * num_selected_heads

    def test_active_mask_correctly_distinguishes_active_tokens_from_padding(self):
        """Active mask should mark real packed tokens and exclude padding slots."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        assert torch.all(packed_hidden_states[~active_mask] == 0)
        assert torch.all(packed_positions[~active_mask] == 0)

    def test_active_tokens_are_left_justified(self):
        """Within each expert slot, active entries should appear before all padding."""
        hidden_states, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        assert_left_justified(active_mask)


# ---------------------------------------------------------------------------
# Unpacking / round-trip behavior
# ---------------------------------------------------------------------------

class TestUnpacking:
    def test_unpack_restores_token_choice_shape(self):
        """Unpacking should return a tensor of shape (B, N, K, d)."""
        hidden_states, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        restored = unpack_experts(
            expert_outputs=packed_hidden_states,
            selected_heads=selected_heads,
            active_mask=active_mask,
            inverse_permutation=inverse_permutation,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert restored.shape == (
            batch_size,
            sequence_length,
            num_selected_heads,
            hidden_states.shape[-1],
        )

    def test_round_trip_identity_on_active_entries(self):
        """unpack(pack(x, I)) should recover the original active token copies."""
        hidden_states, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )
        restored = unpack_experts(
            expert_outputs=packed_hidden_states,
            selected_heads=selected_heads,
            active_mask=active_mask,
            inverse_permutation=inverse_permutation,
        )

        expected = hidden_states.unsqueeze(2).expand(
            -1,
            -1,
            selected_heads.shape[-1],
            -1,
        )
        torch.testing.assert_close(restored, expected)

    def test_padding_regions_remain_inactive_through_unpacking(self):
        """Padding data should not become active outputs during unpacking."""
        hidden_states, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        expert_outputs = packed_hidden_states.clone()
        expert_outputs[~active_mask] = torch.randn_like(expert_outputs[~active_mask])

        restored = unpack_experts(
            expert_outputs=expert_outputs,
            selected_heads=selected_heads,
            active_mask=active_mask,
            inverse_permutation=inverse_permutation,
        )

        expected = hidden_states.unsqueeze(2).expand(
            -1,
            -1,
            selected_heads.shape[-1],
            -1,
        )
        torch.testing.assert_close(restored, expected)