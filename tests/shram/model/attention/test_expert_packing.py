"""Tests for expert packing and unpacking.

Invariants verified: paper-specified packing behavior, stable-sort-based causal
order preservation within expert buckets, failure of a deliberately unstable
alternative, active-mask cardinality and padding identification, left-justified
active packing, authoritative upstream original-token positions, restoration of
token-choice shape on unpacking, round-trip identity on active entries, and
padding inactivity through unpacking.
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

def make_example(
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
    if position_ids is None:
        position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

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
    return hidden_states, position_ids, selected_heads, num_experts


def make_batch_example(
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
    if position_ids is None:
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=torch.long,
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
    return hidden_states, position_ids, selected_heads, num_experts


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
        hidden_states, position_ids, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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

    def test_packing_preserves_advanced_upstream_positions_through_expert_rearrangement(self):
        """Advanced upstream positions should survive packing rather than collapse to local indices."""
        hidden_states, _, selected_heads, num_experts = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long),
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        expected_positions = torch.tensor(
            [[
                [100, 101, 103],
                [100, 102, 103],
                [101, 102, 0],
            ]],
            dtype=torch.long,
        )
        expected_hidden_states = torch.tensor(
            [[
                [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0]],
                [[10.0, 11.0], [30.0, 31.0], [40.0, 41.0]],
                [[20.0, 21.0], [30.0, 31.0], [0.0, 0.0]],
            ]]
        )

        torch.testing.assert_close(packed_hidden_states, expected_hidden_states)
        torch.testing.assert_close(packed_positions, expected_positions)

        # The blocker fix exists precisely so these positions are not regenerated
        # from the local chunk as 0, 1, 2, 3 during cached inference.
        assert not torch.equal(
            packed_positions[active_mask],
            torch.tensor([0, 1, 3, 0, 2, 3, 1, 2], dtype=torch.long),
        )

    def test_packed_positions_remain_aligned_with_packed_hidden_state_ordering(self):
        """Packed positions should stay aligned with the packed hidden-state ordering."""
        hidden_states, position_ids, selected_heads, num_experts = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        expected_hidden_by_position = {
            100: hidden_states[0, 0],
            101: hidden_states[0, 1],
            102: hidden_states[0, 2],
            103: hidden_states[0, 3],
        }

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][active_mask[0, expert_idx]].tolist()
            active_hidden_states = packed_hidden_states[0, expert_idx][active_mask[0, expert_idx]]
            for position, hidden_state in zip(active_positions, active_hidden_states):
                torch.testing.assert_close(
                    hidden_state,
                    expected_hidden_by_position[position],
                )

    def test_main_sequence_positions_no_longer_depend_on_locally_regenerated_chunk_positions(self):
        """Changing only upstream positions should change packed positions but not packed hidden states."""
        hidden_states, _, selected_heads, num_experts = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)

        packed_hidden_states_a, packed_positions_a, active_mask_a = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )
        packed_hidden_states_b, packed_positions_b, active_mask_b = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        torch.testing.assert_close(packed_hidden_states_a, packed_hidden_states_b)
        torch.testing.assert_close(active_mask_a, active_mask_b)
        assert not torch.equal(packed_positions_a, packed_positions_b)


# ---------------------------------------------------------------------------
# Stable-sort causal behavior
# ---------------------------------------------------------------------------

class TestStableSortBehavior:
    def test_stable_sort_preserves_causal_order_within_each_expert_bucket(self):
        """Stable expert-major packing must preserve original token order within each expert."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][active_mask[0, expert_idx]]
            # Strictly increasing: each token has a unique position and routing assigns
            # each token to K distinct experts, so no token appears twice in one bucket.
            assert torch.all(active_positions[1:] > active_positions[:-1])

    def test_deliberately_unstable_alternative_breaks_causal_order(self):
        """A non-stable expert-major ordering should fail the causal-order invariant."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()

        flattened_selected_heads, _, _ = setup_packing(selected_heads)
        unstable_permutation = make_unstable_permutation(flattened_selected_heads)

        _, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert active_mask.sum().item() == batch_size * sequence_length * num_selected_heads

    def test_active_mask_behavior_is_unchanged_by_position_sourcing_fix(self):
        """Changing only upstream positions should not change the active-mask layout."""
        hidden_states, _, selected_heads, num_experts = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, active_mask_a = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )
        _, _, active_mask_b = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        torch.testing.assert_close(active_mask_a, active_mask_b)

    def test_active_mask_correctly_distinguishes_active_tokens_from_padding(self):
        """Active mask should mark real packed tokens and exclude padding slots."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        assert torch.all(packed_hidden_states[~active_mask] == 0)
        assert torch.all(packed_positions[~active_mask] == 0)

    def test_active_tokens_are_left_justified(self):
        """Within each expert slot, active entries should appear before all padding."""
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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

    def test_unpacking_behavior_is_unchanged_by_position_sourcing_fix(self):
        """Changing only upstream positions should not change unpacking / inverse mapping behavior."""
        hidden_states, _, selected_heads, num_experts = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)

        packed_hidden_states_a, _, active_mask_a = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )
        packed_hidden_states_b, _, active_mask_b = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        restored_a = unpack_experts(
            expert_outputs=packed_hidden_states_a,
            selected_heads=selected_heads,
            active_mask=active_mask_a,
            inverse_permutation=inverse_permutation,
        )
        restored_b = unpack_experts(
            expert_outputs=packed_hidden_states_b,
            selected_heads=selected_heads,
            active_mask=active_mask_b,
            inverse_permutation=inverse_permutation,
        )

        torch.testing.assert_close(restored_a, restored_b)

    def test_padding_regions_remain_inactive_through_unpacking(self):
        """Padding data should not become active outputs during unpacking."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
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
