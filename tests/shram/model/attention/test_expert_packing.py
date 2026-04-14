"""Tests for expert packing and unpacking.

Invariants verified: paper-specified packing behavior, stable-sort-based causal
order preservation within expert buckets, failure of a deliberately unstable
alternative, unpacking-mask cardinality and padding identification, left-justified
packing, authoritative upstream original-token positions, restoration of
token-choice shape on unpacking, round-trip identity on active entries, padding
inactivity through unpacking, active_mask correctly excludes dead outer tokens,
and the unpacking_mask/active_mask distinction is maintained under masked
continuation.
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


def assert_left_justified(unpacking_mask: torch.Tensor) -> None:
    """Assert each (B, expert) row of the unpacking mask is all True then all False.

    This invariant holds for unpacking_mask because expert buckets are left-justified
    by construction. It does NOT hold for active_mask in general: a dead token may
    fall between live tokens in the packed order, producing a non-contiguous pattern.
    """
    _, _, max_tokens = unpacking_mask.shape
    for row in unpacking_mask.reshape(-1, max_tokens).tolist():
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
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, unpacking_mask, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        expected_positions = torch.tensor(
            [[
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 0],
            ]],
            dtype=torch.long,
        )
        expected_unpacking_mask = torch.tensor(
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
        assert unpacking_mask.shape == expected_unpacking_mask.shape

        torch.testing.assert_close(packed_hidden_states, expected_hidden_states)
        torch.testing.assert_close(packed_positions, expected_positions)
        torch.testing.assert_close(unpacking_mask, expected_unpacking_mask)

    def test_packing_preserves_advanced_upstream_positions_through_expert_rearrangement(self):
        """Advanced upstream positions should survive packing rather than collapse to local indices."""
        hidden_states, _, selected_heads, num_experts = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long),
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
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
            packed_positions[unpacking_mask],
            torch.tensor([0, 1, 3, 0, 2, 3, 1, 2], dtype=torch.long),
        )

    def test_packed_positions_remain_aligned_with_packed_hidden_state_ordering(self):
        """Packed positions should stay aligned with the packed hidden-state ordering."""
        hidden_states, position_ids, selected_heads, num_experts = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        expected_hidden_by_position = {
            100: hidden_states[0, 0],
            101: hidden_states[0, 1],
            102: hidden_states[0, 2],
            103: hidden_states[0, 3],
        }

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][unpacking_mask[0, expert_idx]].tolist()
            active_hidden_states = packed_hidden_states[0, expert_idx][unpacking_mask[0, expert_idx]]
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
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)

        packed_hidden_states_a, packed_positions_a, unpacking_mask_a, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )
        packed_hidden_states_b, packed_positions_b, unpacking_mask_b, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        torch.testing.assert_close(packed_hidden_states_a, packed_hidden_states_b)
        torch.testing.assert_close(unpacking_mask_a, unpacking_mask_b)
        assert not torch.equal(packed_positions_a, packed_positions_b)


# ---------------------------------------------------------------------------
# Stable-sort causal behavior
# ---------------------------------------------------------------------------

class TestStableSortBehavior:
    def test_stable_sort_preserves_causal_order_within_each_expert_bucket(self):
        """Stable expert-major packing must preserve original token order within each expert."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, packed_positions, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][unpacking_mask[0, expert_idx]]
            # Strictly increasing: each token has a unique position and routing assigns
            # each token to K distinct experts, so no token appears twice in one bucket.
            assert torch.all(active_positions[1:] > active_positions[:-1])

    def test_deliberately_unstable_alternative_breaks_causal_order(self):
        """A non-stable expert-major ordering should fail the causal-order invariant."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, _, _ = setup_packing(selected_heads)
        unstable_permutation = make_unstable_permutation(flattened_selected_heads)

        _, packed_positions, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=unstable_permutation,
            outer_active_mask=outer_active_mask,
        )

        # At least one expert bucket should now be out of original token order.
        found_causal_violation = False
        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][unpacking_mask[0, expert_idx]]
            if torch.any(active_positions[1:] < active_positions[:-1]):
                found_causal_violation = True
                break

        assert found_causal_violation


# ---------------------------------------------------------------------------
# Active mask and left-justified packing
# ---------------------------------------------------------------------------

class TestActiveMask:
    def test_unpacking_mask_has_correct_cardinality(self):
        """The unpacking mask must contain exactly B * N * K true entries."""
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert unpacking_mask.sum().item() == batch_size * sequence_length * num_selected_heads

    def test_active_mask_behavior_is_unchanged_by_position_sourcing_fix(self):
        """Changing only upstream positions should not change the unpacking-mask layout."""
        hidden_states, _, selected_heads, num_experts = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask_a, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )
        _, _, unpacking_mask_b, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        torch.testing.assert_close(unpacking_mask_a, unpacking_mask_b)

    def test_active_mask_correctly_distinguishes_active_tokens_from_padding(self):
        """Unpacking mask should mark real packed tokens and exclude padding slots."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        packed_hidden_states, packed_positions, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        assert torch.all(packed_hidden_states[~unpacking_mask] == 0)
        assert torch.all(packed_positions[~unpacking_mask] == 0)

    def test_active_tokens_are_left_justified(self):
        """Within each expert slot, occupied entries should appear before all padding."""
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        assert_left_justified(unpacking_mask)

    def test_dead_outer_token_produces_dead_active_mask_slots(self):
        """A dead outer token must produce False active_mask entries at its packed slots.

        Uses the make_example routing (token 1 → experts 0 and 2) with token 1 marked
        dead. The expected active_mask is hand-computed from the routing:

          expert 0: tokens [0(live), 1(dead), 3(live)] → [True, False, True]
          expert 1: tokens [0(live), 2(live), 3(live)] → [True, True, True]
          expert 2: tokens [1(dead), 2(live), pad]     → [False, True, False]

        The unpacking_mask at those same slots must remain True — dead tokens still
        occupy slots and must be tracked by unpack_experts.
        """
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        expected_unpacking_mask = torch.tensor([[
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]])
        expected_active_mask = torch.tensor([[
            [True, False, True],
            [True, True, True],
            [False, True, False],
        ]])

        torch.testing.assert_close(unpacking_mask, expected_unpacking_mask)
        torch.testing.assert_close(active_mask, expected_active_mask)

    def test_unpacking_mask_cardinality_unchanged_when_outer_tokens_are_dead(self):
        """Dead outer tokens still occupy packed slots: unpacking_mask.sum() == B*N*K."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert unpacking_mask.sum().item() == batch_size * sequence_length * num_selected_heads

    def test_active_mask_cardinality_matches_live_token_count(self):
        """active_mask.sum() must equal the number of live outer tokens times K."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        # 3 live tokens, K=2: active_mask should have 6 True entries.
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, _, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        _, _, num_selected_heads = selected_heads.shape
        num_live_tokens = outer_active_mask.sum().item()
        assert active_mask.sum().item() == num_live_tokens * num_selected_heads

    def test_all_live_outer_mask_makes_active_mask_equal_unpacking_mask(self):
        """When all tokens are live, active_mask must equal unpacking_mask exactly."""
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, _ = setup_packing(selected_heads)
        _, _, unpacking_mask, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        torch.testing.assert_close(active_mask, unpacking_mask)


# ---------------------------------------------------------------------------
# Unpacking / round-trip behavior
# ---------------------------------------------------------------------------

class TestUnpacking:
    def test_unpack_restores_token_choice_shape(self):
        """Unpacking should return a tensor of shape (B, N, K, d)."""
        hidden_states, position_ids, selected_heads, num_experts = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        restored = unpack_experts(
            expert_outputs=packed_hidden_states,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask,
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
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )
        restored = unpack_experts(
            expert_outputs=packed_hidden_states,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask,
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
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)

        packed_hidden_states_a, _, unpacking_mask_a, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=sequential_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )
        packed_hidden_states_b, _, unpacking_mask_b, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=advanced_position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        restored_a = unpack_experts(
            expert_outputs=packed_hidden_states_a,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask_a,
            inverse_permutation=inverse_permutation,
        )
        restored_b = unpack_experts(
            expert_outputs=packed_hidden_states_b,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask_b,
            inverse_permutation=inverse_permutation,
        )

        torch.testing.assert_close(restored_a, restored_b)

    def test_padding_regions_remain_inactive_through_unpacking(self):
        """Padding data should not become active outputs during unpacking."""
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        expert_outputs = packed_hidden_states.clone()
        expert_outputs[~unpacking_mask] = torch.randn_like(expert_outputs[~unpacking_mask])

        restored = unpack_experts(
            expert_outputs=expert_outputs,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask,
            inverse_permutation=inverse_permutation,
        )

        expected = hidden_states.unsqueeze(2).expand(
            -1,
            -1,
            selected_heads.shape[-1],
            -1,
        )
        torch.testing.assert_close(restored, expected)

    def test_unpack_uses_unpacking_mask_not_active_mask(self):
        """Unpacking must use unpacking_mask so dead-token copies are correctly un-scattered.

        With a dead outer token, active_mask has fewer True entries than B*N*K. Passing
        active_mask to unpack_experts would break the reshape invariant. This test
        verifies the round-trip succeeds when the outer active mask has a dead token,
        confirming that unpacking_mask is the correct mask for unpacking.
        """
        hidden_states, position_ids, selected_heads, num_experts = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(selected_heads)
        packed_hidden_states, _, unpacking_mask, _ = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
            outer_active_mask=outer_active_mask,
        )

        restored = unpack_experts(
            expert_outputs=packed_hidden_states,
            selected_heads=selected_heads,
            unpacking_mask=unpacking_mask,
            inverse_permutation=inverse_permutation,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert restored.shape == (batch_size, sequence_length, num_selected_heads, hidden_states.shape[-1])
