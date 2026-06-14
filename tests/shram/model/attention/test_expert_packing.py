"""Tests for expert packing and unpacking.

Invariants verified: paper-specified packing behavior, stable-sort-based causal
order preservation within expert buckets, failure of a deliberately unstable
alternative, transfer-index fixed shape (B*N*K) independent of routing distribution,
left-justified packing, authoritative upstream original-token positions, restoration
of token-choice shape on unpacking, round-trip identity on active entries, padding
inactivity through unpacking, active_mask correctly excludes dead outer tokens,
and dead outer tokens still occupy B*N*K transfer slots.
"""

import pytest
import torch
import torch._dynamo

from src.shram.model.attention.expert_packing import (
    _count_tokens_per_expert,
    pack_experts,
    setup_packing,
    unpack_experts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_example(
    position_ids: torch.Tensor | None = None,
    packed_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
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
    if packed_length is None:
        packed_length = selected_heads.shape[1]
    return hidden_states, position_ids, selected_heads, num_experts, packed_length


def make_batch_example(
    position_ids: torch.Tensor | None = None,
    packed_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
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
    if packed_length is None:
        packed_length = selected_heads.shape[1]
    return hidden_states, position_ids, selected_heads, num_experts, packed_length


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
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        packed_hidden_states = packed["hidden_states"]
        packed_positions = packed["position_ids"]
        active_mask = packed["active_mask"]

        expected_positions = torch.tensor(
            [[
                [0, 1, 3, 0],
                [0, 2, 3, 0],
                [1, 2, 0, 0],
            ]],
            dtype=torch.long,
        )
        expected_hidden_states = torch.tensor(
            [[
                [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0], [0.0, 0.0]],
                [[10.0, 11.0], [30.0, 31.0], [40.0, 41.0], [0.0, 0.0]],
                [[20.0, 21.0], [30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]]
        )

        assert packed_hidden_states.shape == expected_hidden_states.shape
        assert packed_positions.shape == expected_positions.shape

        torch.testing.assert_close(packed_hidden_states, expected_hidden_states)
        torch.testing.assert_close(packed_positions, expected_positions)

    def test_packing_preserves_advanced_upstream_positions_through_expert_rearrangement(self):
        """Advanced upstream positions should survive packing rather than collapse to local indices."""
        hidden_states, _, selected_heads, num_experts, packed_length = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (torch.tensor([[100, 101, 102, 103]], dtype=torch.long), 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        packed_hidden_states = packed["hidden_states"]
        packed_positions = packed["position_ids"]

        expected_positions = torch.tensor(
            [[
                [100, 101, 103, 0],
                [100, 102, 103, 0],
                [101, 102, 0, 0],
            ]],
            dtype=torch.long,
        )
        expected_hidden_states = torch.tensor(
            [[
                [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0], [0.0, 0.0]],
                [[10.0, 11.0], [30.0, 31.0], [40.0, 41.0], [0.0, 0.0]],
                [[20.0, 21.0], [30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]]
        )

        torch.testing.assert_close(packed_hidden_states, expected_hidden_states)
        torch.testing.assert_close(packed_positions, expected_positions)

        # The blocker fix exists precisely so these positions are not regenerated
        # from the local chunk as 0, 1, 2, 3 during cached inference.
        assert not torch.equal(
            packed_positions[packed["active_mask"]],
            torch.tensor([0, 1, 3, 0, 2, 3, 1, 2], dtype=torch.long),
        )

    def test_packed_positions_remain_aligned_with_packed_hidden_state_ordering(self):
        """Packed positions should stay aligned with the packed hidden-state ordering."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example(
            position_ids=torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        )
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        packed_hidden_states = packed["hidden_states"]
        packed_positions = packed["position_ids"]
        active_mask = packed["active_mask"]

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
        hidden_states, _, selected_heads, num_experts, packed_length = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)

        packed_a, transfer_indices_a = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (sequential_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )
        packed_b, transfer_indices_b = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (advanced_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )
        packed_hidden_states_a = packed_a["hidden_states"]
        packed_positions_a = packed_a["position_ids"]
        packed_hidden_states_b = packed_b["hidden_states"]
        packed_positions_b = packed_b["position_ids"]

        torch.testing.assert_close(packed_hidden_states_a, packed_hidden_states_b)
        assert torch.equal(transfer_indices_a, transfer_indices_b)
        assert not torch.equal(packed_positions_a, packed_positions_b)


# ---------------------------------------------------------------------------
# Stable-sort causal behavior
# ---------------------------------------------------------------------------

class TestStableSortBehavior:
    def test_stable_sort_preserves_causal_order_within_each_expert_bucket(self):
        """Stable expert-major packing must preserve original token order within each expert."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        packed_positions = packed["position_ids"]
        active_mask = packed["active_mask"]

        for expert_idx in range(num_experts):
            active_positions = packed_positions[0, expert_idx][active_mask[0, expert_idx]]
            # Strictly increasing: each token has a unique position and routing assigns
            # each token to K distinct experts, so no token appears twice in one bucket.
            assert torch.all(active_positions[1:] > active_positions[:-1])

    def test_deliberately_unstable_alternative_breaks_causal_order(self):
        """A non-stable expert-major ordering should fail the causal-order invariant."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        unstable_permutation = make_unstable_permutation(setup["flattened_selected_heads"])
        unstable_setup = {**setup, "permutation": unstable_permutation}

        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, unstable_setup, selected_heads, num_experts, packed_length)
        packed_positions = packed["position_ids"]
        active_mask = packed["active_mask"]

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
    def test_unpacking_mask_has_correct_cardinality(self):
        """The transfer index must contain exactly B * N * K entries."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        _, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert transfer_indices.shape[0] == batch_size * sequence_length * num_selected_heads

    def test_active_mask_behavior_is_unchanged_by_position_sourcing_fix(self):
        """Changing only upstream positions should not change the unpacking-mask layout."""
        hidden_states, _, selected_heads, num_experts, packed_length = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        _, transfer_indices_a = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (sequential_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )
        _, transfer_indices_b = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (advanced_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )

        assert torch.equal(transfer_indices_a, transfer_indices_b)

    def test_active_mask_correctly_distinguishes_active_tokens_from_padding(self):
        """Unpacking mask should mark real packed tokens and exclude padding slots."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        packed_hidden_states = packed["hidden_states"]
        packed_positions = packed["position_ids"]
        active_mask = packed["active_mask"]

        assert torch.all(packed_hidden_states[~active_mask] == 0)
        assert torch.all(packed_positions[~active_mask] == 0)

    def test_active_tokens_are_left_justified(self):
        """Within each expert slot, occupied entries should appear before all padding."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        assert_left_justified(packed["active_mask"])

    def test_dead_outer_token_produces_dead_active_mask_slots(self):
        """A dead outer token must produce False active_mask entries at its packed slots.

        Uses the make_example routing (token 1 → experts 0 and 2) with token 1 marked
        dead. The expected active_mask is hand-computed from the routing:

          expert 0: tokens [0(live), 1(dead), 3(live), pad] → [True, False, True, False]
          expert 1: tokens [0(live), 2(live), 3(live), pad] → [True, True, True, False]
          expert 2: tokens [1(dead), 2(live), pad, pad]     → [False, True, False, False]

        Dead tokens still occupy transfer slots: transfer_indices.shape[0] == B*N*K
        regardless of outer token liveness.
        """
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        active_mask = packed["active_mask"]

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert transfer_indices.shape[0] == batch_size * sequence_length * num_selected_heads

        expected_active_mask = torch.tensor([[
            [True, False, True, False],
            [True, True, True, False],
            [False, True, False, False],
        ]])
        torch.testing.assert_close(active_mask, expected_active_mask)

    def test_unpacking_mask_cardinality_unchanged_when_outer_tokens_are_dead(self):
        """Dead outer tokens still occupy packed slots: transfer_indices.shape[0] == B*N*K."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        _, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert transfer_indices.shape[0] == batch_size * sequence_length * num_selected_heads

    def test_active_mask_cardinality_matches_live_token_count(self):
        """active_mask.sum() must equal the number of live outer tokens times K."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        # 3 live tokens, K=2: active_mask should have 6 True entries.
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, _ = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        active_mask = packed["active_mask"]

        _, _, num_selected_heads = selected_heads.shape
        num_live_tokens = outer_active_mask.sum().item()
        assert active_mask.sum().item() == num_live_tokens * num_selected_heads

    def test_all_live_outer_mask_makes_active_mask_equal_unpacking_mask(self):
        """When all tokens are live, active_mask must mark exactly B*N*K slots and be left-justified."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        active_mask = packed["active_mask"]

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert active_mask.sum().item() == batch_size * sequence_length * num_selected_heads
        assert_left_justified(active_mask)


# ---------------------------------------------------------------------------
# Unpacking / round-trip behavior
# ---------------------------------------------------------------------------

class TestUnpacking:
    def test_unpack_restores_token_choice_shape(self):
        """Unpacking should return a tensor of shape (B, N, K, d)."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        restored = unpack_experts(
            expert_outputs=packed["hidden_states"],
            setup=setup,
            flat_packed_transfer_indices=transfer_indices,
            selected_heads=selected_heads,
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
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_batch_example()
        outer_active_mask = torch.ones(2, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)
        restored = unpack_experts(
            expert_outputs=packed["hidden_states"],
            setup=setup,
            flat_packed_transfer_indices=transfer_indices,
            selected_heads=selected_heads,
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
        hidden_states, _, selected_heads, num_experts, packed_length = make_example()
        sequential_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        advanced_position_ids = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)

        packed_a, transfer_indices_a = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (sequential_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )
        packed_b, transfer_indices_b = pack_experts(
            {"hidden_states": (hidden_states, 0.0), "position_ids": (advanced_position_ids, 0), "active_mask": (outer_active_mask, False)},
            setup, selected_heads, num_experts, packed_length,
        )

        restored_a = unpack_experts(
            expert_outputs=packed_a["hidden_states"],
            setup=setup,
            flat_packed_transfer_indices=transfer_indices_a,
            selected_heads=selected_heads,
        )
        restored_b = unpack_experts(
            expert_outputs=packed_b["hidden_states"],
            setup=setup,
            flat_packed_transfer_indices=transfer_indices_b,
            selected_heads=selected_heads,
        )

        torch.testing.assert_close(restored_a, restored_b)

    def test_padding_regions_remain_inactive_through_unpacking(self):
        """Padding data should not become active outputs during unpacking."""
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        expert_outputs = packed["hidden_states"].clone()
        expert_outputs[~packed["active_mask"]] = torch.randn_like(expert_outputs[~packed["active_mask"]])

        restored = unpack_experts(
            expert_outputs=expert_outputs,
            setup=setup,
            flat_packed_transfer_indices=transfer_indices,
            selected_heads=selected_heads,
        )

        expected = hidden_states.unsqueeze(2).expand(
            -1,
            -1,
            selected_heads.shape[-1],
            -1,
        )
        torch.testing.assert_close(restored, expected)

    def test_unpack_uses_unpacking_mask_not_active_mask(self):
        """Unpacking must use the transfer index so dead-token copies are correctly un-scattered.

        With a dead outer token, active_mask has fewer True entries than B*N*K. Using
        active_mask to index into the packed frame would produce the wrong number of
        entries for the inverse permutation reshape. This test verifies the round-trip
        succeeds when the outer active mask has a dead token, confirming that the
        transfer index (not active_mask) is the correct unpacking artifact.
        """
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        packed, transfer_indices = pack_experts(entries, setup, selected_heads, num_experts, packed_length)

        restored = unpack_experts(
            expert_outputs=packed["hidden_states"],
            setup=setup,
            flat_packed_transfer_indices=transfer_indices,
            selected_heads=selected_heads,
        )

        batch_size, sequence_length, num_selected_heads = selected_heads.shape
        assert restored.shape == (batch_size, sequence_length, num_selected_heads, hidden_states.shape[-1])


# ---------------------------------------------------------------------------
# Count computation
# ---------------------------------------------------------------------------

class TestCountTokensPerExpert:
    def test_single_batch_counts_match_expected(self):
        """_count_tokens_per_expert must produce correct per-expert token counts.

        Routing: token 0 → [1,0], token 1 → [0,2], token 2 → [1,2], token 3 → [0,1]
        expert 0: tokens 0, 1, 3 → 3; expert 1: tokens 0, 2, 3 → 3; expert 2: tokens 1, 2 → 2
        """
        _, _, selected_heads, num_experts, _ = make_example()
        flattened = selected_heads.reshape(1, -1)

        counts = _count_tokens_per_expert(flattened, num_experts)

        expected = torch.tensor([[3, 3, 2]], dtype=flattened.dtype)
        torch.testing.assert_close(counts, expected)

    def test_batch_counts_match_expected(self):
        """_count_tokens_per_expert must produce correct counts for each batch item independently.

        Batch 0 routing same as single-batch example: expert counts [3, 3, 2].
        Batch 1: token 0 → [2,1], token 1 → [2,0], token 2 → [1,0], token 3 → [0,2]
                 expert 0: tokens 1, 2, 3 → 3; expert 1: tokens 0, 2 → 2; expert 2: tokens 0, 1, 3 → 3
        """
        _, _, selected_heads, num_experts, _ = make_batch_example()
        flattened = selected_heads.reshape(selected_heads.shape[0], -1)

        counts = _count_tokens_per_expert(flattened, num_experts)

        expected = torch.tensor([[3, 3, 2], [3, 2, 3]], dtype=flattened.dtype)
        torch.testing.assert_close(counts, expected)

    def test_output_shape_is_static(self):
        """Output shape must be (B, num_experts) regardless of routing distribution."""
        _, _, selected_heads, num_experts, _ = make_batch_example()
        flattened = selected_heads.reshape(selected_heads.shape[0], -1)
        batch_size = selected_heads.shape[0]

        counts = _count_tokens_per_expert(flattened, num_experts)

        assert counts.shape == (batch_size, num_experts)

    def test_counts_sum_to_n_times_k(self):
        """Total token assignments must equal N*K per batch item."""
        _, _, selected_heads, num_experts, _ = make_batch_example()
        flattened = selected_heads.reshape(selected_heads.shape[0], -1)
        _, sequence_length, num_selected_heads = selected_heads.shape

        counts = _count_tokens_per_expert(flattened, num_experts)

        expected_total = sequence_length * num_selected_heads
        for b in range(selected_heads.shape[0]):
            assert counts[b].sum().item() == expected_total


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------

class TestOverflowDetection:
    def test_overflow_raises_in_eager_mode(self):
        """pack_experts must raise when actual token count exceeds packed_length.

        The standard example routing places 3 tokens in experts 0 and 1, so
        packed_length=2 is insufficient and must trigger the overflow check.
        """
        hidden_states, position_ids, selected_heads, num_experts, _ = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }
        with pytest.raises(RuntimeError, match="mosrah_packed_length"):
            pack_experts(entries, setup, selected_heads, num_experts, packed_length=2)

    def test_overflow_raises_in_compiled_mode(self):
        """The overflow check must fire inside a compiled graph when overflow occurs."""
        hidden_states, position_ids, selected_heads, num_experts, _ = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }

        torch._dynamo.reset()
        compiled_pack = torch.compile(pack_experts)
        with pytest.raises(RuntimeError):
            compiled_pack(entries, setup, selected_heads, num_experts, packed_length=2)
    def test_exact_capacity_succeeds_in_eager_mode(self):
        """pack_experts must allow an expert bucket containing exactly packed_length entries.

        Capacity is a count of available packed slots, not a maximum valid index.
        A bucket with exactly C routed token copies occupies slots 0..C-1 and is
        therefore valid.
        """

        packed_length = 5
        num_experts = 2

        hidden_states = torch.arange(
            packed_length * 2,
            dtype=torch.float32,
        ).view(1, packed_length, 2)

        position_ids = torch.arange(
            packed_length,
            dtype=torch.long,
        ).view(1, packed_length)

        selected_heads = torch.zeros(
            1,
            packed_length,
            1,
            dtype=torch.long,
        )

        outer_active_mask = torch.ones(
            1,
            packed_length,
            dtype=torch.bool,
        )

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }

        packed, transfer_indices = pack_experts(
            entries,
            setup,
            selected_heads,
            num_experts,
            packed_length,
        )

        expected_expert_zero_mask = torch.ones(packed_length, dtype=torch.bool)
        expected_expert_one_mask = torch.zeros(packed_length, dtype=torch.bool)

        torch.testing.assert_close(
            packed["active_mask"][0, 0],
            expected_expert_zero_mask,
        )
        torch.testing.assert_close(
            packed["active_mask"][0, 1],
            expected_expert_one_mask,
        )
        torch.testing.assert_close(
            packed["hidden_states"][0, 0],
            hidden_states[0],
        )

    def test_one_over_capacity_raises_in_eager_mode(self):
        """pack_experts must reject an expert bucket containing packed_length + 1 entries."""

        packed_length = 5
        num_tokens = packed_length + 1
        num_experts = 2

        hidden_states = torch.arange(
            num_tokens * 2,
            dtype=torch.float32,
        ).view(1, num_tokens, 2)

        position_ids = torch.arange(
            num_tokens,
            dtype=torch.long,
        ).view(1, num_tokens)

        selected_heads = torch.zeros(
            1,
            num_tokens,
            1,
            dtype=torch.long,
        )

        outer_active_mask = torch.ones(
            1,
            num_tokens,
            dtype=torch.bool,
        )

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }

        with pytest.raises(RuntimeError, match="mosrah_packed_length"):
            pack_experts(
                entries,
                setup,
                selected_heads,
                num_experts,
                packed_length,
            )

# ---------------------------------------------------------------------------
# Compiled packing
# ---------------------------------------------------------------------------

class TestCompiledPacking:
    def test_pack_experts_compiles_without_graph_breaks(self):
        """pack_experts must compile and run without graph breaks under fullgraph=True.

        fullgraph=True raises immediately if dynamo inserts any graph break, so a
        successful run confirms the entire packing path is traceable end-to-end.
        """
        hidden_states, position_ids, selected_heads, num_experts, packed_length = make_example()
        outer_active_mask = torch.ones(1, 4, dtype=torch.bool)

        setup = setup_packing(selected_heads)
        entries = {
            "hidden_states": (hidden_states, 0.0),
            "position_ids": (position_ids, 0),
            "active_mask": (outer_active_mask, False),
        }

        torch._dynamo.reset()
        compiled_pack = torch.compile(pack_experts, fullgraph=True)
        compiled_pack(entries, setup, selected_heads, num_experts, packed_length)
