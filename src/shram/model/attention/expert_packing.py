"""Expert packing and unpacking for the MoSRAH path.

This module implements the low-level token-choice -> expert-choice -> token-choice
conversion boundary specified in the paper. The externally visible behavior is fixed:

- setup_packing() prepares the auxiliary ordering data.
- pack_experts() converts routed token-choice state into packed expert-choice state.
- unpack_experts() restores token-choice ordering afterward.

Stable sort is a correctness requirement. It preserves causal ordering inside each
expert bucket, which is the foundation on which BEA's later triangular causal mask
is correct.

pack_experts() returns two distinct masks that serve different roles and must not
be interchanged:

- unpacking_mask: marks every packed slot that contains a routed token copy,
  live or dead. Always has exactly B*N*K True entries. Required by unpack_experts
  so its reshape invariant holds regardless of outer token liveness.
- active_mask: marks only the packed slots whose source token was semantically
  live. This is what BEA consumes for attention gating. Dead outer tokens must
  not influence sparse attention outputs.
"""

import torch


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_packing(
    selected_heads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the auxiliary ordering data used by pack/unpack.

    Routing produces token-choice state I of shape (B, N, K): for each token, which
    K experts were selected. Packing needs the same routed token copies reordered into
    expert-major order so each expert bucket becomes contiguous.

    The paper's setup step does this by flattening (N, K) into one axis to produce
    H in token-major order, then computing a stable argsort permutation Pi over the
    expert indices stored in H. Applying Pi reorders the flattened routed copies into
    expert-major order while preserving their original token order *within* each expert
    bucket. That preservation is why stable sort is required for causality.

    Args:
        selected_heads: Routed token-choice head selections I of shape (B, N, K).

    Returns:
        Tuple of:
          - flattened_selected_heads: H of shape (B, N*K)
          - permutation: stable expert-major permutation Pi of shape (B, N*K)
          - inverse_permutation: inverse permutation Pi^{-1} of shape (B, N*K)
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    flattened_selected_heads = selected_heads.reshape(
        batch_size,
        sequence_length * num_selected_heads,
    )

    permutation = torch.argsort(flattened_selected_heads, dim=-1, stable=True)
    inverse_permutation = torch.argsort(permutation, dim=-1)

    return flattened_selected_heads, permutation, inverse_permutation


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_experts(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    selected_heads: torch.Tensor,
    num_experts: int,
    flattened_selected_heads: torch.Tensor,
    permutation: torch.Tensor,
    outer_active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack token-choice hidden states into expert-choice padded form.

    The paper's packing path has two jobs:

    1. Convert routed token-choice copies into expert-major order.
    2. Materialize that expert-major order into a padded tensor layout BEA can consume.

    The routed hidden-state copies are not stored explicitly in token-choice form.
    Instead, the same token hidden state is conceptually copied once per selected expert.
    The packing step reconstructs those copies by expanding local source-token indices,
    reordering those indices with Pi, then gathering hidden states, positions, and outer
    liveness in that packed order. All three are carried through the same expert-major
    rearrangement so they remain aligned in the packed frame.

    Packed positions are sourced from the authoritative upstream position_ids tensor
    rather than synthesized locally from arange(N). This preserves advanced positions
    correctly during cached inference while leaving training/full-sequence behavior
    unchanged when position_ids is the ordinary sequential token positions.

    Args:
        hidden_states: Token-choice hidden states x of shape (B, N, d).
        position_ids: Authoritative upstream token positions J of shape (B, N).
        selected_heads: Routed head selections I of shape (B, N, K).
        num_experts: Total number of experts L.
        flattened_selected_heads: H from setup_packing(), shape (B, N*K).
        permutation: Pi from setup_packing(), shape (B, N*K).
        outer_active_mask: Current-chunk active mask of shape (B, N), where True
            means the token is semantically live. Dead tokens do not become
            semantically active in the packed sparse representation.

    Returns:
        Tuple of:
          - packed_hidden_states: x' of shape (B, L, T, d)
          - packed_positions: J' of shape (B, L, T)
          - unpacking_mask: of shape (B, L, T). True where a slot contains any
            routed token copy, live or dead. Always has exactly B*N*K True entries.
            Pass this to unpack_experts — not active_mask.
          - active_mask: of shape (B, L, T). True only where a slot contains a
            copy of a live outer token. Pass this to BEA for attention gating.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    _, _, num_selected_heads = selected_heads.shape

    # -----------------------------------------------------------------------
    # Reconstruct routed local source-token indices in token-choice order.
    #
    # The internal arange(N) is no longer the packed position tensor. It is only
    # the local source-row index object used to gather from the current chunk
    # tensor x. Flattening this object gives a (B, N*K) tensor aligned with H's
    # token-major routed-copy order.
    # -----------------------------------------------------------------------
    source_token_indices = torch.arange(
        sequence_length,
        device=hidden_states.device,
        dtype=torch.long,
    ).view(1, sequence_length, 1).expand(
        batch_size,
        sequence_length,
        num_selected_heads,
    )
    flattened_source_indices = source_token_indices.reshape(
        batch_size,
        sequence_length * num_selected_heads,
    )

    # -----------------------------------------------------------------------
    # Reorder source-token indices into expert-major order.
    #
    # Applying Pi yields the local source-token rows in the packed expert-major
    # order required by the paper. Those same reordered source indices are then
    # used to gather hidden states, authoritative upstream positions, and outer
    # liveness so all three remain aligned under the exact same packing
    # transformation.
    # -----------------------------------------------------------------------
    sorted_source_indices = flattened_source_indices.gather(
        dim=1,
        index=permutation,
    )
    sorted_hidden_states = hidden_states.gather(
        dim=1,
        index=sorted_source_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )
    sorted_positions = position_ids.gather(
        dim=1,
        index=sorted_source_indices,
    )
    sorted_active_mask = outer_active_mask.gather(
        dim=1,
        index=sorted_source_indices,
    )

    # -----------------------------------------------------------------------
    # Count how many routed copies land in each expert bucket.
    #
    # S[b, l] is the number of routed token copies assigned to expert l in batch b.
    # T is the maximum such count across all batches and experts; it determines the
    # padded expert-length dimension of the packed representation.
    # -----------------------------------------------------------------------
    tokens_per_expert = _bincount_rows(
        values=flattened_selected_heads,
        num_bins=num_experts,
    )
    max_tokens_per_expert = int(tokens_per_expert.max().item())

    # -----------------------------------------------------------------------
    # Construct the active-token mask M.
    #
    # Each expert bucket is left-justified: if S[b, l] = s, then slots
    # t = 0, ..., s-1 are active and all later slots are padding. The resulting
    # mask therefore both identifies real packed tokens and enforces left-justified
    # packing. This is the unpacking_mask — it marks slot occupancy regardless of
    # outer token liveness, and always has exactly B*N*K True entries.
    # -----------------------------------------------------------------------
    time_axis = torch.arange(
        max_tokens_per_expert,
        device=hidden_states.device,
        dtype=torch.long,
    ).view(1, 1, max_tokens_per_expert)
    unpacking_mask = time_axis < tokens_per_expert.unsqueeze(-1)

    # -----------------------------------------------------------------------
    # Materialize the padded packed tensors.
    #
    # The packed hidden states x', packed original-token positions J', and packed
    # active-token mask are allocated as zero-filled tensors. Active entries are
    # then written into those buffers in the expert-major order established above.
    # Padding remains zero / inactive.
    # -----------------------------------------------------------------------
    packed_hidden_states = hidden_states.new_zeros(
        batch_size,
        num_experts,
        max_tokens_per_expert,
        hidden_dim,
    )
    packed_positions = position_ids.new_zeros(
        batch_size,
        num_experts,
        max_tokens_per_expert,
    )
    active_mask = torch.zeros(
        batch_size,
        num_experts,
        max_tokens_per_expert,
        dtype=torch.bool,
        device=hidden_states.device,
    )

    packed_hidden_states[unpacking_mask] = sorted_hidden_states.reshape(-1, hidden_dim)
    packed_positions[unpacking_mask] = sorted_positions.reshape(-1)
    active_mask[unpacking_mask] = sorted_active_mask.reshape(-1)

    return packed_hidden_states, packed_positions, unpacking_mask, active_mask


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------

def unpack_experts(
    expert_outputs: torch.Tensor,
    selected_heads: torch.Tensor,
    unpacking_mask: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    """Restore token-choice ordering from BEA expert-choice output.

    Unpacking inverts the packing path only on occupied entries. Padding does not
    participate: the output tensor is first filtered by unpacking_mask to recover
    only the real routed-token copies in expert-major order, then Pi^{-1} restores
    the original token-choice ordering, and finally the tensor is reshaped back to
    (B, N, K, d).

    The unpacking_mask — not active_mask — must be used here. Even copies of dead
    outer tokens occupy slots and must be un-scattered correctly for the inverse
    permutation to hold. The total True entry count in unpacking_mask is always
    B*N*K, which is exactly what the reshape to (B, N*K, d) requires.

    Args:
        expert_outputs: Expert-choice BEA output y of shape (B, L, T, d).
        selected_heads: Routed head selections I of shape (B, N, K).
        unpacking_mask: From pack_experts(), shape (B, L, T). Identifies all
            occupied packed slots regardless of outer token liveness.
        inverse_permutation: Pi^{-1} from setup_packing(), shape (B, N*K).

    Returns:
        Restored token-choice tensor y_tilde of shape (B, N, K, d).
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    hidden_dim = expert_outputs.shape[-1]

    active_outputs = expert_outputs[unpacking_mask]
    sorted_token_choice_outputs = active_outputs.reshape(
        batch_size,
        sequence_length * num_selected_heads,
        hidden_dim,
    )
    restored_outputs = sorted_token_choice_outputs.gather(
        dim=1,
        index=inverse_permutation.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )

    return restored_outputs.reshape(
        batch_size,
        sequence_length,
        num_selected_heads,
        hidden_dim,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bincount_rows(
    values: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """Count per-row integer occurrences for a 2D tensor.

    torch.bincount operates on a flat 1D vector, but the packing algorithm needs
    one bincount per batch row. The trick used here is to shift each row into its
    own disjoint bin range before flattening:

      row 0 uses bins [0, ..., num_bins - 1]
      row 1 uses bins [num_bins, ..., 2*num_bins - 1]
      row 2 uses bins [2*num_bins, ..., 3*num_bins - 1]
      ...

    After that shift, one global torch.bincount produces all row-local counts at
    once. Reshaping the result back to (B, num_bins) recovers the per-row bincount.

    This is a vectorized implementation detail only; externally visible behavior
    remains exactly the paper's S tensor of per-batch per-expert token counts.

    Args:
        values: Integer tensor of shape (B, M) with entries in [0, num_bins).
        num_bins: Number of bins.

    Returns:
        Counts tensor of shape (B, num_bins).
    """
    batch_size = values.shape[0]

    row_offsets = torch.arange(
        batch_size,
        device=values.device,
        dtype=values.dtype,
    ).unsqueeze(1) * num_bins
    shifted_values = values + row_offsets

    counts = torch.bincount(
        shifted_values.reshape(-1),
        minlength=batch_size * num_bins,
    )
    return counts.reshape(batch_size, num_bins)
