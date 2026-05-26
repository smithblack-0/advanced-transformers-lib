"""Expert packing and unpacking for the MoSRAH path.

This module implements the low-level token-choice -> expert-choice -> token-choice
conversion boundary specified in the paper. The externally visible behavior is fixed:

- setup_packing() prepares the auxiliary ordering data and returns it as a dict
  payload forwarded whole to pack_experts and unpack_experts.
- pack_experts() converts a dict of routed token-choice tensors into packed
  expert-choice form. Each entry is paired with its intended padding value; all
  entries undergo the same expert-major gather-scatter so they remain aligned.
- unpack_experts() restores token-choice ordering afterward.

Stable sort is a correctness requirement. It preserves causal ordering inside each
expert bucket, which is the foundation on which BEA's later triangular causal mask
is correct.

pack_experts() returns the packed entries dict together with a separate unpacking_mask.
Two masks serve different roles and must not be interchanged:

- unpacking_mask: marks every packed slot that contains a routed token copy,
  live or dead. Always has exactly B*N*K True entries. Required by unpack_experts
  so its reshape invariant holds regardless of outer token liveness.
- active_mask (caller-supplied entry): marks only the packed slots whose source
  token was semantically live. This is what BEA consumes for attention gating.
  Dead outer tokens must not influence sparse attention outputs.
"""

import torch
from typing import Any


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_packing(
    selected_heads: torch.Tensor,
) -> dict[str, torch.Tensor]:
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
        Auxiliary payload dict with keys:
          - "flattened_selected_heads": H of shape (B, N*K)
          - "permutation": stable expert-major permutation Pi of shape (B, N*K)
          - "inverse_permutation": inverse permutation Pi^{-1} of shape (B, N*K)
        This dict is forwarded whole to pack_experts and unpack_experts.
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    flattened_selected_heads = selected_heads.reshape(
        batch_size,
        sequence_length * num_selected_heads,
    )
    num_elements = batch_size*sequence_length*num_selected_heads
    permutation = torch.argsort(flattened_selected_heads, dim=-1, stable=True)
    inverse_permutation = torch.argsort(permutation, dim=-1)

    return {
        "flattened_selected_heads": flattened_selected_heads,
        "permutation": permutation,
        "inverse_permutation": inverse_permutation,
        "num_elements" : num_elements,
    }


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_experts(
    entries: dict[str, tuple[torch.Tensor, Any]],
    setup: dict[str, torch.Tensor],
    selected_heads: torch.Tensor,
    num_experts: int,
    packed_length: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Pack token-choice tensors into expert-choice padded form.

    The paper's packing path has two jobs:

    1. Convert routed token-choice copies into expert-major order.
    2. Materialize that expert-major order into a padded tensor layout BEA can consume.

    All entries in the provided dict undergo the same expert-major gather-scatter so
    they remain mutually aligned in the packed frame. Each entry is paired with its
    intended padding value, which fills slots that contain no routed token copy.

    Packed positions are sourced from the authoritative upstream position_ids tensor
    rather than synthesized locally from arange(N). This preserves advanced positions
    correctly during cached inference while leaving training/full-sequence behavior
    unchanged when position_ids is the ordinary sequential token positions.

    Args:
        entries: Mapping from string keys to (tensor, padding_value) pairs. Each
            tensor has shape (B, N, ...) and is rearranged into expert-choice layout
            (B, L, T, ...). The returned dict carries the same keys.
        setup: Auxiliary payload returned by setup_packing().
        selected_heads: Routed head selections I of shape (B, N, K).
        num_experts: Total number of experts L.
        packed_length: Static packed time dimension T. All per-expert buffers are
            allocated to exactly this length. Use config.mosrah_packed_length as the
            source of this value. Raises if any actual per-expert token count exceeds
            this value.

    Returns:
        Tuple of:
          - packed_entries: Dict with same keys as entries; each value is the
            packed tensor of shape (B, L, T, ...).
          - unpacking_mask: Boolean tensor of shape (B, L, T). True where a slot
            contains any routed token copy, live or dead. Always has exactly
            B*N*K True entries. Pass this to unpack_experts — not active_mask.
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape

    flattened_selected_heads = setup["flattened_selected_heads"]
    permutation = setup["permutation"]

    # -----------------------------------------------------------------------
    # Reconstruct routed local source-token indices in token-choice order.
    #
    # The internal arange(N) is only the local source-row index object used to
    # gather from the current chunk tensors. Flattening gives a (B, N*K) tensor
    # aligned with H's token-major routed-copy order.
    # -----------------------------------------------------------------------
    source_token_indices = torch.arange(
        sequence_length,
        device=flattened_selected_heads.device,
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
    # order required by the paper. All entries are then gathered using these same
    # reordered indices so they remain aligned under the exact same transformation.
    # -----------------------------------------------------------------------
    sorted_source_indices = flattened_source_indices.gather(
        dim=1,
        index=permutation,
    )

    # -----------------------------------------------------------------------
    # Count how many routed copies land in each expert bucket and verify
    # that no bucket exceeds the statically preallocated packed_length T.
    #
    # S[b, l] is the number of routed token copies assigned to expert l in
    # batch b. T (packed_length) is a static allocation derived from config,
    # not a data-dependent maximum. Overflow is detected here and raises in
    # both eager and compiled modes.
    # -----------------------------------------------------------------------
    tokens_per_expert = _count_tokens_per_expert(flattened_selected_heads, num_experts)
    max_count = tokens_per_expert.max().item()
    no_overflow = max_count <= packed_length
    _enforce_no_overflow(no_overflow)

    # -----------------------------------------------------------------------
    # Construct the unpacking mask.
    #
    # Each expert bucket is left-justified: if S[b, l] = s, then slots
    # t = 0, ..., s-1 are occupied and all later slots are padding. The mask
    # marks slot occupancy regardless of outer token liveness, and always has
    # exactly B*N*K True entries.
    # -----------------------------------------------------------------------
    time_axis = torch.arange(
        packed_length,
        device=flattened_selected_heads.device,
        dtype=torch.long,
    ).view(1, 1, packed_length)
    unpacking_mask = time_axis < tokens_per_expert.unsqueeze(-1)

    # -----------------------------------------------------------------------
    # Materialize all entries into the packed expert-choice frame.
    #
    # Each entry is gathered using the expert-major sorted source indices, then
    # scattered into a padded buffer. The gather index is expanded to cover each
    # tensor's trailing dimensions. Padding slots receive the caller-supplied fill
    # value rather than an implicit zero.
    # -----------------------------------------------------------------------
    packed_entries: dict[str, torch.Tensor] = {}
    for key, (tensor, padding_value) in entries.items():
        extra_shape = tensor.shape[2:]

        # Expand gather index to cover trailing dimensions, if any.
        idx = sorted_source_indices.view(
            batch_size,
            sequence_length * num_selected_heads,
            *(1,) * len(extra_shape),
        ).expand(-1, -1, *extra_shape)
        sorted_tensor = tensor.gather(dim=1, index=idx)

        packed_tensor = tensor.new_full(
            (batch_size, num_experts, packed_length, *extra_shape),
            fill_value=padding_value,
        )

        packed_tensor[unpacking_mask] = sorted_tensor.reshape(-1, *extra_shape)
        packed_entries[key] = packed_tensor

    return packed_entries, unpacking_mask


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------

def unpack_experts(
    expert_outputs: torch.Tensor,
    setup: dict[str, torch.Tensor],
    unpacking_mask: torch.Tensor,
    selected_heads: torch.Tensor,
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
        setup: Auxiliary payload returned by setup_packing().
        unpacking_mask: From pack_experts(), shape (B, L, T). Identifies all
            occupied packed slots regardless of outer token liveness.
        selected_heads: Routed head selections I of shape (B, N, K).

    Returns:
        Restored token-choice tensor y_tilde of shape (B, N, K, d).
    """
    inverse_permutation = setup["inverse_permutation"]

    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    hidden_dim = expert_outputs.shape[-1]

    coords = torch.nonzero_static(
        unpacking_mask,
        size=setup["num_elements"],
    )  # shape: (B*N*K, 3)

    active_outputs = expert_outputs[
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
    ]  # shape: (B*N*K, d)

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

def _enforce_no_overflow(condition: bool) -> None:
    """Enforce that no expert bucket exceeds the preallocated packed length.

    This check fires when the number of tokens assigned to any expert in any
    batch item exceeds mosrah_packed_length. When that limit is exceeded, the
    packed buffer is too small to hold all assignments and data would be dropped.
    Increase mosrah_overallocation_factor in ShramConfig to resolve.

    The caller must derive condition via .item() on the max count tensor so that
    dynamo captures a SymInt and the comparison produces a SymBool. Passing a
    tensor comparison result directly bypasses the SymInt mechanism and prevents
    the check from firing at compiled runtime.

    Args:
        condition: True means no overflow has occurred; False means at least one
            expert bucket exceeds packed_length. In compiled mode this is a SymBool
            produced by comparing a SymInt against the static packed_length.
    """
    if torch.compiler.is_compiling():
        torch._check(condition)
    else:
        if not condition:
            raise RuntimeError(
                "Expert packing overflow: at least one expert bucket contains more "
                "tokens than mosrah_packed_length allows. Increase "
                "mosrah_overallocation_factor in ShramConfig to resolve."
            )


def _count_tokens_per_expert(
    flattened_selected_heads: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count how many routed token copies are assigned to each expert per batch item.

    Uses scatter_add into a pre-sized (B, num_experts) zero buffer, producing a
    statically-shaped output that compiles without graph breaks. Each position in
    flattened_selected_heads contributes one count to the corresponding expert slot.

    Args:
        flattened_selected_heads: Expert assignments of shape (B, N*K) with values
            in [0, num_experts).
        num_experts: Total number of experts L.

    Returns:
        Counts tensor of shape (B, num_experts).
    """
    batch_size = flattened_selected_heads.shape[0]
    counts = torch.zeros(
        batch_size,
        num_experts,
        device=flattened_selected_heads.device,
        dtype=flattened_selected_heads.dtype,
    )
    counts.scatter_add_(
        dim=1,
        index=flattened_selected_heads,
        src=torch.ones_like(flattened_selected_heads),
    )
    return counts
