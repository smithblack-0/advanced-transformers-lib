"""Expert packing and unpacking for the MoSRAH path.

This module owns the token-choice -> expert-choice -> token-choice conversion
boundary used by the sparse routed attention path. Its public behavior is fixed:

- setup_packing() prepares the auxiliary ordering data forwarded through packing
  and unpacking.
- pack_experts() converts routed token-choice tensors into padded expert-choice
  tensors.
- unpack_experts() restores token-choice ordering from padded expert-choice output.

Packed expert-choice tensors are expert-major and left-justified. For each expert,
routed token copies occupy the prefix of that expert's packed block; padding occupies
the suffix. Every packed entry uses the same ordering and transfer artifact, so
hidden states, positions, masks, and probabilities remain aligned across the boundary.

pack_experts() returns a flat transfer index together with the packed entries. This
index replaces the old boolean unpacking artifact as the source of truth for
pack/unpack data movement: packing writes to those flat packed slots, and unpacking
reads from those same slots.
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

    Args:
        selected_heads: Routed token-choice head selections I of shape (B, N, K).

    Returns:
        Auxiliary payload dict with keys:
          - "flattened_selected_heads": H of shape (B, N*K)
          - "permutation": expert-major permutation Pi of shape (B, N*K)
          - "inverse_permutation": inverse permutation Pi^{-1} of shape (B, N*K)
        This dict is forwarded whole to pack_experts and unpack_experts.
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    flattened_selected_heads = selected_heads.reshape(
        batch_size,
        sequence_length * num_selected_heads,
    )

    # -----------------------------------------------------------------------
    # Establish the expert-major ordering invariant.
    #
    # BEA later applies a triangular causal mask inside each expert bucket. That
    # mask is only meaningful if routed copies for the same expert preserve their
    # source-token order. Stable sorting by selected head establishes that order.
    # -----------------------------------------------------------------------
    permutation = torch.argsort(flattened_selected_heads, dim=-1, stable=True)
    inverse_permutation = torch.argsort(permutation, dim=-1)

    return {
        "flattened_selected_heads": flattened_selected_heads,
        "permutation": permutation,
        "inverse_permutation": inverse_permutation,
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

    Args:
        entries: Mapping from string keys to (tensor, padding_value) pairs. Each
            tensor has shape (B, N, ...) and is rearranged into expert-choice layout
            (B, L, T, ...). The returned dict carries the same keys.
        setup: Auxiliary payload returned by setup_packing().
        selected_heads: Routed head selections I of shape (B, N, K).
        num_experts: Total number of experts L.
        packed_length: Static packed time dimension T. All per-expert buffers are
            allocated to exactly this length. Raises if any actual per-expert token
            count exceeds this value.

    Returns:
        Tuple of:
          - packed_entries: Dict with same keys as entries; each value is the
            packed tensor of shape (B, L, T, ...).
          - flat_packed_transfer_indices: Long tensor of shape (B*N*K,). Each value
            is the flattened padded expert-choice slot occupied by the corresponding
            routed-copy row. Pass this to unpack_experts().
    """
    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    num_routed_copies_per_batch = sequence_length * num_selected_heads
    num_routed_copies = batch_size * num_routed_copies_per_batch

    flattened_selected_heads = setup["flattened_selected_heads"]
    permutation = setup["permutation"]

    # -----------------------------------------------------------------------
    # Algorithm overview.
    #
    # Packing first builds one routed-copy row for each selected token/expert
    # pair, ordered by the stable expert-major permutation. Those rows contain
    # no padding. The final packed tensor reserves packed_length slots per expert.
    # The flat transfer index bridges those layouts by adding back the cumulative
    # padding skipped before each expert block.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Build the shared routed-copy source rows.
    #
    # This tensor identifies the source token row for each selected token/expert
    # pair after the stable expert-major permutation. Every packed entry uses this
    # same row plan, so all entries remain aligned before padded materialization.
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
    flattened_source_token_indices = source_token_indices.reshape(
        batch_size,
        num_routed_copies_per_batch,
    )
    sorted_source_token_indices = flattened_source_token_indices.gather(
        dim=1,
        index=permutation,
    )

    # -----------------------------------------------------------------------
    # Establish packed expert occupancy and capacity.
    #
    # tokens_per_expert tells how many routed-copy rows occupy the prefix of each
    # expert block. The padded layout is valid only when every prefix fits inside
    # the configured packed_length.
    # -----------------------------------------------------------------------
    tokens_per_expert = _count_tokens_per_expert(flattened_selected_heads, num_experts)
    _enforce_no_overflow(tokens_per_expert, packed_length)

    # -----------------------------------------------------------------------
    # Build the flat insertion points for the padded expert frame.
    #
    # Routed-copy rows omit padding, while the packed frame reserves packed_length
    # slots for every expert. The transfer index adds back the cumulative padding
    # skipped before each expert block, producing one flat destination slot for
    # every routed-copy row. This tensor is forwarded to unpack_experts so removal
    # uses the same positions that insertion used.
    # -----------------------------------------------------------------------
    flat_tokens_per_expert = tokens_per_expert.reshape(-1)
    flat_padding_per_expert = packed_length - flat_tokens_per_expert
    flat_padding_before_expert = (
        flat_padding_per_expert.cumsum(dim=0) - flat_padding_per_expert
    )

    flat_padding_for_routed_rows = torch.repeat_interleave(
        flat_padding_before_expert,
        flat_tokens_per_expert,
        output_size=num_routed_copies,
    )
    flat_routed_row_indices = torch.arange(
        num_routed_copies,
        device=flattened_selected_heads.device,
        dtype=torch.long,
    )
    flat_packed_transfer_indices = (
        flat_routed_row_indices + flat_padding_for_routed_rows
    )

    # -----------------------------------------------------------------------
    # Materialize each entry through the shared routing and transfer artifacts.
    #
    # Each entry first gathers into the shared routed-copy order. The flat packed
    # allocation supplies padding, and the transfer index writes each routed-copy
    # row into its padded expert slot before the public shape is restored.
    # -----------------------------------------------------------------------
    packed_entries: dict[str, torch.Tensor] = {}
    for key, (tensor, padding_value) in entries.items():
        extra_shape = tensor.shape[2:]

        # The sorted source index is shared across all entries; expanding it over
        # trailing dimensions lets the same routing/order plan apply to hidden
        # states, positions, masks, probabilities, and any other packed tensor.
        sorted_gather_indices = sorted_source_token_indices.view(
            batch_size,
            num_routed_copies_per_batch,
            *(1,) * len(extra_shape),
        ).expand(-1, -1, *extra_shape)
        sorted_tensor = tensor.gather(dim=1, index=sorted_gather_indices)

        packed_tensor = tensor.new_full(
            (batch_size * num_experts * packed_length, *extra_shape),
            fill_value=padding_value,
        )
        packed_tensor[flat_packed_transfer_indices] = sorted_tensor.reshape(
            num_routed_copies,
            *extra_shape,
        )
        packed_entries[key] = packed_tensor.reshape(
            batch_size,
            num_experts,
            packed_length,
            *extra_shape,
        )

    return packed_entries, flat_packed_transfer_indices


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------

def unpack_experts(
    expert_outputs: torch.Tensor,
    setup: dict[str, torch.Tensor],
    flat_packed_transfer_indices: torch.Tensor,
    selected_heads: torch.Tensor,
) -> torch.Tensor:
    """Restore token-choice ordering from BEA expert-choice output.

    Args:
        expert_outputs: Expert-choice BEA output y of shape (B, L, T, d).
        setup: Auxiliary payload returned by setup_packing().
        flat_packed_transfer_indices: Transfer index returned by pack_experts().
            Each value identifies a routed-copy slot in the flattened padded
            expert-choice frame.
        selected_heads: Routed head selections I of shape (B, N, K).

    Returns:
        Restored token-choice tensor y_tilde of shape (B, N, K, d).
    """
    inverse_permutation = setup["inverse_permutation"]

    batch_size, sequence_length, num_selected_heads = selected_heads.shape
    num_routed_copies_per_batch = sequence_length * num_selected_heads
    hidden_dim = expert_outputs.shape[-1]

    # -----------------------------------------------------------------------
    # Recover routed-copy rows from the same packed slots used at insertion.
    #
    # Packing writes into the forwarded flat slots, and unpacking reads from those
    # same slots before applying the inverse routing permutation back to
    # token-choice order.
    # -----------------------------------------------------------------------
    flat_expert_outputs = expert_outputs.reshape(-1, hidden_dim)
    flat_routed_copy_outputs = flat_expert_outputs[flat_packed_transfer_indices]

    sorted_token_choice_outputs = flat_routed_copy_outputs.reshape(
        batch_size,
        num_routed_copies_per_batch,
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


def _enforce_no_overflow(tokens_per_expert: torch.Tensor, packed_length: int) -> None:
    """Enforce that no expert bucket exceeds the preallocated packed length.

    This check fires when the number of tokens assigned to any expert in any batch
    item exceeds mosrah_packed_length. When that limit is exceeded, the packed buffer
    is too small to hold all assignments and data would be dropped. Reduce the input
    sequence length or increase training_sequence_length (for training) or
    inference_sequence_length (for inference) in ShramConfig to resolve.

    Args:
        tokens_per_expert: Per-expert token counts, shape (B, num_experts).
        packed_length: The preallocated packed time dimension.
    """
    if torch.compiler.is_compiling():
        torch._assert_async(
            tokens_per_expert.max() <= packed_length,
            "Expert packing overflow: expert bucket exceeds mosrah_packed_length. "
            "Reduce sequence length or increase training_sequence_length / "
            "inference_sequence_length in ShramConfig.",
        )
    else:
        max_count = tokens_per_expert.max().item()
        if max_count > packed_length:
            raise RuntimeError(
                "Expert packing overflow: at least one expert bucket contains more "
                "tokens than mosrah_packed_length allows. Reduce sequence length or "
                "increase training_sequence_length / inference_sequence_length in "
                "ShramConfig to resolve.\n"
                f"Packed length: {packed_length}\n"
                f"Head lengths: {tokens_per_expert}\n"
            )


def _count_tokens_per_expert(
    flattened_selected_heads: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count how many routed token copies are assigned to each expert per batch item.

    Uses scatter_add into a pre-sized (B, num_experts) buffer. Each position in
    flattened_selected_heads contributes one count to the corresponding expert slot.

    Args:
        flattened_selected_heads: Expert assignments of shape (B, N*K) with values
            in [0, num_experts).
        num_experts: Total number of experts L.

    Returns:
        Counts tensor of shape (B, num_experts).
    """
    batch_size = flattened_selected_heads.shape[0]
    tokens_per_expert = torch.zeros(
        batch_size,
        num_experts,
        device=flattened_selected_heads.device,
        dtype=torch.long,
    )
    tokens_per_expert.scatter_add_(
        dim=1,
        index=flattened_selected_heads,
        src=torch.ones_like(flattened_selected_heads, dtype=torch.long),
    )
    return tokens_per_expert