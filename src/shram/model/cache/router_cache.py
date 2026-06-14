"""Block-state cache for the MoSRAH causal block-balanced router.

The block-balanced router partitions the token sequence into non-overlapping blocks
of W = L/K tokens. Within each block every expert is assigned exactly once, giving
perfect load balance by construction. During training the full sequence is available
and block state is managed locally in MoSRAHRouter.forward(). During inference tokens
arrive one at a time and the router must remember which experts have been claimed in
the current partial block across decode steps.

RouterCache holds two pieces of state across decode steps:

  - _used_in_block: Boolean mask (B, L) tracking which experts have been claimed by
    earlier tokens in the current block. The decode router masks these to -inf before
    TopK, preserving the one-usage-per-block invariant.

  - _step_in_block: Integer counter (B,) of how many tokens have been processed in
    the current block. Reaches block_length W when the block completes, at which
    point both tensors are reset in-place for the next block.

All decode-step operations (update_decode) use fixed-shape in-place tensor ops and
are fully compileable under torch.compile(dynamic=False, fullgraph=True). The prefill
update (update_prefill) may use data-dependent indexing and must not be called inside
a compiled graph; prefill runs in eager mode before the compiled decode loop in
standard HuggingFace generate().

RouterCache is constructed by ShramLayerCache and passed directly to
MoSRAHRouter.forward(). ShramLayerCache.reset() clears the router state atomically
with the KV caches it also owns.
"""

import torch
from transformers.cache_utils import CacheLayerMixin


class RouterCache(CacheLayerMixin):
    """Block-state cache for the MoSRAH causal block-balanced router.

    Tracks which experts have been claimed in the current routing block and how
    far into that block the current decode step is. This allows the router to
    maintain its one-usage-per-block contract across decode steps without
    reprocessing the full accumulated sequence.

    All state is pre-allocated at construction time. The primary decode method
    (update_decode) uses only in-place fixed-shape operations and is fully
    compileable.

    Args:
        block_length: Tokens per routing block, W = num_mosrah_heads // num_selected_heads.
            The router resets block state after every W consecutive decode tokens.
        num_mosrah_heads: Total expert count L. Determines the width of the
            used-expert mask.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate state tensors.
    """

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        block_length: int,
        num_mosrah_heads: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._block_length = block_length
        self._device = device

        # used_in_block: which experts are already claimed in the current block.
        # False = expert is still available for the next decode token that needs it.
        # Reset to all-False when step_in_block reaches block_length.
        self._used_in_block = torch.zeros(
            batch_size, num_mosrah_heads, dtype=torch.bool, device=device
        )

        # step_in_block: how many tokens have been processed in the current block.
        # Range [0, block_length - 1]. Resets to 0 when a block completes.
        self._step_in_block = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # ---------------------------------------------------------------------------
    # is_initialized — pre-allocated at construction, always True
    # ---------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True always — RouterCache pre-allocates all state at construction."""
        return True

    @is_initialized.setter
    def is_initialized(self, value: bool) -> None:
        # CacheLayerMixin.__init__ assigns self.is_initialized = False as an
        # instance attribute. Absorb it silently — state is always initialized.
        pass

    # ---------------------------------------------------------------------------
    # Public interface for the router
    # ---------------------------------------------------------------------------

    def get_used_in_block(self) -> torch.Tensor:
        """Return the current block's used-expert mask.

        Returns:
            Boolean mask of shape (B, L). True entries mark experts already claimed
            by earlier tokens in the current block and must be excluded from TopK.
        """
        return self._used_in_block

    def update_decode(self, step_heads: torch.Tensor) -> None:
        """Record a single decode-step expert selection and advance the block counter.

        Marks the K selected experts as used in the current block, then either
        advances the per-batch step counter or resets both tensors in-place when
        the block completes. All operations are in-place and compile-compatible.

        Args:
            step_heads: Expert indices selected at this decode step, shape (B, K).
        """
        # Mark the K selected experts as unavailable for the rest of this block.
        self._used_in_block.scatter_(-1, step_heads, True)

        # Detect block completion before incrementing: step was W-1 (0-indexed),
        # meaning this token is the last one in the current block.
        block_done = self._step_in_block.eq(self._block_length - 1)  # (B,) bool

        # Advance step counter, then zero it for any batch item that just finished a block.
        self._step_in_block.add_(1)
        self._step_in_block.masked_fill_(block_done, 0)

        # Clear expert availability for batch items that completed a block, so the
        # next decode token for those items starts with a clean slate.
        self._used_in_block.masked_fill_(block_done.unsqueeze(-1), False)

    def update_prefill(
        self,
        selected_heads_blocked: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Record the partial block state left over at the end of a prefill pass.

        After processing a prefill sequence of length seq_len with the training-style
        block solver, the last block may be incomplete when seq_len is not a multiple
        of block_length. This method saves the partial block state so decode steps can
        continue the current block without a gap.

        Not compile-compatible: uses a data-dependent slice [:seq_mod] on the W
        dimension. Must only be called in eager mode. Standard HuggingFace generate()
        runs prefill in eager before entering the compiled decode loop.

        Args:
            selected_heads_blocked: Block-solver assignment output from the prefill pass,
                shape (B, num_blocks, W, K). The final block entry contains expert
                assignments for both real tokens (steps 0..seq_mod-1) and padding
                artefacts (steps seq_mod..W-1) which must be discarded.
            seq_len: Actual prefill sequence length before block padding. Determines
                how many steps of the last block contain real assignments.
        """
        B = selected_heads_blocked.shape[0]
        seq_mod = seq_len % self._block_length

        self._used_in_block.zero_()

        if seq_mod == 0:
            # All blocks were complete — start fresh for the next decode token.
            self._step_in_block.zero_()
        else:
            # Last block is partial: only the first seq_mod steps are real assignments.
            # Rebuild the used-expert mask from those steps and record the step position.
            last_block_real_steps = selected_heads_blocked[:, -1, :seq_mod, :]  # (B, seq_mod, K)
            real_experts_flat = last_block_real_steps.reshape(B, -1)             # (B, seq_mod * K)
            self._used_in_block.scatter_(-1, real_experts_flat, True)
            self._step_in_block.fill_(seq_mod)

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — reset and beam-search coordination
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear block state for a new generation session.

        Zeros both state tensors in-place. Called by ShramLayerCache.reset()
        atomically with the KV cache reset.
        """
        self._used_in_block.zero_()
        self._step_in_block.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension for beam search.

        Args:
            beam_idx: Permutation indices of shape (batch,).
        """
        self._used_in_block = self._used_in_block[beam_idx]
        self._step_in_block = self._step_in_block[beam_idx]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension for beam search initialisation.

        Args:
            repeats: Number of times to repeat each batch entry along the batch dimension.
        """
        self._used_in_block = self._used_in_block.repeat_interleave(repeats, dim=0)
        self._step_in_block = self._step_in_block.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset of batch entries for contrastive search.

        Args:
            indices: 1-D integer tensor of batch indices to retain.
        """
        self._used_in_block = self._used_in_block[indices]
        self._step_in_block = self._step_in_block[indices]

    def offload(self) -> None:
        """Move state tensors to CPU for memory management between decode steps."""
        self._used_in_block = self._used_in_block.cpu()
        self._step_in_block = self._step_in_block.cpu()

    def prefetch(self) -> None:
        """Move state tensors back to model device ahead of the next decode step."""
        self._used_in_block = self._used_in_block.to(self._device)
        self._step_in_block = self._step_in_block.to(self._device)

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — unsupported abstract methods
    # ---------------------------------------------------------------------------

    def lazy_initialization(  # type: ignore[override]
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """No-op — RouterCache pre-allocates all state at construction."""
        pass

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported — use update_decode() or update_prefill() instead."""
        raise NotImplementedError(
            "RouterCache has no composite key/value update interface. "
            "Use update_decode() for single decode steps or update_prefill() after prefill."
        )

    def get_seq_length(self) -> int:
        """Not supported — RouterCache tracks block position, not sequence length."""
        raise NotImplementedError("RouterCache does not track sequence length.")

    def get_max_cache_shape(self) -> int:
        """Not supported — RouterCache does not hold KV pairs."""
        raise NotImplementedError("RouterCache does not have a KV cache shape.")

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        """Not supported — RouterCache does not participate in KV attention masking."""
        raise NotImplementedError("RouterCache does not participate in KV masking.")
