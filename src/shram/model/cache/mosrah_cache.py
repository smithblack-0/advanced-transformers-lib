"""MoSRAH sparse KV cache — single-layer implementation.

MoSRAH routes each token to K of L available expert heads, so its KV cache is indexed
by head rather than by sequence position. The routing is dynamic and produces a ragged
distribution of token counts across (batch, head) slots — different batch items may
route different numbers of tokens to the same head, and different heads accumulate at
different rates. DynamicCache cannot represent this correctly: it concatenates along
the sequence dimension and assumes uniform token counts across the batch. MoSRAHCache
therefore uses a custom buffer design.

Keys and values are stored in the CacheLayerMixin-standard self.keys and self.values
attributes as (B, L, T, u) tensors, where B is batch size, L is the number of expert
heads (num_mosrah_heads), T is the current buffer capacity, and u is the bottlenecked
head embedding width (head_dim). A (B, L) integer count tensor _counts tracks the
valid occupancy of each (batch, head) slot. Buffer capacity is exposed as the
buffer_capacity property and is derived directly from self.keys rather than tracked
as a separate variable.

The primary interface is update(key_states, value_states, active_mask), which accepts
expert-choice layout, stores only active entries in causal order, and returns the full
accumulated (keys, values, active_mask) for immediate use by BEA. The returned
active_mask identifies valid cached positions; everything beyond each slot's count is
junk data that downstream attention must exclude.

BEA applies RoPE and calls update() with post-RoPE keys (K̃). The occupancy counts
exposed by get_heads_lengths() must be read before update() if the caller needs the
pre-update occupancy for position computation (Unit 10.A). update() increments counts
in-place and the pre-update values are not recoverable afterward.

All buffers are allocated at construction time. MoSRAHCache is constructed by
ShramLayerCache, which has access to batch size, device, and all model config parameters
needed to fully specify the storage layout upfront.
"""

import torch
from transformers.cache_utils import CacheLayerMixin


class MoSRAHCache(CacheLayerMixin):
    """KV cache for the MoSRAH sparse attention path — single decoder layer.

    Subclasses CacheLayerMixin to satisfy the HuggingFace per-layer cache role.
    Stores keys and values in the mixin-standard self.keys and self.values attributes
    using a custom (B, L, T, u) layout rather than delegating to DynamicCache,
    which cannot represent MoSRAH's ragged per-(batch, head) token counts correctly.

    All storage is allocated at construction time and is_initialized is True
    immediately. The caller (ShramLayerCache) provides batch size, device, and model
    config parameters so no lazy allocation is needed.

    Input is expected in expert-choice layout: (B, L, T, u) key/value tensors with a
    (B, L, T) boolean active_mask. Only positions where active_mask is True are written.
    This matches the packed representation produced by expert packing in the MoSRAH
    forward pass, where BEA has already applied RoPE before calling update().

    Args:
        num_mosrah_heads: Total number of MoSRAH expert heads (L). Determines the
            second dimension of all storage tensors.
        head_dim: Bottlenecked head embedding width (u). Determines the fourth
            dimension of all storage tensors.
        batch_size: Number of sequences in the batch. Determines the first dimension
            of all storage tensors.
        device: Device on which to allocate all tensors. Should match the model device.
        initial_buffer_size: Initial sequence capacity per (batch, head) slot. Doubled
            when any slot overflows. Defaults to 64 to avoid repeated reallocation
            during prompt processing.
    """

    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        num_mosrah_heads: int,
        head_dim: int,
        batch_size: int,
        device: torch.device,
        initial_buffer_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_mosrah_heads = num_mosrah_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device

        # Allocate primary storage into the mixin-standard self.keys / self.values so
        # that inherited methods (offload, prefetch) operate on real tensors. _counts
        # tracks valid occupancy per (batch, head) slot.
        self.keys: torch.Tensor = torch.zeros(
            batch_size, num_mosrah_heads, initial_buffer_size, head_dim, device=device
        )
        self.values: torch.Tensor = torch.zeros(
            batch_size, num_mosrah_heads, initial_buffer_size, head_dim, device=device
        )
        self._counts: torch.Tensor = torch.zeros(
            batch_size, num_mosrah_heads, dtype=torch.long, device=device
        )

        # Storage is fully allocated at construction — the cache is initialized.
        self.is_initialized = True

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def buffer_capacity(self) -> int:
        """Current number of slots allocated per (batch, head) pair.

        Derived directly from self.keys rather than tracked separately, so it is
        always consistent with the actual buffer after expansion.
        """
        return self.keys.shape[2]

    # ---------------------------------------------------------------------------
    # Primary API
    # ---------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        active_mask: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter active key/value states into the buffer and return the full cache state.

        Accepts expert-choice layout: key_states and value_states are (B, L, T, u);
        active_mask is (B, L, T) bool with True marking real tokens. Only active
        positions are written; inactive positions are ignored.

        Uses a cumsum construction to derive the absolute buffer position for each
        active token without any Python loops. For a given (batch, head) slot,
        positions are assigned in the order tokens appear along the T dimension,
        preserving causal ordering.

        Returns the full accumulated (keys, values, active_mask) across the cached
        sparse sequence. The returned active_mask is True exactly for slots t <
        counts[b, l]; everything beyond is junk data that BEA must exclude.

        Note: get_heads_lengths() must be called before update() if the caller needs
        the pre-update occupancy for position computation (Unit 10.A). update()
        increments counts in-place and the pre-update values are not recoverable.

        Args:
            key_states: Shape (B, L, T, u) — post-RoPE key vectors in expert-choice layout.
            value_states: Shape (B, L, T, u) — value vectors in expert-choice layout.
            active_mask: Shape (B, L, T) bool — True for real tokens, False for padding.
            cache_kwargs: Unused; present to satisfy the CacheLayerMixin signature.

        Returns:
            Tuple of (keys, values, active_mask):
              keys: (B, L, T, u) float — full key buffer including junk slots.
              values: (B, L, T, u) float — full value buffer including junk slots.
              active_mask: (B, L, T) bool — True iff slot (b, l, t) has been written.
        """
        incoming_delta = active_mask.long().sum(dim=2)  # (B, L)

        if (self._counts + incoming_delta).max().item() > self.buffer_capacity:
            self._expand()

        # Cumulative count of active positions along T for each (b, l) slot. Entry
        # [b, l, t] is the 1-based rank of position t among all active positions in
        # that slot. Subtract 1 for a zero-based within-slot index. Inactive positions
        # produce a negative value, which is excluded by the mask gate below.
        within_slot = active_mask.long().cumsum(dim=2) - 1  # (B, L, T)

        # Add the pre-update count to get the absolute buffer position for each
        # active token.
        abs_pos = within_slot + self._counts.unsqueeze(-1)  # (B, L, T)

        # Scatter key and value vectors at all active positions.
        b_idx, l_idx, t_idx = torch.where(active_mask)
        self.keys[b_idx, l_idx, abs_pos[b_idx, l_idx, t_idx]] = (
            key_states[b_idx, l_idx, t_idx]
        )
        self.values[b_idx, l_idx, abs_pos[b_idx, l_idx, t_idx]] = (
            value_states[b_idx, l_idx, t_idx]
        )

        self._counts += incoming_delta

        return self.keys, self.values, self._make_active_mask()

    def get_heads_lengths(self) -> torch.Tensor:
        """Return the per-(batch, head) token count for this layer.

        This is the authoritative occupancy tensor consumed by BEA for attention
        masking and by position computation (Unit 10.A) for semantic-sequence
        position computation.

        Note: in the MoSRAH forward pass, this must be called before update() if the
        caller needs the pre-update occupancy. update() increments these counts in-place.

        Returns:
            Integer tensor of shape (B, L) where entry [b, h] is the number of valid
            tokens stored in the (b, h) slot. Zero for slots with no writes yet.
        """
        return self._counts

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — overridden coordination methods
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cached key and value tensors.

        Zeroes self.keys, self.values, and _counts in place. Storage remains allocated
        and is_initialized remains True — only the contents are cleared.
        """
        self.keys.zero_()
        self.values.zero_()
        self._counts.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension of all cached tensors for beam search.

        Applied atomically across self.keys, self.values, and _counts. Beam search
        must reorder all three together or the occupancy counts and buffer contents
        will correspond to different beam hypotheses.

        Overrides the parent because the parent's implementation calls get_seq_length(),
        which is not supported for this cache.

        Args:
            beam_idx: Permutation indices of shape (batch,) produced by the beam
                search algorithm.
        """
        self.keys = self.keys[beam_idx]
        self.values = self.values[beam_idx]
        self._counts = self._counts[beam_idx]

    def offload(self) -> None:
        """Offload all cached tensors to CPU.

        Extends the parent to also offload _counts, which the parent does not know
        about. All three tensors are moved atomically so device state remains consistent.
        """
        super().offload()
        self._counts = self._counts.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        """Move all cached tensors back to the model device ahead of time.

        Extends the parent to also prefetch _counts, which the parent does not know
        about. _counts is synced to self.keys.device after the parent moves keys and
        values, so all three remain consistent.
        """
        super().prefetch()
        if self._counts.device != self.keys.device:
            self._counts = self._counts.to(self.keys.device, non_blocking=True)

    def lazy_initialization(  # type: ignore[override]
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """No-op — storage is fully allocated at construction time."""
        pass

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — unsupported abstract methods
    # ---------------------------------------------------------------------------

    def get_seq_length(self) -> int:  # type: ignore[override]
        """Not supported — no single sequence length represents this cache's state.

        MoSRAH heads accumulate independently; (batch, head) slots have different
        lengths depending on routing history. There is no meaningful scalar summary.
        Use get_heads_lengths() for per-head occupancy.
        """
        raise NotImplementedError(
            "MoSRAHCache has no single sequence length. "
            "Use get_heads_lengths() for per-head occupancy."
        )

    def get_max_cache_shape(self) -> int:  # type: ignore[override]
        """Not supported — MoSRAHCache is dynamic and unbounded."""
        raise NotImplementedError(
            "MoSRAHCache is unbounded; get_max_cache_shape() is not supported."
        )

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        """Not supported — MoSRAHCache does not participate in HF mask construction."""
        raise NotImplementedError(
            "MoSRAHCache does not support get_mask_sizes()."
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _make_active_mask(self) -> torch.Tensor:
        """Construct the (B, L, T) active mask from current counts.

        Returns True at position [b, l, t] iff t < _counts[b, l], i.e. the slot
        has been written. Positions at or beyond the count are junk and must be
        excluded by downstream attention.
        """
        cap = self.buffer_capacity
        return (
            torch.arange(cap, device=self.keys.device)
            .expand(self.batch_size, self.num_mosrah_heads, cap)
            < self._counts.unsqueeze(-1)
        )

    def _expand(self) -> None:
        """Double the buffer capacity, preserving existing data.

        Called by update() when an incoming batch of tokens would cause any
        (batch, head) slot to exceed the current buffer capacity. All existing
        key and value data is copied into the low half of the new buffer; the
        high half is zero-initialised and will be filled by subsequent writes.
        After reassignment, buffer_capacity reflects the new size automatically.
        """
        old_cap = self.buffer_capacity
        new_cap = old_cap * 2
        dev = self.keys.device
        new_keys = torch.zeros(
            self.batch_size, self.num_mosrah_heads, new_cap, self.head_dim, device=dev
        )
        new_values = torch.zeros(
            self.batch_size, self.num_mosrah_heads, new_cap, self.head_dim, device=dev
        )
        new_keys[:, :, :old_cap, :] = self.keys
        new_values[:, :, :old_cap, :] = self.values
        self.keys = new_keys
        self.values = new_values
