"""Unvectorized reference implementation of the MoSRAH sparse KV cache.

This module exists solely as a correctness oracle. SlowMoSRAHCache implements the same
interface and storage layout as MoSRAHCache but uses an explicit Python loop over
(b, l, t) triples in update(). The loop is obviously correct by inspection: each active
position's key and value are written to the next available slot for that (batch, head)
pair, in the order positions appear along the T dimension, which directly enforces
causal ordering without any index arithmetic to verify.

SlowMoSRAHCache is never instantiated in the model path. Its role is to provide a
trusted ground truth against which the vectorized MoSRAHCache.update() is validated in
Unit 6.A tests, and as a reference for the Unit 10.A position decoder. Because the
vectorized implementation is validated by asserting exact agreement with this one on all
test inputs, the correctness of SlowMoSRAHCache is load-bearing: its own test suite
(test_slow_mosrah_cache.py) must establish it is trustworthy before it can be used as
an oracle.
"""

import torch
from transformers.cache_utils import CacheLayerMixin


class SlowMoSRAHCache(CacheLayerMixin):
    """Unvectorized reference implementation of the MoSRAH KV cache.

    Identical storage layout to MoSRAHCache: (B, L, T, u) tensors in the
    mixin-standard self.keys and self.values attributes, plus a (B, L) _counts tensor,
    with the same constructor signature and the same CacheLayerMixin protocol methods.
    The sole difference is update(), which uses an explicit Python loop over (b, l, t)
    triples rather than vectorized index arithmetic.

    This class is not used in the model path. It exists so that MoSRAHCache.update()
    can be validated by asserting exact agreement with this implementation on all test
    inputs. See module docstring for the trust chain this enables.

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
        """Scatter active key/value states using an explicit loop; return full cache state.

        Iterates over every (b, l, t) triple. For each position where active_mask is
        True, the key and value are written to the next available slot for that
        (batch, head) pair and the count is incremented. Causal ordering is guaranteed
        because the t dimension is traversed from 0 to T-1 and counts are updated
        immediately after each write.

        Buffer expansion (doubling buffer_capacity) is triggered before any writes if
        the incoming tokens would cause any slot to overflow the current capacity.

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
        B, L, T = active_mask.shape

        # Expansion check uses the total active tokens per slot, same as the
        # vectorized implementation, so both expand under identical conditions.
        incoming_delta = active_mask.long().sum(dim=2)  # (B, L)
        if (self._counts + incoming_delta).max().item() > self.buffer_capacity:
            self._expand()

        # Write each active position into the next available slot for its (batch, head)
        # pair. Iterating t from 0 to T-1 preserves causal ordering within each slot.
        for b in range(B):
            for l in range(L):
                for t in range(T):
                    if active_mask[b, l, t]:
                        pos = self._counts[b, l].item()
                        self.keys[b, l, pos, :] = key_states[b, l, t, :]
                        self.values[b, l, pos, :] = value_states[b, l, t, :]
                        self._counts[b, l] += 1

        return self.keys, self.values, self._make_active_mask()

    def get_heads_lengths(self) -> torch.Tensor:
        """Return the per-(batch, head) token count for this layer.

        This is the authoritative occupancy tensor consumed by BEA for attention
        masking and by position computation (Unit 10.A) for semantic-sequence
        position computation.

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

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension by repeating each entry repeats times.

        Used at beam search initialisation to expand the cache from batch size B to
        B * repeats, matching the expanded beam candidate batch. Applied atomically
        across keys, values, and _counts; batch_size is updated to reflect the new size.

        Args:
            repeats: Number of times to repeat each batch entry.
        """
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self._counts = self._counts.repeat_interleave(repeats, dim=0)
        self.batch_size = self.batch_size * repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset of batch entries by index.

        Used in contrastive search to retain only the selected candidate entries.
        Applied atomically across keys, values, and _counts; batch_size is updated
        to reflect the number of retained entries.

        Args:
            indices: 1-D integer tensor of batch indices to retain.
        """
        self.keys = self.keys[indices]
        self.values = self.values[indices]
        self._counts = self._counts[indices]
        self.batch_size = indices.shape[0]

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
            "SlowMoSRAHCache has no single sequence length. "
            "Use get_heads_lengths() for per-head occupancy."
        )

    def get_max_cache_shape(self) -> int:  # type: ignore[override]
        """Not supported — SlowMoSRAHCache is dynamic and unbounded."""
        raise NotImplementedError(
            "SlowMoSRAHCache is unbounded; get_max_cache_shape() is not supported."
        )

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        """Not supported — SlowMoSRAHCache does not participate in HF mask construction."""
        raise NotImplementedError(
            "SlowMoSRAHCache does not support get_mask_sizes()."
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
