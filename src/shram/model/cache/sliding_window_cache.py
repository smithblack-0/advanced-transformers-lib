# src/shram/model/cache/sliding_window_cache.py

"""Local sliding-window cache for the SHRAM local attention path.

This file defines `LocalSlidingWindowLayerCache`, the local sub-cache owned by
`ShramLayerCache` and consumed by `SlidingWindowAttention`.

Its job is narrow:

- accept the current chunk's local key/value tensors and active mask
- return the current-step local frame consumed by local attention
- separately retain the next-step sliding-window cache state

It does not decide local causal visibility. That is owned by
`SlidingWindowAttention`, which consumes the returned key/value/mask frame and
constructs the effective local attention mask from it.
"""

import torch
from transformers.cache_utils import CacheLayerMixin


class LocalSlidingWindowLayerCache(CacheLayerMixin):
    """Fixed-width local cache for one SHRAM decoder layer.

    The cache keeps a retained local sliding-window buffer and an aligned active
    mask. On update, it returns the current-step local frame formed by
    concatenating retained cache state with the new chunk, then remembers only
    the last `sliding_window` positions for the next step.

    Dead positions are allowed to remain in both the returned frame and the
    retained cache. Correctness is carried by the aligned active mask.

    Args:
        sliding_window: Width of the retained local sliding-window buffer.
        num_heads: Number of local attention heads.
        head_dim: Per-head embedding width for the local path.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate cache storage.
    """

    is_compileable = False
    is_sliding = True

    def __init__(
        self,
        sliding_window: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        if sliding_window < 1:
            raise ValueError(
                f"sliding_window must be >= 1, got {sliding_window}."
            )
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self.sliding_window = sliding_window
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device

        # Retained next-step local cache state. Storage is fixed-width from the
        # start; semantic validity is carried by `active_mask`.
        self.keys = torch.zeros(
            batch_size,
            num_heads,
            sliding_window,
            head_dim,
            device=device,
        )
        self.values = torch.zeros(
            batch_size,
            num_heads,
            sliding_window,
            head_dim,
            device=device,
        )
        self.active_mask = torch.zeros(
            batch_size,
            sliding_window,
            dtype=torch.bool,
            device=device,
        )

        self.is_initialized = True

        # Cumulative count of all token positions presented through update() for
        # this cache instance. This is the quantity HuggingFace generation reads
        # through get_seq_length() to track how far along the sequence we are.
        self._total_processed: int = 0

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        active_mask: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the current-step local frame and retain the next-step window.

        Args:
            key_states: Shape `(B, H, T_new, D)` local key vectors for the
                current chunk.
            value_states: Shape `(B, H, T_new, D)` local value vectors for the
                current chunk.
            active_mask: Shape `(B, T_new)` bool. `True` means the
                corresponding token position in the current chunk is active.
            cache_kwargs: Present only to satisfy the `CacheLayerMixin`
                interface. Unused by this cache.

        Returns:
            Tuple of:
              - visible_keys: `(B, H, sliding_window + T_new, D)`
              - visible_values: `(B, H, sliding_window + T_new, D)`
              - visible_active_mask: `(B, sliding_window + T_new)`

            These are the tensors the local attention path should consume
            directly for the current step.
        """
        self._ensure_state_compatibility(
            key_states=key_states,
            value_states=value_states,
        )

        # The current-step local frame is just retained cache state followed by
        # the current chunk in chronological order.
        composite_keys, composite_values, composite_mask = self._make_composite_frame(
            key_states=key_states,
            value_states=value_states,
            active_mask=active_mask,
        )

        # The cache remembers only the last raw sliding-window positions of that
        # composite frame for the next step. Dead positions are allowed to
        # survive; downstream local attention will ignore them using the mask.
        self._retain_next_window(
            composite_keys=composite_keys,
            composite_values=composite_values,
            composite_mask=composite_mask,
        )

        self._total_processed += key_states.shape[2]

        return composite_keys, composite_values, composite_mask

    def _ensure_state_compatibility(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Keep retained cache buffers compatible with the incoming update tensors.

        The cache is allocated eagerly for simplicity. If later updates arrive on
        a different device or in a different floating dtype, move the retained
        state to match while preserving its contents.
        """
        if self.keys.dtype != key_states.dtype or self.keys.device != key_states.device:
            self.keys = self.keys.to(
                device=key_states.device,
                dtype=key_states.dtype,
            )

        if (
            self.values.dtype != value_states.dtype
            or self.values.device != value_states.device
        ):
            self.values = self.values.to(
                device=value_states.device,
                dtype=value_states.dtype,
            )

        if self.active_mask.device != key_states.device:
            self.active_mask = self.active_mask.to(
                key_states.device,
                non_blocking=True,
            )

    def _make_composite_frame(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the current-step local frame in chronological order."""
        return (
            torch.cat([self.keys, key_states], dim=-2),
            torch.cat([self.values, value_states], dim=-2),
            torch.cat([self.active_mask, active_mask], dim=-1),
        )

    def _retain_next_window(
        self,
        composite_keys: torch.Tensor,
        composite_values: torch.Tensor,
        composite_mask: torch.Tensor,
    ) -> None:
        """Remember the next-step retained local state.

        This is a raw positional trim to the last `sliding_window` positions, not
        a semantic live-token trim.
        """
        self.keys = composite_keys[:, :, -self.sliding_window :, :]
        self.values = composite_values[:, :, -self.sliding_window :, :]
        self.active_mask = composite_mask[:, -self.sliding_window :]

    def get_seq_length(self) -> int:
        """Return the cumulative number of token positions processed by this cache.

        This is the total count of token positions presented across all update()
        calls since construction or the last reset(). It is the quantity HuggingFace
        generation reads to track sequence progress and is not the same as active-token
        count or current window occupancy.
        """
        return self._total_processed

    def get_max_cache_shape(self) -> int:
        return self.sliding_window

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        raise NotImplementedError(
            "LocalSlidingWindowLayerCache does not support get_mask_sizes()."
        )

    def reset(self) -> None:
        """Restore fresh-cache behavior."""
        self.keys.zero_()
        self.values.zero_()
        self.active_mask.zero_()
        self._total_processed = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension for beam search."""
        self.keys = self.keys[beam_idx]
        self.values = self.values[beam_idx]
        self.active_mask = self.active_mask[beam_idx]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension for beam-search initialisation."""
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self.active_mask = self.active_mask.repeat_interleave(repeats, dim=0)
        self.batch_size = self.batch_size * repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset of batch entries for contrastive search."""
        self.keys = self.keys[indices]
        self.values = self.values[indices]
        self.active_mask = self.active_mask[indices]
        self.batch_size = int(indices.shape[0])

    def offload(self) -> None:
        """Offload cache tensors to CPU."""
        super().offload()
        self.active_mask = self.active_mask.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        """Move cache tensors back to the model device ahead of time."""
        super().prefetch()
        if self.active_mask.device != self.keys.device:
            self.active_mask = self.active_mask.to(
                self.keys.device,
                non_blocking=True,
            )

    def crop(self, max_length: int) -> None:
        raise NotImplementedError(
            "LocalSlidingWindowLayerCache does not support crop()."
        )

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """No-op — this cache allocates its fixed buffers at construction time."""
        return