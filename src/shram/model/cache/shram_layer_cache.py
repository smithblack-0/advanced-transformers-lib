"""SHRAM per-layer cache — composite owner for one SHRAM decoder layer.

A SHRAM decoder layer contains two distinct attention pathways at one attention slot: the
local sliding-window path and the MoSRAH sparse path. Each path has its own cache with
different semantics and a different downstream consumer. ShramLayerCache owns both, satisfies
the HuggingFace per-layer cache role, and exposes each sub-cache directly so its attention
path can interact with it without indirection.

ShramLayerCache does not define a composite update() interface. The two paths have materially
different update semantics — the sliding-window side uses standard key/value concatenation
while the MoSRAH side uses expert-choice scatter with an active mask — and merging these
behind a single update() would hide those differences behind a misleading abstraction. Instead,
each attention path calls update() on the sub-cache it owns. ShramLayerCache acts as the
ownership, coordination, and reset/reorder boundary for one decoder layer.

The scalar sequence length concept is exposed at this boundary and sourced from the
sliding-window cache, which tracks the full cumulative token count. The MoSRAH cache is
ragged across (batch, head) slots and has no meaningful scalar summary.
"""

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicSlidingWindowLayer

from .mosrah_cache import MoSRAHCache


class ShramLayerCache(CacheLayerMixin):
    """Cache subsystem for one SHRAM decoder layer.

    Owns and coordinates two sub-caches:
      - sliding_window_cache: DynamicSlidingWindowLayer for the local sliding-window path.
      - mosrah_cache: MoSRAHCache for the MoSRAH sparse attention path.

    Satisfies the HuggingFace per-layer cache role (CacheLayerMixin). The two sub-caches are
    exposed directly for their downstream attention paths — no composite update() interface is
    provided, because the two paths have materially different update semantics.

    The scalar sequence length is sourced from the sliding-window cache, which tracks the full
    cumulative token count across all forward passes. The MoSRAH cache is ragged across
    (batch, head) slots and cannot contribute a truthful scalar summary.

    Args:
        sliding_window: Number of tokens retained by the sliding-window cache.
        num_mosrah_heads: Total number of MoSRAH expert heads (L).
        mosrah_head_dim: Bottlenecked head embedding width (u) for the MoSRAH path.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate MoSRAH cache tensors.
        initial_buffer_size: Initial per-(batch, head) capacity for MoSRAHCache. Doubled
            when any slot overflows. Defaults to 64 to avoid repeated reallocation during
            prompt processing.
    """

    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        sliding_window: int,
        num_mosrah_heads: int,
        mosrah_head_dim: int,
        batch_size: int,
        device: torch.device,
        initial_buffer_size: int = 64,
    ) -> None:
        super().__init__()
        self.sliding_window_cache = DynamicSlidingWindowLayer(sliding_window=sliding_window)
        self.mosrah_cache = MoSRAHCache(
            num_mosrah_heads=num_mosrah_heads,
            head_dim=mosrah_head_dim,
            batch_size=batch_size,
            device=device,
            initial_buffer_size=initial_buffer_size,
        )

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True iff both sub-caches have allocated their storage.

        Derived from the sub-caches rather than tracked as a flag, so it is always
        consistent with actual sub-cache state. The effective gate is the sliding-window
        cache: MoSRAHCache pre-allocates at construction and is always ready, while
        DynamicSlidingWindowLayer allocates lazily on the first update() call.
        """
        return self.sliding_window_cache.is_initialized and self.mosrah_cache.is_initialized

    @is_initialized.setter
    def is_initialized(self, value: bool) -> None:
        # CacheLayerMixin.__init__ assigns self.is_initialized = False as an instance
        # attribute. Since property is a data descriptor it takes precedence, but Python
        # still routes the assignment through __set__. Absorb it silently — state is
        # derived from sub-caches, not stored here.
        pass

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — composite-meaningful methods
    # ---------------------------------------------------------------------------

    def get_seq_length(self) -> int:
        """Return the cumulative token count from the sliding-window path.

        The sliding-window cache tracks the total number of tokens seen across all forward
        passes, which is the meaningful scalar sequence length at this layer boundary. The
        MoSRAH cache is ragged across (batch, head) slots and cannot contribute a truthful
        scalar summary.
        """
        return self.sliding_window_cache.get_seq_length()

    def reset(self) -> None:
        """Clear both sub-caches.

        Delegates reset to each sub-cache. Both are cleared atomically so the sliding-window
        state and MoSRAH sparse state remain consistent.
        """
        self.sliding_window_cache.reset()
        self.mosrah_cache.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension of both sub-caches for beam search.

        Delegates to each sub-cache. Both are reordered atomically so the sliding-window
        and MoSRAH state correspond to the same beam hypotheses after reordering.

        Args:
            beam_idx: Permutation indices of shape (batch,) produced by beam search.
        """
        self.sliding_window_cache.reorder_cache(beam_idx)
        self.mosrah_cache.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension of both sub-caches for beam search initialisation.

        Delegates atomically to each sub-cache. Both must be expanded together so the
        sliding-window and MoSRAH state correspond to the same beam candidates.

        Args:
            repeats: Number of times to repeat each batch entry.
        """
        self.sliding_window_cache.batch_repeat_interleave(repeats)
        self.mosrah_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset of batch entries in both sub-caches for contrastive search.

        Delegates atomically to each sub-cache. Both must be trimmed together so the
        sliding-window and MoSRAH state remain consistent.

        Args:
            indices: 1-D integer tensor of batch indices to retain.
        """
        self.sliding_window_cache.batch_select_indices(indices)
        self.mosrah_cache.batch_select_indices(indices)

    def offload(self) -> None:
        """Offload both sub-caches to CPU.

        Delegates to each sub-cache's offload method. Does not call super() — ShramLayerCache
        does not own self.keys/self.values directly; all cached data lives in the sub-caches.
        """
        self.sliding_window_cache.offload()
        self.mosrah_cache.offload()

    def prefetch(self) -> None:
        """Move both sub-caches back to their model device ahead of time.

        Delegates to each sub-cache's prefetch method. Does not call super() — ShramLayerCache
        does not own self.keys/self.values directly; all cached data lives in the sub-caches.
        """
        self.sliding_window_cache.prefetch()
        self.mosrah_cache.prefetch()

    def lazy_initialization(  # type: ignore[override]
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """No-op — both sub-caches handle their own initialization."""
        pass

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — unsupported abstract methods
    # ---------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported — ShramLayerCache has no composite update interface.

        The two sub-caches have materially different update semantics: the sliding-window
        side uses standard key/value concatenation while the MoSRAH side uses expert-choice
        scatter with an active mask. Callers must update each sub-cache directly via
        sliding_window_cache.update() or mosrah_cache.update().
        """
        raise NotImplementedError(
            "ShramLayerCache has no composite update interface. "
            "Update sliding_window_cache or mosrah_cache directly."
        )

    def get_max_cache_shape(self) -> int:  # type: ignore[override]
        """Not supported — the composite cache has no single maximum shape.

        The sliding-window cache is bounded by sliding_window; the MoSRAH cache is
        unbounded. No truthful scalar maximum represents the composite.
        """
        raise NotImplementedError(
            "ShramLayerCache has no single maximum cache shape. "
            "Query sliding_window_cache or mosrah_cache directly."
        )

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        """Not supported — ShramLayerCache does not participate in HF mask construction.

        The two sub-caches have different mask semantics and their respective attention
        paths handle masking directly.
        """
        raise NotImplementedError(
            "ShramLayerCache does not support get_mask_sizes(). "
            "Query sliding_window_cache or mosrah_cache directly."
        )
