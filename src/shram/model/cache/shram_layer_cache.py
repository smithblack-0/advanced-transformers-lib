"""SHRAM per-layer cache — composite owner for one SHRAM decoder layer.

A SHRAM decoder layer contains two distinct attention pathways at one attention slot: the
local sliding-window path and the MoSRAH sparse path. Each path has its own cache with
different semantics and a different downstream consumer. ShramLayerCache owns both, satisfies
the HuggingFace per-layer cache role, and exposes each sub-cache directly so its attention
path can interact with it without indirection.

ShramLayerCache does not define a composite update() interface. The two paths have materially
different update semantics — the local side uses chunk-local key/value/mask concatenation
while the MoSRAH side uses expert-choice scatter with an active mask — and merging these
behind a single update() would hide those differences behind a misleading abstraction. Instead,
each attention path calls update() on the sub-cache it owns. ShramLayerCache acts as the
ownership, coordination, and reset/reorder boundary for one decoder layer.

Sequence length at this boundary is reported by delegating to the local sliding-window
sub-cache, which tracks the cumulative count of token positions processed. This is the
quantity HuggingFace generation reads through get_seq_length().
"""

import torch
from transformers.cache_utils import CacheLayerMixin

from ..configuration import ShramConfig
from .mosrah_cache import MoSRAHCache
from .router_cache import RouterCache
from .sliding_window_cache import LocalSlidingWindowLayerCache


class ShramLayerCache(CacheLayerMixin):
    """Cache subsystem for one SHRAM decoder layer.

    Owns and coordinates three sub-caches:
      - sliding_window_cache: LocalSlidingWindowLayerCache for the local sliding-window path.
      - mosrah_cache: MoSRAHCache for the MoSRAH sparse attention path.
      - router_cache: RouterCache for the block-balanced router's block state.

    Satisfies the HuggingFace per-layer cache role (CacheLayerMixin). The sub-caches are
    exposed directly for their downstream consumers — no composite update() interface is
    provided, because the paths have materially different update semantics.

    Sequence length is reported by delegating to the local sliding-window sub-cache, which
    tracks the cumulative count of token positions processed across all update() calls.

    Args:
        config: ShramConfig instance. All sub-cache dimensions and capacities are derived
            from config so that a single source of truth governs every buffer size.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate cache tensors.
    """

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        config: ShramConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._inference_sequence_length = config.inference_sequence_length
        self.sliding_window_cache = LocalSlidingWindowLayerCache(
            sliding_window=config.window_size,
            num_heads=config.num_sliding_window_heads,
            head_dim=config.head_dim,
            batch_size=batch_size,
            device=device,
        )
        self.mosrah_cache = MoSRAHCache(
            num_mosrah_heads=config.num_mosrah_heads,
            head_dim=config.head_dim,
            batch_size=batch_size,
            device=device,
            mosrah_cache_length=config.mosrah_cache_length,
        )
        self.router_cache = RouterCache(
            block_length=config.block_length,
            num_mosrah_heads=config.num_mosrah_heads,
            batch_size=batch_size,
            device=device,
        )

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True iff both sub-caches have allocated their storage.

        Both LocalSlidingWindowLayerCache and MoSRAHCache pre-allocate at construction,
        so this is True immediately after ShramLayerCache.__init__ returns.
        """
        return (
            self.sliding_window_cache.is_initialized
            and self.mosrah_cache.is_initialized
            and self.router_cache.is_initialized
        )

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

    def get_seq_length(self) -> int:  # type: ignore[override]
        """Return the cumulative sequence length from the local sliding-window path.

        The local path is authoritative for sequence progress: it sees every token
        presented to this layer and accumulates a truthful total. Delegates to
        sliding_window_cache.get_seq_length().
        """
        return self.sliding_window_cache.get_seq_length()

    def reset(self) -> None:
        """Clear both sub-caches.

        Delegates reset to each sub-cache. Both are cleared atomically so the sliding-window
        state and MoSRAH sparse state remain consistent.
        """
        self.sliding_window_cache.reset()
        self.mosrah_cache.reset()
        self.router_cache.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension of both sub-caches for beam search.

        Delegates to each sub-cache. Both are reordered atomically so the sliding-window
        and MoSRAH state correspond to the same beam hypotheses after reordering.

        Args:
            beam_idx: Permutation indices of shape (batch,) produced by beam search.
        """
        self.sliding_window_cache.reorder_cache(beam_idx)
        self.mosrah_cache.reorder_cache(beam_idx)
        self.router_cache.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension of both sub-caches for beam search initialisation.

        Delegates atomically to each sub-cache. Both must be expanded together so the
        sliding-window and MoSRAH state correspond to the same beam candidates.

        Args:
            repeats: Number of times to repeat each batch entry.
        """
        self.sliding_window_cache.batch_repeat_interleave(repeats)
        self.mosrah_cache.batch_repeat_interleave(repeats)
        self.router_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset of batch entries in both sub-caches for contrastive search.

        Delegates atomically to each sub-cache. Both must be trimmed together so the
        sliding-window and MoSRAH state remain consistent.

        Args:
            indices: 1-D integer tensor of batch indices to retain.
        """
        self.sliding_window_cache.batch_select_indices(indices)
        self.mosrah_cache.batch_select_indices(indices)
        self.router_cache.batch_select_indices(indices)

    def offload(self) -> None:
        """Offload both sub-caches to CPU.

        Delegates to each sub-cache's offload method. Does not call super() — ShramLayerCache
        does not own self.keys/self.values directly; all cached data lives in the sub-caches.
        """
        self.sliding_window_cache.offload()
        self.mosrah_cache.offload()
        self.router_cache.offload()

    def prefetch(self) -> None:
        """Move both sub-caches back to their model device ahead of time.

        Delegates to each sub-cache's prefetch method. Does not call super() — ShramLayerCache
        does not own self.keys/self.values directly; all cached data lives in the sub-caches.
        """
        self.sliding_window_cache.prefetch()
        self.mosrah_cache.prefetch()
        self.router_cache.prefetch()

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
        """Return the maximum sequence length this layer cache can serve.

        The authoritative upper bound is ``config.inference_sequence_length``, which
        governs the full accumulated token history the model is configured to handle.
        HuggingFace's static-cache machinery reads this value to determine whether the
        cache is compileable and to size generation loops.
        """
        return self._inference_sequence_length

    def get_mask_sizes(  # type: ignore[override]
        self,
        cache_position: torch.Tensor,
    ) -> tuple[int, int]:
        """Return the KV dimensions for HuggingFace causal mask construction.

        Returns (inference_sequence_length, 0): the full static cache capacity as
        kv_length and zero offset. HuggingFace reads these values to size the causal
        attention mask when is_compileable is True.
        """
        return self._inference_sequence_length, 0
