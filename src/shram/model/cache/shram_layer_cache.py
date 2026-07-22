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
        """True iff all owned sub-caches have allocated their storage."""
        return (
            self.sliding_window_cache.is_initialized
            and self.mosrah_cache.is_initialized
            and self.router_cache.is_initialized
        )

    @is_initialized.setter
    def is_initialized(self, value: bool) -> None:
        # CacheLayerMixin.__init__ assigns self.is_initialized = False as an instance
        # attribute. State is derived from the owned sub-caches, not stored here.
        pass

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — composite-meaningful methods
    # ---------------------------------------------------------------------------

    def get_seq_length(self) -> int:  # type: ignore[override]
        """Return cumulative sequence progress from the local cache owner."""
        return self.sliding_window_cache.get_seq_length()

    def get_max_length(self) -> int:
        """Return the configured maximum inference sequence length."""
        return self._inference_sequence_length

    def get_max_cache_shape(self) -> int:  # type: ignore[override]
        """Compatibility alias for the deprecated cache-shape interface."""
        return self.get_max_length()

    def reset(self) -> None:
        """Clear all owned sub-caches atomically."""
        self.sliding_window_cache.reset()
        self.mosrah_cache.reset()
        self.router_cache.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension of all sub-caches for beam search."""
        self.sliding_window_cache.reorder_cache(beam_idx)
        self.mosrah_cache.reorder_cache(beam_idx)
        self.router_cache.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Expand the batch dimension of all sub-caches together."""
        self.sliding_window_cache.batch_repeat_interleave(repeats)
        self.mosrah_cache.batch_repeat_interleave(repeats)
        self.router_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select matching batch entries in every owned sub-cache."""
        self.sliding_window_cache.batch_select_indices(indices)
        self.mosrah_cache.batch_select_indices(indices)
        self.router_cache.batch_select_indices(indices)

    def offload(self) -> None:
        """Offload all owned sub-caches to CPU."""
        self.sliding_window_cache.offload()
        self.mosrah_cache.offload()
        self.router_cache.offload()

    def prefetch(self) -> None:
        """Move all owned sub-caches back to their model device."""
        self.sliding_window_cache.prefetch()
        self.mosrah_cache.prefetch()
        self.router_cache.prefetch()

    def lazy_initialization(  # type: ignore[override]
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """No-op — every sub-cache handles its own initialization."""
        pass

    # ---------------------------------------------------------------------------
    # CacheLayerMixin — unsupported composite methods
    # ---------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported — ShramLayerCache has no composite update interface."""
        raise NotImplementedError(
            "ShramLayerCache has no composite update interface. "
            "Update sliding_window_cache or mosrah_cache directly."
        )

    def get_mask_sizes(  # type: ignore[override]
        self,
        query_length: int,
    ) -> tuple[int, int]:
        """Return full static KV dimensions for Hugging Face mask construction."""
        return self._inference_sequence_length, 0
