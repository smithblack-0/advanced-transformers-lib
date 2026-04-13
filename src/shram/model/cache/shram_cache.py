"""SHRAM top-level cache — model-wide owner for the full SHRAM decoder stack.

The HuggingFace Cache protocol expects a single top-level Cache object that owns one
CacheLayerMixin per decoder layer. The actual SHRAM caching responsibilities live one level
lower in ShramLayerCache — each of which owns a LocalSlidingWindowLayerCache and a MoSRAHCache.
ShramCache bridges those two levels: it constructs one ShramLayerCache per decoder layer,
presents them through the Cache interface, and transparently forwards model-wide operations
across all of them.

ShramCache does not define a composite update() interface. The two attention paths inside each
SHRAM layer have different update semantics, and neither the layer-level boundary (Unit 6.B)
nor the model-level boundary here can meaningfully unify them. Callers must reach down to the
relevant sub-cache directly. ShramCache's role is ownership, construction, and model-wide
coordination of the layer caches — not routing attention inputs.

No scalar sequence length is exposed at this boundary. Ragged masked continuation means
different batch items may carry different numbers of live tokens, making a single truthful
scalar unavailable. get_seq_length() raises NotImplementedError.
"""

import torch
from transformers.cache_utils import Cache

from .shram_layer_cache import ShramLayerCache


class ShramCache(Cache):
    """Top-level cache for the full SHRAM model.

    Owns one ShramLayerCache per decoder layer. Satisfies the HuggingFace top-level Cache
    role and transparently forwards reset, reorder, and sequence-length queries across all
    owned layer caches.

    No composite update() interface is provided. The two attention paths inside each SHRAM
    layer have materially different update semantics; callers must update sub-caches directly
    via cache.layers[layer_idx].sliding_window_cache or cache.layers[layer_idx].mosrah_cache.

    Args:
        num_hidden_layers: Number of SHRAM decoder layers. Determines how many
            ShramLayerCache objects are constructed.
        sliding_window: Token window size passed to each layer's LocalSlidingWindowLayerCache.
        num_local_heads: Number of local attention heads per layer.
        local_head_dim: Per-head embedding width for the local path.
        num_mosrah_heads: Total number of MoSRAH expert heads (L) per layer.
        mosrah_head_dim: Bottlenecked head embedding width (u) for the MoSRAH path.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate cache tensors.
        initial_buffer_size: Initial per-(batch, head) capacity for each MoSRAHCache.
            Doubled when any slot overflows. Defaults to 64 to avoid repeated reallocation
            during prompt processing.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        sliding_window: int,
        num_local_heads: int,
        local_head_dim: int,
        num_mosrah_heads: int,
        mosrah_head_dim: int,
        batch_size: int,
        device: torch.device,
        initial_buffer_size: int = 64,
    ) -> None:
        layers = [
            ShramLayerCache(
                sliding_window=sliding_window,
                num_local_heads=num_local_heads,
                local_head_dim=local_head_dim,
                num_mosrah_heads=num_mosrah_heads,
                mosrah_head_dim=mosrah_head_dim,
                batch_size=batch_size,
                device=device,
                initial_buffer_size=initial_buffer_size,
            )
            for _ in range(num_hidden_layers)
        ]
        super().__init__(layers=layers)

    # ---------------------------------------------------------------------------
    # Cache — composite-meaningful methods
    # ---------------------------------------------------------------------------
    #
    # reset(): Inherited. Iterates all layer caches and calls reset() on each.
    #
    # reorder_cache(beam_idx): Inherited. Iterates all layer caches and reorders each.
    #
    # is_initialized: Inherited property. True iff all layer caches are initialized.
    #   Since ShramLayerCache.is_initialized is True from construction, this is True
    #   immediately after ShramCache.__init__ returns.

    def get_seq_length(self, layer_idx: int = 0) -> int:  # type: ignore[override]
        """Not supported — no single scalar sequence length is available at this boundary.

        Ragged masked continuation means different batch items may carry different numbers
        of live tokens, making a truthful scalar unavailable at any cache boundary.
        """
        raise NotImplementedError(
            "ShramCache has no single scalar sequence length. "
            "Ragged masked continuation makes a truthful scalar unavailable."
        )

    # ---------------------------------------------------------------------------
    # Cache — unsupported methods
    # ---------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported — ShramCache has no composite update interface.

        The two attention paths inside each SHRAM layer have different update semantics.
        Callers must update sub-caches directly:
          cache.layers[layer_idx].sliding_window_cache.update(key_states, value_states)
          cache.layers[layer_idx].mosrah_cache.update(key_states, value_states, active_mask)
        """
        raise NotImplementedError(
            "ShramCache has no composite update interface. "
            "Update sliding_window_cache or mosrah_cache on the relevant layer directly."
        )

    def crop(self, max_length: int) -> None:
        """Not supported — ShramCache layers do not implement crop()."""
        raise NotImplementedError("ShramCache does not support crop().")

    @property
    def max_batch_size(self) -> int:
        """Not supported — ShramCache does not track a uniform batch size across layers."""
        raise NotImplementedError("ShramCache does not expose max_batch_size.")

    @property
    def max_cache_len(self) -> int:
        """Not supported — ShramCache has no single maximum cache length.

        The sliding-window side is bounded by sliding_window; the MoSRAH side is unbounded.
        No truthful scalar maximum represents the composite.
        """
        raise NotImplementedError("ShramCache does not expose max_cache_len.")
