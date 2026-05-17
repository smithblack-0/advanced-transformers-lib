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

Sequence length is reported by delegating to the local sliding-window sub-cache of the
specified layer, which tracks the cumulative count of token positions processed. This is
what HuggingFace generation reads through get_seq_length().
"""

import torch
from transformers.cache_utils import Cache

from ..configuration import ShramConfig
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
        config: ShramConfig instance. All layer counts, buffer sizes, and sub-cache
            dimensions are derived from config so that a single source of truth governs
            every buffer size across the full cache stack.
        batch_size: Number of sequences in the batch.
        device: Device on which to allocate cache tensors.
    """

    is_compileable = True

    def __init__(
        self,
        config: ShramConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        layers = [
            ShramLayerCache(
                config=config,
                batch_size=batch_size,
                device=device,
            )
            for _ in range(config.num_decoder_layers)
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
        """Return the cumulative sequence length for the specified layer.

        Delegates to the layer cache at layer_idx, which in turn delegates to the
        local sliding-window sub-cache. That sub-cache is authoritative for sequence
        progress: it sees every token presented to the layer and accumulates a truthful
        total count. Defaults to layer 0, which is sufficient for HuggingFace generation.
        """
        return self.layers[layer_idx].get_seq_length()

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
        """Return the maximum sequence length the cache can serve.

        Delegates to layers[0].get_max_cache_shape(), which returns
        config.inference_sequence_length. HuggingFace's static-cache machinery reads
        this value to size generation loops and verify compileable cache contracts.
        """
        return self.layers[0].get_max_cache_shape()
