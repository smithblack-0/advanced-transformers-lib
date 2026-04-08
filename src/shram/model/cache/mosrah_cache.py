"""MoSRAH sparse KV cache.

MoSRAH routes each token to K of L available expert heads, so its KV cache is indexed
by (layer_idx, head_idx) rather than by layer alone. The routing is dynamic and produces
a ragged distribution of token counts across (batch, head) slots — different batch items
may route different numbers of tokens to the same head, and different heads accumulate
at different rates. DynamicCache cannot represent this correctly: it concatenates along
the sequence dimension and assumes uniform token counts across the batch. MoSRAHCache
therefore uses a custom buffer design.

Per layer, keys and values are stored as (B, L, T_max, u) tensors, where B is batch
size, L is the number of expert heads (num_mosrah_heads), T_max is the current capacity
(doubled when any slot overflows), and u is the bottlenecked head embedding width
(head_dim). A (B, L) integer count tensor tracks the valid occupancy of each
(batch, head) slot. Everything beyond the count for a given slot is junk data; consumers
call get_expert_lengths() to obtain the counts and are responsible for masking.

All buffers are allocated at construction time. MoSRAHCache is constructed by ShramCache
inside _prepare_cache_for_generation, which has access to batch size, device, and all
model config parameters needed to fully specify the storage layout upfront.
"""

import torch
from transformers.cache_utils import Cache


class MoSRAHCache(Cache):
    """KV cache for the MoSRAH sparse attention path.

    Subclasses transformers.cache_utils.Cache to satisfy the HF Cache protocol.
    Stores keys and values per (layer_idx, head_idx) using a custom buffer design
    rather than delegating to DynamicCache, which cannot represent MoSRAH's ragged
    per-(batch, head) token counts correctly.

    All storage is allocated at construction time. The caller (ShramCache) provides
    batch size, device, and model config parameters so no lazy allocation is needed.

    At each generation step, only the K heads selected by the router are updated.
    Different batch items may select different heads, and different heads accumulate
    at different rates. The custom buffer tracks this independently per (batch, head)
    slot via explicit count tensors.

    Args:
        num_hidden_layers: Number of decoder layers. Determines the number of
            per-layer buffer sets allocated at construction.
        num_mosrah_heads: Total number of MoSRAH expert heads (L). Determines the
            second dimension of all per-layer storage tensors.
        head_dim: Bottlenecked head embedding width (u). Determines the fourth
            dimension of all per-layer storage tensors.
        batch_size: Number of sequences in the batch. Determines the first dimension
            of all per-layer storage tensors.
        device: Device on which to allocate all tensors. Should match the model device.
        initial_t_max: Initial sequence capacity per (batch, head) slot. Doubled when
            any slot overflows. Defaults to 64 to avoid repeated reallocation during
            prompt processing.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        num_mosrah_heads: int,
        head_dim: int,
        batch_size: int,
        device: torch.device,
        initial_t_max: int = 64,
    ) -> None:
        # Cache.__init__ requires layers or layer_class_to_replicate. We pass an
        # empty list — self.layers is unused since all storage is in the lists below.
        super().__init__(layers=[])
        self.num_hidden_layers = num_hidden_layers
        self.num_mosrah_heads = num_mosrah_heads
        self.head_dim = head_dim
        self.batch_size = batch_size

        # Allocate all per-layer buffers upfront. Each layer gets independent
        # (B, L, T_max, u) key and value tensors and a (B, L) count tensor.
        self._t_max: list[int] = [initial_t_max] * num_hidden_layers
        self._keys: list[torch.Tensor] = [
            torch.zeros(batch_size, num_mosrah_heads, initial_t_max, head_dim, device=device)
            for _ in range(num_hidden_layers)
        ]
        self._values: list[torch.Tensor] = [
            torch.zeros(batch_size, num_mosrah_heads, initial_t_max, head_dim, device=device)
            for _ in range(num_hidden_layers)
        ]
        self._counts: list[torch.Tensor] = [
            torch.zeros(batch_size, num_mosrah_heads, dtype=torch.long, device=device)
            for _ in range(num_hidden_layers)
        ]

    # ---------------------------------------------------------------------------
    # Primary API
    # ---------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        layer_idx: int,
        head_idx: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter key/value states into the per-head buffer for the given layer.

        Not implemented in Unit 6.A.A — scatter logic is Unit 6.A.B's responsibility.
        """
        raise NotImplementedError(
            "MoSRAHCache.update() scatter is implemented in Unit 6.A.B."
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Not supported — no single sequence length represents this cache's state.

        MoSRAH heads accumulate independently; (batch, head) slots have different
        lengths depending on routing history. There is no meaningful scalar summary.
        """
        raise NotImplementedError(
            "MoSRAHCache has no single sequence length. "
            "Use get_expert_lengths(layer_idx) for per-head occupancy."
        )

    def get_expert_lengths(self, layer_idx: int) -> torch.Tensor:
        """Return the per-(batch, head) token count for the given layer.

        This is the authoritative occupancy tensor consumed by BEA for attention
        masking and by the position decoder (Unit 6.D) for semantic-sequence
        position computation.

        Args:
            layer_idx: Decoder layer index (0-based).

        Returns:
            Integer tensor of shape (B, L) where entry [b, h] is the number of
            valid tokens stored in the (b, h) slot for this layer. Zero for slots
            that have not yet received any updates.
        """
        return self._counts[layer_idx]

    # ---------------------------------------------------------------------------
    # Cache protocol — coordination methods
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cached key and value tensors.

        Zeroes the count tensors and reinitialises the key/value buffers in place.
        After reset, get_expert_lengths() returns all zeros for every layer and
        subsequent updates start from position 0.
        """
        for layer_idx in range(self.num_hidden_layers):
            self._counts[layer_idx].zero_()
            self._keys[layer_idx].zero_()
            self._values[layer_idx].zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch dimension of all cached tensors for beam search.

        Applied atomically across both the key/value buffers and the count tensors.
        Beam search must reorder both together or the occupancy counts and buffer
        contents will correspond to different beam hypotheses.

        Args:
            beam_idx: Permutation indices of shape (batch,) produced by the beam
                search algorithm.
        """
        for layer_idx in range(self.num_hidden_layers):
            self._keys[layer_idx] = self._keys[layer_idx][beam_idx]
            self._values[layer_idx] = self._values[layer_idx][beam_idx]
            self._counts[layer_idx] = self._counts[layer_idx][beam_idx]

    # ---------------------------------------------------------------------------
    # Cache protocol — properties
    # ---------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True once any (batch, head) slot across any layer has been written to."""
        return any(self._counts[i].any() for i in range(self.num_hidden_layers))

    @property
    def is_compileable(self) -> bool:
        """Not compileable — dynamic buffer expansion is not torch.compile-safe."""
        return False

    @property
    def is_sliding(self) -> list[bool]:
        """MoSRAH cache is not a sliding-window cache — all slots are full-history."""
        return [False] * self.num_hidden_layers

    # ---------------------------------------------------------------------------
    # Cache protocol — unsupported operations
    # ---------------------------------------------------------------------------

    def get_max_cache_shape(self, layer_idx: int | None = None) -> int:
        """Not supported — MoSRAHCache is dynamic and unbounded."""
        raise NotImplementedError(
            "MoSRAHCache is unbounded; get_max_cache_shape() is not supported."
        )

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int | None = None,
    ) -> tuple[int, int]:
        """Not supported — MoSRAHCache does not participate in HF mask construction."""
        raise NotImplementedError(
            "MoSRAHCache does not support get_mask_sizes()."
        )
