"""Unvectorized reference implementation of the MoSRAH sparse KV cache.

This module exists solely as a correctness oracle. SlowMoSRAHCache implements the same
interface and storage layout as MoSRAHCache but uses an explicit Python loop over
(batch, token, head) triples in update(). The loop is obviously correct by inspection:
each token's key and value are written to the next available slot for that (batch, head)
pair, in the order tokens appear along the sequence dimension, which directly enforces
causal ordering without any index arithmetic to verify.

SlowMoSRAHCache is never instantiated in the model path. Its role is to provide a
trusted ground truth against which the vectorized MoSRAHCache.update() is validated in
Unit 6.A.B tests, and as a reference for the Unit 6.D position decoder. Because the
vectorized implementation is validated by asserting exact agreement with this one on all
test inputs, the correctness of SlowMoSRAHCache is load-bearing: its own test suite
(test_slow_mosrah_cache.py) must establish it is trustworthy before it can be used as
an oracle.
"""

import torch
from transformers.cache_utils import Cache


class SlowMoSRAHCache(Cache):
    """Unvectorized reference implementation of the MoSRAH KV cache.

    Identical storage layout to MoSRAHCache: per-layer (B, L, T_max, u) key/value
    buffers and (B, L) count tensors, with the same constructor signature and the same
    HF Cache protocol methods. The sole difference is update(), which uses an explicit
    Python loop over (b, n, k) triples rather than vectorized index arithmetic.

    This class is not used in the model path. It exists so that MoSRAHCache.update()
    can be validated by asserting exact agreement with this implementation on all test
    inputs. See module docstring for the trust chain this enables.

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
        """Scatter key/value states into the per-head buffer using an explicit loop.

        Iterates over every (batch, token, head-choice) triple in sequence order.
        For each triple, the key and value are written to the next available slot for
        that (batch, head) pair and the count is incremented. Causal ordering is
        guaranteed because the token dimension is traversed from 0 to N-1 and counts
        are updated immediately after each write.

        Buffer expansion (doubling T_max) is triggered before any writes if the
        incoming tokens would cause any slot to overflow the current capacity.

        Args:
            layer_idx: Decoder layer index (0-based).
            head_idx: Shape (B, N, K) — for each batch item and token, the K head
                indices selected by the router.
            key_states: Shape (B, N, K, u) — key vectors for each token-head pair.
            value_states: Shape (B, N, K, u) — value vectors for each token-head pair.

        Returns:
            Tuple of (keys, values), each the full (B, L, T_max, u) buffer for this
            layer after the update.
        """
        B, N, K = head_idx.shape
        counts = self._counts[layer_idx]

        # Compute how many tokens each (batch, head) slot will receive this call so
        # we can check for overflow before writing anything.
        delta = torch.zeros(
            B, self.num_mosrah_heads, dtype=torch.long, device=head_idx.device
        )
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    delta[b, head_idx[b, n, k].item()] += 1

        # Expand buffers if any slot would overflow.
        if (counts + delta).max().item() > self._t_max[layer_idx]:
            self._expand_layer(layer_idx)

        # Write each token into the next available slot for its (batch, head) pair.
        # Iterating n from 0 to N-1 preserves causal ordering within each slot.
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    h = head_idx[b, n, k].item()
                    pos = counts[b, h].item()
                    self._keys[layer_idx][b, h, pos, :] = key_states[b, n, k, :]
                    self._values[layer_idx][b, h, pos, :] = value_states[b, n, k, :]
                    counts[b, h] += 1

        return self._keys[layer_idx], self._values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Not supported — no single sequence length represents this cache's state.

        MoSRAH heads accumulate independently; (batch, head) slots have different
        lengths depending on routing history. There is no meaningful scalar summary.
        """
        raise NotImplementedError(
            "SlowMoSRAHCache has no single sequence length. "
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
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _expand_layer(self, layer_idx: int) -> None:
        """Double the T_max capacity for the given layer, preserving existing data.

        Called by update() when an incoming batch of tokens would cause any
        (batch, head) slot to exceed the current buffer capacity. All existing
        key and value data is copied into the low half of the new buffer; the
        high half is zero-initialised and will be filled by subsequent writes.
        """
        old_t = self._t_max[layer_idx]
        new_t = old_t * 2
        dev = self._keys[layer_idx].device
        new_keys = torch.zeros(
            self.batch_size, self.num_mosrah_heads, new_t, self.head_dim, device=dev
        )
        new_values = torch.zeros(
            self.batch_size, self.num_mosrah_heads, new_t, self.head_dim, device=dev
        )
        new_keys[:, :, :old_t, :] = self._keys[layer_idx]
        new_values[:, :, :old_t, :] = self._values[layer_idx]
        self._keys[layer_idx] = new_keys
        self._values[layer_idx] = new_values
        self._t_max[layer_idx] = new_t

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
        """SlowMoSRAHCache is not a sliding-window cache — all slots are full-history."""
        return [False] * self.num_hidden_layers

    # ---------------------------------------------------------------------------
    # Cache protocol — unsupported operations
    # ---------------------------------------------------------------------------

    def get_max_cache_shape(self, layer_idx: int | None = None) -> int:
        """Not supported — MoSRAHCache is dynamic and unbounded."""
        raise NotImplementedError(
            "SlowMoSRAHCache is unbounded; get_max_cache_shape() is not supported."
        )

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int | None = None,
    ) -> tuple[int, int]:
        """Not supported — MoSRAHCache does not participate in HF mask construction."""
        raise NotImplementedError(
            "SlowMoSRAHCache does not support get_mask_sizes()."
        )
