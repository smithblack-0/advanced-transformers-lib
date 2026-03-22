"""Rotary Position Embeddings (RoPE).

RoPE encodes position in the *relationship* between query and key vectors rather than
adding it to the inputs directly. When the attention dot product Q·Kᵀ is computed, the
per-position rotations cancel to produce a score that depends only on the relative
distance between positions — not on their absolute values. This is what gives RoPE
better length generalisation than absolute learned embeddings.

Each pair of head dimensions (d, d+1) is assigned a rotation frequency
    1 / theta^(2d / head_dim)
Higher theta → slower rotation per position → position encodings remain distinguishable
further apart before wrapping around. Llama 3 uses theta=500,000 as a prerequisite for
supporting 128K context.

HuggingFace's ROPE_INIT_FUNCTIONS handles inv_freq computation for all scaled RoPE
variants (linear, dynamic, yarn, longrope, llama3). The unscaled 'default' case is not
in that registry and is computed here directly.
"""

import torch
import torch.nn as nn

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Rope types whose inverse frequencies may change at runtime based on sequence length.
# All other types compute inv_freq once at initialisation and leave it fixed.
_DYNAMIC_ROPE_TYPES = {"dynamic", "longrope"}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the 90° rotation used in the RoPE update formula.

    Splits the last dimension into two halves [x1, x2] and returns [-x2, x1].
    Combined with the standard formula ``x * cos + rotate_half(x) * sin``, this
    implements a 2D rotation on each consecutive pair of dimensions.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings as an nn.Module.

    Computes position-dependent rotation frequencies from the model config, maintains a
    lazily-extended cos/sin cache, and applies the rotations to query and key tensors.

    Supports all HuggingFace rope_type values via ROPE_INIT_FUNCTIONS: linear, dynamic,
    yarn, longrope, and llama3. The unscaled default case is handled directly.

    The cos/sin cache grows automatically at runtime when a sequence longer than the
    current cache is seen. ``config.max_position_embeddings`` records the training
    context length (used by HF's scaling computations) but does not cap inference length.

    For dynamic and longrope types, inverse frequencies may be recomputed in the forward
    pass when the sequence length crosses the relevant threshold. The cache is invalidated
    whenever inv_freq changes so that rotations remain consistent.

    Args:
        config: Model config. Must expose ``rope_theta``, ``rope_parameters`` (set by
            HF's RotaryEmbeddingConfigMixin), and ``head_dim`` (or ``hidden_size`` and
            ``num_attention_heads`` to derive it).
        device: Optional device for initial buffer placement. Buffers move automatically
            when the model is transferred via ``.to()`` or ``.cuda()``.
    """

    def __init__(self, config, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config

        # rope_parameters is None when no rope_scaling was set in the config.
        rope_params = config.rope_parameters
        self.rope_type = (
            rope_params.get("rope_type", "default") if rope_params is not None else "default"
        )

        inv_freq, self.attention_scaling = self._init_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Retained for dynamic types that reset inv_freq when the sequence drops back
        # below the original training length.
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        # cos/sin cache: built lazily in _extend_cache and registered as buffers.
        # None signals that the cache has not yet been built or has been invalidated.
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

        # Tracks the sequence length that was current when inv_freq was last computed,
        # used to detect when dynamic/longrope types need a recomputation.
        self._inv_freq_seq_len: int = 0

    def _init_inv_freq(self, device: torch.device | None) -> tuple[torch.Tensor, float]:
        """Compute the initial inverse frequencies.

        The 'default' case (standard, unscaled RoPE) is not in ROPE_INIT_FUNCTIONS and
        is computed here. All other types are delegated to HF's registry.

        Returns:
            Tuple of (inv_freq tensor, attention_scaling float). attention_scaling is
            1.0 for most types; YaRN and longrope use values > 1.0 to correct attention
            magnitude after frequency manipulation.
        """
        if self.rope_type == "default":
            head_dim = getattr(self.config, "head_dim", None) or (
                self.config.hidden_size // self.config.num_attention_heads
            )
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                    / head_dim
                )
            )
            return inv_freq, 1.0

        return ROPE_INIT_FUNCTIONS[self.rope_type](self.config, device)

    def _extend_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Build the cos/sin table to cover positions [0, seq_len).

        Registered as buffers so they automatically move with the model on .to() calls.
        Called whenever the current cache is too short or has been invalidated by an
        inv_freq recomputation.
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # (seq_len, head_dim // 2) → duplicate → (seq_len, head_dim)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype), persistent=False)

    def _maybe_update_inv_freq(self, seq_len: int, device: torch.device) -> bool:
        """Recompute inv_freq for dynamic rope types if the sequence length warrants it.

        Dynamic NTK scaling adjusts the base frequency when seq_len exceeds the
        original training length. LongRoPE switches between long and short frequency
        tables based on the same threshold. For all other types this is a no-op.

        Returns:
            True if inv_freq was updated (cache must be rebuilt), False otherwise.
        """
        if self.rope_type not in _DYNAMIC_ROPE_TYPES:
            return False

        original_max = self.config.max_position_embeddings

        if "dynamic" in self.rope_type:
            # Recompute when growing beyond the original length; reset when dropping
            # back below it.
            if seq_len > self._inv_freq_seq_len and seq_len > original_max:
                inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](
                    self.config, device, seq_len=seq_len
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)
                self._inv_freq_seq_len = seq_len
                return True
            if seq_len <= original_max and self._inv_freq_seq_len > original_max:
                self.register_buffer(
                    "inv_freq", self.original_inv_freq.to(device), persistent=False
                )
                self._inv_freq_seq_len = original_max
                return True

        elif self.rope_type == "longrope":
            # longrope uses a different frequency table above/below original_max.
            # Recompute whenever the sequence crosses that boundary.
            prev_was_long = self._inv_freq_seq_len > original_max
            now_is_long = seq_len > original_max
            if prev_was_long != now_is_long or self._inv_freq_seq_len == 0:
                inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS["longrope"](
                    self.config, device, seq_len=seq_len
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)
                self._inv_freq_seq_len = seq_len
                return True

        return False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply rotary embeddings to query and key tensors.

        Extends the cos/sin cache lazily when position_ids reference positions beyond
        its current length. For dynamic and longrope types, inv_freq is recomputed
        when the sequence crosses the relevant threshold.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            position_ids: Integer positions of shape (batch, seq_len).

        Returns:
            Tuple of (q_rotated, k_rotated, attention_scaling). attention_scaling is
            1.0 for most rope types; callers must multiply attention logits by it for
            types like YaRN that adjust frequency magnitude.
        """
        seq_len = int(position_ids.max().item()) + 1

        # Update inv_freq for dynamic types; invalidate cache if it changed.
        inv_freq_changed = self._maybe_update_inv_freq(seq_len, device=q.device)
        if inv_freq_changed:
            self._cos_cached = None
            self._sin_cached = None

        # Extend cache if it has not been built or is too short for the current sequence.
        if self._cos_cached is None or seq_len > self._cos_cached.shape[0]:
            self._extend_cache(seq_len, device=q.device, dtype=q.dtype)

        # Index cos/sin for the given absolute positions → (batch, seq_len, head_dim).
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]

        # Unsqueeze the head dimension so cos/sin broadcast over all heads.
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)

        q_rotated = q * cos + _rotate_half(q) * sin
        k_rotated = k * cos + _rotate_half(k) * sin

        return q_rotated, k_rotated, self.attention_scaling
