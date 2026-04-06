"""Rotary Position Embeddings (RoPE).

RoPE encodes position in the *relationship* between query and key vectors rather than
adding it to the inputs directly. When the attention dot product Q·Kᵀ is computed, the
per-position rotations cancel to produce a score that depends only on the relative
distance between positions — not on their absolute values. This is what gives RoPE
better length generalisation than absolute learned embeddings.

Each pair of head dimensions (d, d+1) is assigned a rotation frequency
    1 / theta^(2d / head_dim)
Higher theta → slower rotation per position → position encodings remain distinguishable
further apart before wrapping. Llama 3 uses theta=500,000 as a prerequisite for
128K context support.

Supported rope types: "default" (standard unscaled RoPE), "linear", and "yarn".
HuggingFace's ROPE_INIT_FUNCTIONS handles inv_freq computation for linear and yarn;
the default case is not in that registry and is computed directly here.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

_SUPPORTED_ROPE_TYPES = {"default", "linear", "yarn"}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the 90° rotation used in the RoPE update formula.

    Splits the last dimension into two halves [x1, x2] and returns [-x2, x1].
    Combined with ``x * cos + rotate_half(x) * sin``, this implements a 2D rotation
    on each consecutive pair of dimensions.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings as an nn.Module.

    Computes position-dependent rotation frequencies from the model config, maintains
    a lazily-extended cos/sin cache, and applies the rotations to query and key tensors.

    The cos/sin cache grows automatically at runtime when a sequence longer than the
    current cache is encountered. ``config.max_position_embeddings`` records the
    training context length (required by HF's scaling computations) but does not cap
    inference length.

    Args:
        config: Model config. Must expose ``rope_theta``, ``rope_parameters`` (set by
            HF's RotaryEmbeddingConfigMixin), and ``head_dim``.
        device: Optional device for initial buffer placement. Buffers move with the
            model on ``.to()`` / ``.cuda()`` calls.

    Raises:
        NotImplementedError: If ``config.rope_parameters`` specifies an unsupported
            rope type. Supported types: "default", "linear", "yarn".
    """

    def __init__(self, config: PretrainedConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config

        # rope_parameters is None when no rope_scaling was passed to the config.
        rope_params = config.rope_parameters
        self.rope_type = (
            rope_params.get("rope_type", "default") if rope_params is not None else "default"
        )

        if self.rope_type not in _SUPPORTED_ROPE_TYPES:
            raise NotImplementedError(
                f"rope_type '{self.rope_type}' is not supported. "
                f"Supported types: {sorted(_SUPPORTED_ROPE_TYPES)}"
            )

        if self.rope_type == "default":
            # Standard RoPE: inv_freq = 1 / theta^(2i / head_dim).
            # Not in ROPE_INIT_FUNCTIONS, so computed directly.
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim)
            )
            self.attention_scaling: float = 1.0
        else:
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Initialised as None; built on first forward call and extended lazily thereafter.
        # Registered as buffers so they move with the model across devices.
        self.register_buffer("_cos_cached", None, persistent=False)
        self.register_buffer("_sin_cached", None, persistent=False)

    def _extend_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build the cos/sin table to cover positions [0, seq_len).

        Registered as buffers so subsequent calls to ``.to()`` / ``.cuda()`` will
        move them to the correct device. Rebuilds whenever the sequence grows or
        the dtype changes (e.g. switching between fp32 and bf16).
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # outer product → (seq_len, head_dim // 2); duplicate → (seq_len, head_dim)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply rotary embeddings to query and key tensors.

        The cos/sin cache is extended lazily when position_ids reference positions
        beyond its current length.

        position_ids may be any integer tensor whose values are valid position indices.
        Its shape must match the non-head, non-head_dim dimensions of q and k:

        - Standard causal attention: position_ids (B, N), q (B, H, N, head_dim).
        - BEA packed attention:      position_ids (B, L, T), q (B, L, T, head_dim).

        Head dimensions sit between the batch dimension and the position dimensions in
        q/k. They are absent from position_ids and are handled by inserting broadcast
        dimensions automatically.

        Args:
            q: Query tensor of shape (batch, [num_heads,] *pos_dims, head_dim).
            k: Key tensor of shape (batch, [num_kv_heads,] *pos_dims, head_dim).
            position_ids: Integer positions of shape (batch, *pos_dims).

        Returns:
            Tuple of (q_rotated, k_rotated, attention_scaling). attention_scaling is
            1.0 for default and linear; YaRN returns a value != 1.0 that callers must
            apply to attention logits to correct for frequency magnitude changes.
        """
        seq_len = int(position_ids.max().item()) + 1

        if self._cos_cached is None or seq_len > self._cos_cached.shape[0] or self._cos_cached.dtype != q.dtype:
            self._extend_cache(seq_len, device=q.device, dtype=q.dtype)

        # Direct index gather: works for any position_ids shape.
        # cos/sin shape is (*position_ids.shape, head_dim).
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]

        # q/k may have head dimensions between the batch dimension and the position
        # dimensions that are absent from position_ids. Insert one broadcast dimension
        # at dim 1 per missing head dimension so that cos/sin align with q/k.
        # Standard case: position_ids (B, N) → cos (B, N, D); q (B, H, N, D) needs
        # cos (B, 1, N, D) — one unsqueeze. BEA case: position_ids (B, L, T) → cos
        # (B, L, T, D); q (B, L, T, D) — no unsqueeze needed.
        for _ in range(q.ndim - cos.ndim):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_rotated = q * cos + _rotate_half(q) * sin
        k_rotated = k * cos + _rotate_half(k) * sin

        return q_rotated, k_rotated, self.attention_scaling
