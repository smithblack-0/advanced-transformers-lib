"""Local sliding-window attention path for one SHRAM decoder layer.

This module implements h_l, the short-range causal sliding-window attention path
inside one SHRAM decoder layer. It is standard multi-head attention (MHA), not
GQA: each local query head has its own key and value head.

Attention is executed with PyTorch FlexAttention using a native block-sparse
mask for causal sliding-window behavior. A manually materialized dense
sliding-window boolean attention mask is not used.

Caching is handled directly through HuggingFace's DynamicSlidingWindowLayer.
This module owns attention computation only; the cache owns key/value
accumulation and sliding retention semantics.
"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.cache_utils import DynamicSlidingWindowLayer

from ..configuration import ShramConfig
from ..rope import RotaryEmbedding


class SlidingWindowAttention(nn.Module):
    """Causal local sliding-window attention for one SHRAM decoder layer.

    The local path h_l preserves short-range autoregressive structure while the
    MoSRAH path handles long-range sparse routed behavior. This module is the
    local path only.

    Architectural properties:
      - standard MHA over `num_sliding_window_heads`
      - no projection bias
      - RoPE always constructed in `default` mode with `local_rope_theta`
      - causal sliding-window masking expressed natively via FlexAttention
      - optional HuggingFace DynamicSlidingWindowLayer cache support
      - returns output in `(B, N, hidden_size)` form for SHRAM composition

    Args:
        config: SHRAM config. Must expose `hidden_size`,
            `num_sliding_window_heads`, `head_dim`, `window_size`,
            `attention_dropout`, and `local_rope_theta`.

    Raises:
        NotImplementedError: If `attention_dropout != 0.0`. The paper setting is
            zero dropout, and this unit uses FlexAttention without a separate
            dropout path.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_sliding_window_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.attention_dropout = config.attention_dropout

        if self.attention_dropout != 0.0:
            raise NotImplementedError(
                "SlidingWindowAttention currently supports only attention_dropout == 0.0. "
                "This matches the paper/runtime setting for SHRAM."
            )

        self.inner_dim = self.num_heads * self.head_dim

        # Standard MHA projections for the local path.
        self.q_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        # Local path RoPE is always default-mode and never responds to YaRN fields.
        self.rope = RotaryEmbedding(
            mode="default",
            head_dim=self.head_dim,
            theta=config.local_rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DynamicSlidingWindowLayer | None = None,
    ) -> torch.Tensor:
        """Apply local causal sliding-window attention.

        Args:
            x: Input tensor of shape `(B, N, hidden_size)`.
            position_ids: Position tensor of shape `(B, N)`, consumed directly by
                the local-path RoPE implementation.
            cache: Optional `DynamicSlidingWindowLayer`. When provided, newly
                projected local keys and values are appended to the cache, and the
                full currently visible sliding-window K/V tensors are used for
                attention.

        Returns:
            Output tensor of shape `(B, N, hidden_size)`.

        Raises:
            ValueError: If the input tensors violate the module's basic shape
                contract.
        """
        batch_size, query_len, _ = x.shape
        self._validate_tensor_shape(x)
        self._validate_position_shape(x, position_ids)

        # (B, N, H*D) -> (B, H, N, D)
        q = self.q_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, attention_scaling = self.rope(q, k, position_ids)

        if cache is not None:
            k_full, v_full = cache.update(k, v)
        else:
            k_full, v_full = k, v

        kv_len = k_full.shape[-2]
        block_mask = self._make_block_mask(
            batch_size=batch_size,
            num_heads=self.num_heads,
            query_len=query_len,
            kv_len=kv_len,
            window_size=self.window_size,
            device_str=str(x.device),
        )

        attn_output = flex_attention(
            q,
            k_full,
            v_full,
            block_mask=block_mask,
            scale=attention_scaling / math.sqrt(self.head_dim),
        )

        # (B, H, N, D) -> (B, N, H*D) -> (B, N, hidden_size)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.inner_dim)
        )
        return self.o_proj(attn_output)

    def _validate_tensor_shape(self, x: torch.Tensor) -> None:
        """Validate the local tensor-shape contract required by this module."""
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got last dim {x.shape[-1]}."
            )

    def _validate_position_shape(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Validate the position-shape contract required by this module."""
        if position_ids.shape != x.shape[:2]:
            raise ValueError(
                f"position_ids must have shape {tuple(x.shape[:2])}, "
                f"got {tuple(position_ids.shape)}."
            )

    @staticmethod
    @lru_cache(maxsize=8)
    def _make_block_mask(
        batch_size: int,
        num_heads: int,
        query_len: int,
        kv_len: int,
        window_size: int,
        device_str: str,
    ) -> Any:
        """Create a cached FlexAttention block mask for causal local attention.

        The visible KV tensor always ends at the current query chunk. Therefore the
        query indices occupy the tail of the visible KV range. If the query chunk
        has length Q and the visible KV length is K, then query index `i` has
        absolute position `(K - Q) + i` within the visible KV frame, while key
        index `j` has position `j`.

        A key may be attended iff:
          1. it is not from the future: `j <= (K - Q) + i`
          2. it lies within the local window:
             `((K - Q) + i) - j < window_size`
        """
        query_offset = kv_len - query_len
        device = torch.device(device_str)

        def sliding_window_mask(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del batch_idx, head_idx
            q_abs = query_offset + q_idx
            is_causal = kv_idx <= q_abs
            in_window = (q_abs - kv_idx) < window_size
            return is_causal & in_window

        return create_block_mask(
            sliding_window_mask,
            B=batch_size,
            H=num_heads,
            Q_LEN=query_len,
            KV_LEN=kv_len,
            device=device,
        )