# src/shram/model/attention/sliding_window_attention.py

"""Local sliding-window attention path for SHRAM.

This file defines `SlidingWindowAttention`, the local short-range attention path
used inside the SHRAM hybrid layer.

In the masked-continuation variant, the local cache no longer returns a
semantically dense visible frame. Instead, `LocalSlidingWindowLayerCache`
returns:

- the retained local window memory concatenated with the current chunk
- an aligned active mask over that returned frame

This module consumes that returned frame directly and constructs effective local
causal/window visibility from the mask. It does not own cache retention policy;
it owns only local attention semantics.
"""

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ..cache.sliding_window_cache import LocalSlidingWindowLayerCache
from ..configuration import ShramConfig
from ..rope import RotaryEmbedding


class SlidingWindowAttention(nn.Module):
    """Causal local sliding-window attention for one SHRAM layer.

    Args:
        config: SHRAM config. Must expose `hidden_size`,
            `num_sliding_window_heads`, `head_dim`, `window_size`,
            `attention_dropout`, and `local_rope_theta`.

    Raises:
        NotImplementedError: If `attention_dropout != 0.0`.
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
                "SlidingWindowAttention currently supports only "
                "attention_dropout == 0.0."
            )

        self.inner_dim = self.num_heads * self.head_dim

        # Standard MHA projections for the local path.
        self.q_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        # The local path always uses default-mode RoPE with its own theta.
        self.rope = RotaryEmbedding(
            mode="default",
            head_dim=self.head_dim,
            theta=config.local_rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: LocalSlidingWindowLayerCache | None = None,
    ) -> torch.Tensor:
        """Apply local causal sliding-window attention.

        Args:
            x: Input tensor of shape `(B, N, hidden_size)`.
            position_ids: Position tensor of shape `(B, N)`.
            active_mask: Current-chunk active mask of shape `(B, N)`, where
                `True` means active.
            cache: Optional `LocalSlidingWindowLayerCache`.

        Returns:
            Output tensor of shape `(B, N, hidden_size)`.
        """
        batch_size, query_len, _ = x.shape

        self._validate_position_shape(x, position_ids)
        self._validate_active_mask_shape(x, active_mask)

        # (B, N, H*D) -> (B, H, N, D)
        q = self.q_proj(x).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k, attention_scaling = self.rope(q, k, position_ids)

        # The cache returns the current-step visible local frame, not merely the
        # retained next-step cache buffer.
        if cache is not None:
            k_full, v_full, full_active_mask = cache.update(k, v, active_mask)
        else:
            k_full, v_full, full_active_mask = k, v, active_mask

        block_mask = self._make_block_mask(
            active_mask=full_active_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            query_len=query_len,
            kv_len=k_full.shape[-2],
            window_size=self.window_size,
            device=x.device,
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

    def _validate_position_shape(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Validate the position tensor shape expected by local RoPE."""
        if position_ids.shape != x.shape[:2]:
            raise ValueError(
                f"position_ids must have shape {tuple(x.shape[:2])}, "
                f"got {tuple(position_ids.shape)}."
            )

    def _validate_active_mask_shape(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Validate the current-chunk active-mask contract."""
        if active_mask.shape != x.shape[:2]:
            raise ValueError(
                f"active_mask must have shape {tuple(x.shape[:2])}, "
                f"got {tuple(active_mask.shape)}."
            )
        if active_mask.dtype != torch.bool:
            raise ValueError(
                f"active_mask must have dtype torch.bool, got {active_mask.dtype}."
            )

    def _make_block_mask(
        self,
        active_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        query_len: int,
        kv_len: int,
        window_size: int,
        device: torch.device,
    ) -> Any:
        """Create the FlexAttention block mask for masked local continuation.

        The returned local frame is chronological in raw buffer order, but dead
        positions may remain inside it. Effective local order is therefore
        recovered from the active mask itself by taking a cumulative count over
        active positions.

        Queries still occupy the tail of the returned frame, so raw buffer order
        is used to locate query rows. Semantic active-token positions are then
        used to decide causality and sliding-window distance.
        """
        query_offset = kv_len - query_len
        semantic_positions = active_mask.long().cumsum(dim=-1) - 1

        def sliding_window_mask(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:

            q_abs = query_offset + q_idx

            query_is_active = active_mask[batch_idx, q_abs]
            key_is_active = active_mask[batch_idx, kv_idx]

            q_sem = semantic_positions[batch_idx, q_abs]
            k_sem = semantic_positions[batch_idx, kv_idx]

            is_causal = k_sem <= q_sem
            in_window = (q_sem - k_sem) < window_size

            return query_is_active & key_is_active & is_causal & in_window

        return create_block_mask(
            sliding_window_mask,
            B=batch_size,
            H=num_heads,
            Q_LEN=query_len,
            KV_LEN=kv_len,
            device=device,
        )