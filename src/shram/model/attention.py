"""Grouped Query Attention (GQA).

GQA reduces KV cache memory by sharing key-value heads across groups of query heads.
With G query heads per KV head, the KV cache is G× smaller than standard multi-head
attention (MHA). This is the primary motivation for its use in Llama 3 at 128K context:
8 KV heads shared across 32 query heads gives a 4× cache reduction.

Setting num_key_value_heads == num_attention_heads recovers standard MHA.
Setting num_key_value_heads == 1 gives multi-query attention (MQA).

Attention is computed via torch.nn.functional.scaled_dot_product_attention (SDPA),
which selects FlashAttention when hardware and dtype allow, falling back to standard
attention otherwise. No custom kernel or additional dependency required.

KV caching is handled via HuggingFace's Cache protocol. The cache owns K/V storage and
accumulation; attention only calls cache.update() to store new projections and retrieve
the full accumulated history. This cleanly separates attention computation from cache
management: different Cache subclasses (DynamicCache, StaticCache, custom research
variants) can be dropped in without touching the attention logic.

No bias on any projection — a fixed architectural constant of this model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE, causal masking, and KV cache support.

    Implements GQA as used in Llama 3: Q heads are split into groups, each group
    sharing a single KV head. Before attention is computed, K and V are expanded
    by repeating each KV head across its group of query heads.

    The forward pass is strictly causal. An optional pre-built boolean attention
    mask can be threaded in from the caller; when absent, SDPA's native
    ``is_causal`` mode applies — correct for full-sequence training.

    Args:
        config: Model config. Must expose ``num_attention_heads``,
            ``num_key_value_heads``, ``head_dim``, ``hidden_size``,
            and ``attention_dropout``.

    Raises:
        ValueError: If ``num_attention_heads`` is not divisible by
            ``num_key_value_heads``.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads})."
            )

        # No bias on any projection — architectural constant.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = RotaryEmbedding(config)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Cache | None = None,
        layer_idx: int = 0,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply grouped query attention to the input.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).
            position_ids: Absolute positions of shape (batch, seq_len). Used by
                RoPE to rotate Q and K at the correct frequencies.
            cache: HuggingFace Cache object for KV accumulation, or None when
                caching is disabled (``use_cache=False``). When provided,
                ``cache.update(k, v, layer_idx)`` stores the new K/V and returns
                the full accumulated key and value tensors for this layer.
            layer_idx: Which slot in the cache to read and write. Each decoder
                layer has its own index so they accumulate independently.
            causal_mask: Optional boolean attention mask of shape
                (1, 1, seq_len, kv_len), where True indicates a position that
                should be attended to. When None, SDPA's built-in ``is_causal``
                mode is used, which is correct for full-sequence training
                (square Q×K matrix). When provided, ``is_causal`` is disabled
                and the explicit mask governs attention — required for any
                generation pattern where Q and K lengths differ.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        batch, seq_len, _ = x.shape

        # Project and reshape: (batch, seq_len, heads * head_dim)
        #                    → (batch, heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE. attention_scaling is 1.0 for default/linear; YaRN returns a
        # value != 1.0 that corrects attention magnitude after frequency manipulation.
        q, k, attention_scaling = self.rope(q, k, position_ids)

        if cache is not None:
            k_full, v_full = cache.update(k, v, layer_idx)
        else:
            k_full, v_full = k, v

        # Expand KV heads to align with query heads for GQA.
        # Each KV head is repeated num_groups times so SDPA sees matching head counts.
        if self.num_groups > 1:
            k_full = k_full.repeat_interleave(self.num_groups, dim=1)
            v_full = v_full.repeat_interleave(self.num_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None,
            scale=attention_scaling / math.sqrt(self.head_dim),
        )

        # Merge heads and project back to hidden_size.
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.head_dim)
        )

        return self.o_proj(attn_output)
