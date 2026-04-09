"""Decoder layer — a single transformer block.

Each block applies pre-norm attention followed by pre-norm MLP, with residual
connections around both sublayers:

    normed = RMSNorm(x)
    h      = x + Attention(normed, ...)
    normed = RMSNorm(h)
    out    = h + MLP(normed)

Pre-norm keeps the residual stream unnormalised. Gradients flow more cleanly
through unnormalised residuals at depth, and each sublayer receives a stable,
normalised view of the signal.

Two independent RMSNorm instances are used — one before attention, one before MLP.
They learn different scalings because they precede layers with different dynamic
ranges. Sharing them would be wrong.

torch.nn.RMSNorm is used directly (available from PyTorch 2.4+). It omits mean
subtraction, is faster than LayerNorm, and proved more stable at scale.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

#from .attention import GroupedQueryAttention
from .mlp import SwiGLUMLP


class DecoderLayer(nn.Module):
    """A single pre-norm transformer decoder block.

    Composes GroupedQueryAttention and SwiGLUMLP with residual connections and
    independent RMSNorm instances on each sublayer input.

    Args:
        config: Model config passed through to attention and MLP. Must also expose
            ``hidden_size`` and ``rms_norm_eps``.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm  = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp       = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Cache | None = None,
        layer_idx: int = 0,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one decoder block to the input.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).
            position_ids: Absolute positions of shape (batch, seq_len).
            cache: HuggingFace Cache object for KV accumulation, or None when
                caching is disabled. Passed through to attention unchanged.
            layer_idx: Cache slot index for this layer. Each layer has its own
                index so they accumulate independently within the shared cache.
            causal_mask: Optional boolean attention mask of shape
                (1, 1, seq_len, kv_len). Passed through to attention unchanged.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        attn_out = self.attention(self.attn_norm(x), position_ids, cache, layer_idx, causal_mask)
        h = x + attn_out
        return h + self.mlp(self.mlp_norm(h))
