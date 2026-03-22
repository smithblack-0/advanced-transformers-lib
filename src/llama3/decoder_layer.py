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

from src.llama3.attention import GroupedQueryAttention
from src.llama3.mlp import SwiGLUMLP
from src.llama3.type_aliases import KVCache


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
        past_key_value: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Apply one decoder block to the input.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).
            position_ids: Absolute positions of shape (batch, seq_len).
            past_key_value: KV cache from prior steps, or None during prefill.

        Returns:
            Tuple of:
            - Output tensor of shape (batch, seq_len, hidden_size).
            - Updated KV cache to pass to the next step.
        """
        attn_out, present_key_value = self.attention(
            self.attn_norm(x), position_ids, past_key_value
        )
        h = x + attn_out
        out = h + self.mlp(self.mlp_norm(h))
        return out, present_key_value
