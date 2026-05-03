"""Decoder layer — a single transformer block.

Each block applies pre-norm hybrid attention followed by pre-norm MLP, with
residual connections around both sublayers:

    normed_attn = RMSNorm(x)
    attn_out, load_balance_loss, max_vio = SHRAMHybridLayer(normed_attn, ...)
    h = x + attn_out

    normed_mlp = RMSNorm(h)
    mlp_out = SwiGLUMLP(normed_mlp)
    out = h + mlp_out

Pre-norm keeps the residual stream unnormalised. Gradients flow more cleanly
through unnormalised residuals at depth, and each sublayer receives a stable,
normalised view of the signal.

Two independent RMSNorm instances are used — one before attention, one before
MLP. They learn different scalings because they precede layers with different
dynamic ranges. Sharing them would be wrong.

torch.nn.RMSNorm is used directly (available from PyTorch 2.4+). It omits mean
subtraction, is faster than LayerNorm, and proved more stable at scale.
"""

import torch
import torch.nn as nn

from .attention.shram import SHRAMHybridLayer
from .cache.shram_layer_cache import ShramLayerCache
from .configuration import ShramConfig
from .mlp import SwiGLUMLP


class DecoderLayer(nn.Module):
    """A single pre-norm SHRAM decoder block.

    Composes SHRAMHybridLayer and SwiGLUMLP with residual connections and
    independent RMSNorm instances on each sublayer input.

    Args:
        config: SHRAM config. Must expose ``hidden_size`` and ``rms_norm_eps``
            in addition to the fields required by SHRAMHybridLayer and
            SwiGLUMLP.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = SHRAMHybridLayer(config)
        self.mlp = SwiGLUMLP(config)

    def num_mosrah_parameters(self) -> int:
        """Return the total number of trainable MoSRAH parameters in this decoder layer."""
        return self.attention.num_mosrah_parameters()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: ShramLayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one decoder block to the input.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).
            position_ids: Authoritative positions of shape (batch, seq_len).
            active_mask: Current-chunk active mask of shape (batch, seq_len),
                where True means the token is semantically live. Forwarded
                unchanged to the hybrid attention layer.
            cache: Optional per-layer SHRAM cache passed through to the hybrid
                attention layer unchanged.

        Returns:
            output: Tensor of shape (batch, seq_len, hidden_size).
            load_balance_loss: Scalar sparse-path load-balance loss propagated
                from SHRAMHybridLayer.
            max_vio: Detached scalar routing-imbalance summary. Passed through
                unchanged from SHRAMHybridLayer; see MoSRAHRouter for semantics.
        """
        attn_out, load_balance_loss, max_vio = self.attention(
            hidden_states=self.attn_norm(x),
            position_ids=position_ids,
            active_mask=active_mask,
            cache=cache,
        )
        hidden_states = x + attn_out
        output = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return output, load_balance_loss, max_vio