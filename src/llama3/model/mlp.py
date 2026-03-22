"""SwiGLU feed-forward sublayer.

SwiGLU is a gated linear unit variant that multiplies a SiLU-gated projection
element-wise against a separate up-projection:

    output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

The gating mechanism gives the network more expressive control over which features
to propagate than a plain two-matrix FFN. It requires three weight matrices instead
of two, which is why intermediate_size in Llama 3 is set lower than the 4× multiplier
typical of two-matrix FFNs — the total parameter count remains comparable.

SiLU is used as the gate activation because LLama3 commited to SwiGLU specifically
— a fixed architectural choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward sublayer.

    Implements the three-matrix SwiGLU FFN used in Llama 3:

        output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

    No bias on any projection. SiLU as the gate activation is an architectural
    constant — it is what defines SwiGLU specifically.

    Args:
        config: Model config. Must expose ``hidden_size`` and ``intermediate_size``.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
