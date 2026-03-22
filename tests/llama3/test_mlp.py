"""Tests for SwiGLUMLP.

Verifies the invariants documented in the plan: shape preservation, no bias on any
projection, and that the gate is genuinely active (not a no-op).
"""

import torch

from src.llama3.configuration import Llama3Config
from src.llama3.mlp import SwiGLUMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape_matches_input(self):
        """(batch, seq, hidden_size) → (batch, seq, hidden_size)."""
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        x = torch.randn(2, 8, 64)
        assert mlp(x).shape == x.shape

    def test_different_batch_and_seq(self):
        """Shape invariant holds across varying batch and sequence dimensions."""
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        x = torch.randn(4, 32, 64)
        assert mlp(x).shape == x.shape


# ---------------------------------------------------------------------------
# No bias
# ---------------------------------------------------------------------------

class TestNoBias:
    def test_gate_proj_has_no_bias(self):
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.gate_proj.bias is None

    def test_up_proj_has_no_bias(self):
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.up_proj.bias is None

    def test_down_proj_has_no_bias(self):
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.down_proj.bias is None


# ---------------------------------------------------------------------------
# Gating behaviour
# ---------------------------------------------------------------------------

class TestGating:
    def test_zeroing_gate_proj_zeros_output(self):
        """When W_gate produces all zeros, SiLU(0) = 0, so the gate kills the output.

        This confirms the gate is genuinely active: it controls whether any signal
        passes through to W_down. If gating were absent the output would be non-zero.
        """
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        torch.nn.init.zeros_(mlp.gate_proj.weight)

        x = torch.randn(2, 8, 64)
        output = mlp(x)

        assert torch.all(output == 0)
