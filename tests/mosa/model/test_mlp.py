"""Tests for SwiGLUMLP.

Verifies the invariants documented in the plan: shape preservation, no bias on any
projection, and that the gate is genuinely active (not a no-op).
"""

import torch

from src.mosa.model.configuration import MosaConfig
from src.mosa.model.mlp import SwiGLUMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> MosaConfig:
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    defaults.update(kwargs)
    return MosaConfig(**defaults)


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
# Projection dimensions
# ---------------------------------------------------------------------------

class TestProjectionDimensions:
    def test_gate_proj_dimensions(self):
        """gate_proj must map hidden_size → intermediate_size as specified in config."""
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.gate_proj.weight.shape == (128, 64)

    def test_up_proj_dimensions(self):
        """up_proj must map hidden_size → intermediate_size as specified in config."""
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.up_proj.weight.shape == (128, 64)

    def test_down_proj_dimensions(self):
        """down_proj must map intermediate_size → hidden_size as specified in config."""
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        assert mlp.down_proj.weight.shape == (64, 128)


# ---------------------------------------------------------------------------
# Gating behaviour
# ---------------------------------------------------------------------------

class TestGating:
    def test_output_is_nonlinear(self):
        """Output must scale non-linearly with input magnitude.

        A purely linear transform satisfies f(αx) = αf(x). SiLU breaks this, so
        doubling the input must not double the output. This confirms SiLU is applied
        and not silently removed.
        """
        config = small_config(hidden_size=64, intermediate_size=128)
        mlp = SwiGLUMLP(config)
        mlp.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 4, 64)

        out_x = mlp(x)
        out_2x = mlp(2 * x)

        assert not torch.allclose(out_2x, 2 * out_x)

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
