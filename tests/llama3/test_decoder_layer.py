"""Tests for DecoderLayer.

Verifies shape, independent RMSNorm instances, residual connections, and that
attention and MLP are correctly integrated (output feeds through both paths).
"""

import torch

from src.llama3.configuration import Llama3Config
from src.llama3.decoder_layer import DecoderLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


def make_input(
    config: Llama3Config,
    batch: int = 2,
    seq: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq, config.hidden_size)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    return x, position_ids


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape_matches_input(self):
        """Input and output shapes must be identical."""
        config = small_config()
        layer = DecoderLayer(config)
        x, position_ids = make_input(config)
        out, _ = layer(x, position_ids)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_two_independent_rms_norms(self):
        """attn_norm and mlp_norm must be distinct objects with separate parameters."""
        config = small_config()
        layer = DecoderLayer(config)
        assert layer.attn_norm is not layer.mlp_norm
        assert layer.attn_norm.weight.data_ptr() != layer.mlp_norm.weight.data_ptr()


# ---------------------------------------------------------------------------
# Residual connections
# ---------------------------------------------------------------------------

class TestResidualConnections:
    def test_attention_residual_is_present(self):
        """Zeroing the attention sublayer output must not zero the full output.

        With a residual connection, output = x + attn(norm(x)) + ..., so even if
        attn produces zero, x still flows through. Without the residual, output
        would depend solely on attn.
        """
        config = small_config()
        layer = DecoderLayer(config)
        layer.eval()

        # Zero all attention projection weights so attn always outputs zero.
        for proj in (layer.attention.q_proj, layer.attention.k_proj,
                     layer.attention.v_proj, layer.attention.o_proj):
            torch.nn.init.zeros_(proj.weight)

        x, position_ids = make_input(config, batch=1, seq=4)
        out, _ = layer(x, position_ids)

        # Output must not be zero — x flows through the residual.
        assert not torch.all(out == 0)

    def test_mlp_residual_is_present(self):
        """Zeroing the MLP sublayer output must not zero the full output."""
        config = small_config()
        layer = DecoderLayer(config)
        layer.eval()

        # Zero all MLP weights so MLP always outputs zero.
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            torch.nn.init.zeros_(proj.weight)

        x, position_ids = make_input(config, batch=1, seq=4)
        out, _ = layer(x, position_ids)

        assert not torch.all(out == 0)

    def test_removing_residual_changes_output(self):
        """If residuals are bypassed, the output must differ from the full forward.

        Verifies by monkey-patching forward to skip the residual addition and
        confirming the outputs diverge.
        """
        config = small_config()
        layer = DecoderLayer(config)
        layer.eval()
        torch.manual_seed(0)

        x, position_ids = make_input(config, batch=1, seq=4)
        out_with_residual, _ = layer(x, position_ids)

        # Patch: replace forward with a version that drops the residual additions.
        def no_residual_forward(x, position_ids, past_key_value=None):
            attn_out, kv = layer.attention(layer.attn_norm(x), position_ids, past_key_value)
            h = attn_out  # no residual
            out = layer.mlp(layer.mlp_norm(h))  # no residual
            return out, kv

        out_no_residual, _ = no_residual_forward(x, position_ids)

        assert not torch.allclose(out_with_residual, out_no_residual)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_kv_cache_passes_through(self):
        """KV cache returned from the layer must be usable in the next step."""
        config = small_config()
        layer = DecoderLayer(config)
        layer.eval()
        torch.manual_seed(1)

        x = torch.randn(1, 4, config.hidden_size)
        pos_full = torch.arange(4).unsqueeze(0)
        out_full, _ = layer(x, pos_full)

        # Prefill first 2 tokens, then generate tokens 2 and 3 with cache.
        _, kv = layer(x[:, :2, :], torch.arange(2).unsqueeze(0))
        out_2, kv = layer(x[:, 2:3, :], torch.tensor([[2]]), past_key_value=kv)
        out_3, _  = layer(x[:, 3:4, :], torch.tensor([[3]]), past_key_value=kv)

        torch.testing.assert_close(out_2, out_full[:, 2:3, :])
        torch.testing.assert_close(out_3, out_full[:, 3:4, :])
