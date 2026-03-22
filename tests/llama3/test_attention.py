"""Tests for GroupedQueryAttention.

Invariants verified: output shape, no bias on any projection, correct projection
dimensions, GQA head layout, causal masking (future tokens do not affect past outputs),
KV cache consistency (cached generation matches full forward pass), and the MHA/MQA
edge cases.
"""

import torch
import pytest

from src.llama3.configuration import Llama3Config
from src.llama3.attention import GroupedQueryAttention


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
    def test_output_shape(self):
        """(batch, seq, hidden_size) → (batch, seq, hidden_size)."""
        config = small_config()
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config)
        out, _ = attn(x, position_ids)
        assert out.shape == x.shape

    def test_kv_cache_is_list_of_chunks(self):
        """Returned KV cache must be a pair of lists, each containing one chunk tensor."""
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config, seq=6)
        _, (k_chunks, v_chunks) = attn(x, position_ids)
        assert isinstance(k_chunks, list) and len(k_chunks) == 1
        assert isinstance(v_chunks, list) and len(v_chunks) == 1
        expected = (2, config.num_key_value_heads, 6, config.head_dim)
        assert k_chunks[0].shape == expected
        assert v_chunks[0].shape == expected


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

class TestProjections:
    def test_no_bias_on_any_projection(self):
        config = small_config()
        attn = GroupedQueryAttention(config)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.o_proj.bias is None

    def test_q_proj_dimensions(self):
        config = small_config(hidden_size=64, num_attention_heads=4)
        attn = GroupedQueryAttention(config)
        assert attn.q_proj.weight.shape == (config.num_attention_heads * config.head_dim, 64)

    def test_k_proj_dimensions(self):
        config = small_config(hidden_size=64, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        assert attn.k_proj.weight.shape == (config.num_key_value_heads * config.head_dim, 64)

    def test_o_proj_dimensions(self):
        config = small_config(hidden_size=64, num_attention_heads=4)
        attn = GroupedQueryAttention(config)
        assert attn.o_proj.weight.shape == (64, config.num_attention_heads * config.head_dim)

    def test_invalid_head_ratio_raises(self):
        """num_attention_heads not divisible by num_key_value_heads must raise."""
        with pytest.raises(ValueError, match="num_attention_heads"):
            GroupedQueryAttention(small_config(num_attention_heads=4, num_key_value_heads=3))


# ---------------------------------------------------------------------------
# GQA head configurations
# ---------------------------------------------------------------------------

class TestHeadConfigurations:
    def test_mha_is_valid(self):
        """num_kv_heads == num_heads gives standard MHA — must work correctly."""
        config = small_config(num_attention_heads=4, num_key_value_heads=4)
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config)
        out, _ = attn(x, position_ids)
        assert out.shape == x.shape

    def test_mqa_is_valid(self):
        """num_kv_heads == 1 gives MQA — must work correctly."""
        config = small_config(num_attention_heads=4, num_key_value_heads=1)
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config)
        out, _ = attn(x, position_ids)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

class TestCausalMasking:
    def test_future_tokens_do_not_affect_past_outputs(self):
        """Replacing tokens at positions > t must leave outputs at positions <= t unchanged.

        This verifies that the causal mask is active: each token attends only to
        itself and earlier positions, so later tokens cannot influence earlier outputs.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(0)

        seq = 6
        x, position_ids = make_input(config, batch=1, seq=seq)

        out_original, _ = attn(x, position_ids)

        # Replace the last two tokens with different values.
        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn_like(x_modified[:, 4:, :])
        out_modified, _ = attn(x_modified, position_ids)

        # Outputs at positions 0..3 must be identical.
        torch.testing.assert_close(out_original[:, :4, :], out_modified[:, :4, :])


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_single_token_prefill(self):
        """A single-token prefill (no cache, seq_len=1) must match a full forward pass.

        This is the smallest valid prefill and exercises is_causal=True with a 1×1
        attention matrix, which is a trivially causal case.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(5)

        x = torch.randn(1, 4, config.hidden_size)
        pos_full = torch.arange(4).unsqueeze(0)
        out_full, _ = attn(x, pos_full)

        # Single-token prefill then generate the remaining 3 tokens one at a time.
        _, kv = attn(x[:, :1, :], torch.tensor([[0]]))
        for t in range(1, 4):
            out_t, kv = attn(x[:, t:t+1, :], torch.tensor([[t]]), past_key_value=kv)
            torch.testing.assert_close(out_t, out_full[:, t:t+1, :])

    def test_cache_grows_by_one_per_step(self):
        """Each generation step must append exactly one chunk to the cache."""
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(6)

        x = torch.randn(1, 5, config.hidden_size)
        _, kv = attn(x[:, :2, :], torch.arange(2).unsqueeze(0))
        assert len(kv[0]) == 1  # prefill produces one chunk

        for step in range(3):
            _, kv = attn(x[:, 2+step:3+step, :], torch.tensor([[2+step]]), past_key_value=kv)
            assert len(kv[0]) == step + 2  # one more chunk each step

    def test_cached_generation_matches_full_forward(self):
        """Cached generation must produce identical outputs to a full forward pass.

        Runs a 2-token prefill followed by 2 cached generation steps, then verifies
        each cached output matches the corresponding position in a full 4-token forward.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(1)

        batch = 1
        x = torch.randn(batch, 4, config.hidden_size)

        # Full forward over all 4 tokens.
        pos_full = torch.arange(4).unsqueeze(0)
        out_full, _ = attn(x, pos_full)

        # Prefill on first 2 tokens.
        pos_prefill = torch.arange(2).unsqueeze(0)
        _, kv = attn(x[:, :2, :], pos_prefill)

        # Cached step: token 2.
        pos_2 = torch.tensor([[2]])
        out_2, kv = attn(x[:, 2:3, :], pos_2, past_key_value=kv)

        # Cached step: token 3.
        pos_3 = torch.tensor([[3]])
        out_3, _ = attn(x[:, 3:4, :], pos_3, past_key_value=kv)

        torch.testing.assert_close(out_2, out_full[:, 2:3, :])
        torch.testing.assert_close(out_3, out_full[:, 3:4, :])
