"""Tests for GroupedQueryAttention.

Invariants verified: output shape, no bias on any projection, correct projection
dimensions, GQA head layout, causal masking (future tokens do not affect past outputs),
KV cache consistency (cached generation matches full forward pass), and the MHA/MQA
edge cases.
"""

import torch
import pytest
from transformers import DynamicCache

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention import GroupedQueryAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def make_input(
    config: ShramConfig,
    batch: int = 2,
    seq: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq, config.hidden_size)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    return x, position_ids


def make_causal_mask(cache_position: torch.Tensor, k_len: int) -> torch.Tensor:
    """Build a boolean causal mask of shape (1, 1, q_len, k_len).

    True at (q, k) means query q may attend to key k. Equivalent to the
    lower-right aligned causal mask: key position k is within the causal
    horizon of query q when k <= cache_position[q].
    """
    k_positions = torch.arange(k_len)
    return (k_positions[None, :] <= cache_position[:, None]).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape(self):
        """(batch, seq, hidden_size) → (batch, seq, hidden_size)."""
        config = small_config()
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config)
        out = attn(x, position_ids)
        assert out.shape == x.shape

    def test_cache_stores_kv_after_forward(self):
        """After a forward pass with a cache, the cache must contain K and V for this layer."""
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config, seq=6)
        cache = DynamicCache()
        attn(x, position_ids, cache=cache, layer_idx=0)

        # The cache should now hold layer 0's K and V.
        assert len(cache.layers) == 1
        expected_shape = (2, config.num_key_value_heads, 6, config.head_dim)
        assert cache.layers[0].keys.shape == expected_shape
        assert cache.layers[0].values.shape == expected_shape


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
        out = attn(x, position_ids)
        assert out.shape == x.shape

    def test_mqa_is_valid(self):
        """num_kv_heads == 1 gives MQA — must work correctly."""
        config = small_config(num_attention_heads=4, num_key_value_heads=1)
        attn = GroupedQueryAttention(config)
        x, position_ids = make_input(config)
        out = attn(x, position_ids)
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

        out_original = attn(x, position_ids)

        # Replace the last two tokens with different values.
        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn_like(x_modified[:, 4:, :])
        out_modified = attn(x_modified, position_ids)

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

        Decode steps receive an explicit causal mask because is_causal=True is only
        correct for square Q×K matrices. When q_len=1 and k_len>1, PyTorch's built-in
        is_causal uses upper-left alignment and produces wrong results.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(5)

        x = torch.randn(1, 4, config.hidden_size)
        pos_full = torch.arange(4).unsqueeze(0)
        out_full = attn(x, pos_full)

        # Single-token prefill then generate the remaining 3 tokens one at a time.
        cache = DynamicCache()
        attn(x[:, :1, :], torch.tensor([[0]]), cache=cache, layer_idx=0)
        for t in range(1, 4):
            past_len = cache.get_seq_length(0)
            k_len = past_len + 1
            mask = make_causal_mask(torch.tensor([t]), k_len)
            out_t = attn(x[:, t:t+1, :], torch.tensor([[t]]), cache=cache, layer_idx=0, causal_mask=mask)
            torch.testing.assert_close(out_t, out_full[:, t:t+1, :])

    def test_cache_grows_after_each_step(self):
        """Each generation step must increase the cached sequence length by one."""
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(6)

        x = torch.randn(1, 5, config.hidden_size)
        cache = DynamicCache()

        # Prefill 2 tokens.
        attn(x[:, :2, :], torch.arange(2).unsqueeze(0), cache=cache, layer_idx=0)
        assert cache.get_seq_length(0) == 2

        for step in range(3):
            t = 2 + step
            past_len = cache.get_seq_length(0)
            mask = make_causal_mask(torch.tensor([t]), k_len=past_len + 1)
            attn(x[:, t:t+1, :], torch.tensor([[t]]), cache=cache, layer_idx=0, causal_mask=mask)
            assert cache.get_seq_length(0) == 3 + step

    def test_cached_generation_matches_full_forward(self):
        """Cached generation must produce identical outputs to a full forward pass.

        Runs a 2-token prefill followed by 2 cached generation steps, then verifies
        each cached output matches the corresponding position in a full 4-token forward.

        Decode steps receive an explicit causal mask (see test_single_token_prefill
        for the reason). The mask is constructed here as it would be in huggingface.py.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(1)

        batch = 1
        x = torch.randn(batch, 4, config.hidden_size)

        # Full forward over all 4 tokens — no mask, is_causal=True, square Q×K.
        pos_full = torch.arange(4).unsqueeze(0)
        out_full = attn(x, pos_full)

        # Prefill on first 2 tokens — no mask, is_causal=True, square Q×K.
        cache = DynamicCache()
        pos_prefill = torch.arange(2).unsqueeze(0)
        attn(x[:, :2, :], pos_prefill, cache=cache, layer_idx=0)

        # Cached step: token 2. k_len = 3 (2 cached + 1 new).
        pos_2 = torch.tensor([[2]])
        mask_2 = make_causal_mask(torch.tensor([2]), k_len=3)
        out_2 = attn(x[:, 2:3, :], pos_2, cache=cache, layer_idx=0, causal_mask=mask_2)

        # Cached step: token 3. k_len = 4 (3 cached + 1 new).
        pos_3 = torch.tensor([[3]])
        mask_3 = make_causal_mask(torch.tensor([3]), k_len=4)
        out_3 = attn(x[:, 3:4, :], pos_3, cache=cache, layer_idx=0, causal_mask=mask_3)

        torch.testing.assert_close(out_2, out_full[:, 2:3, :])
        torch.testing.assert_close(out_3, out_full[:, 3:4, :])


    def test_causal_mask_parameter_is_used(self):
        """An explicit causal_mask must govern attention when provided.

        Passes a mask that blocks all attention (all False) and verifies the output
        differs from the unmasked case. This confirms the parameter is wired through
        to SDPA and not silently ignored.
        """
        config = small_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(config)
        attn.eval()
        torch.manual_seed(42)

        x, position_ids = make_input(config, batch=1, seq=4)
        out_no_mask = attn(x, position_ids)

        # All-False mask: no token may attend to any key.
        all_blocked = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        out_masked = attn(x, position_ids, causal_mask=all_blocked)

        assert not torch.allclose(out_no_mask, out_masked)
