"""Tests for SlidingWindowAttention.

Invariants verified: output shape and composition compatibility, local sliding-window
restriction, causal ordering within the active local window, direct consumption and
update of DynamicSlidingWindowLayer, repeated cached forward-pass behavior, cached
generation consistency with full forward execution, local RoPE default-mode usage,
insensitivity to YaRN / long-context config changes elsewhere in the model, and
sane backend-facing block-mask caching behavior.
"""

import torch
from transformers.cache_utils import DynamicSlidingWindowLayer

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.sliding_window_attention import SlidingWindowAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_sliding_window_heads=4,
        num_mosrah_heads=4,
        num_selected_heads=4,
        head_dim=8,
        window_size=4,
        attention_dropout=0.0,
        local_rope_theta=10000.0,
        mosrah_rope_theta=10000.0,
        training_sequence_length=128,
        inference_sequence_length=128,
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


def make_cache(config: ShramConfig) -> DynamicSlidingWindowLayer:
    return DynamicSlidingWindowLayer(sliding_window=config.window_size)


# ---------------------------------------------------------------------------
# Shape / composition compatibility
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape(self):
        """(B, N, hidden_size) -> (B, N, hidden_size)."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids = make_input(config)
        out = attn(x, position_ids)
        assert out.shape == x.shape

    def test_output_dtype_matches_input(self):
        """Output dtype should remain compatible with downstream SHRAM composition."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids = make_input(config)
        out = attn(x, position_ids)
        assert out.dtype == x.dtype

    def test_invalid_hidden_size_raises(self):
        """The module should reject inputs whose last dimension is not hidden_size."""
        config = small_config(hidden_size=64)
        attn = SlidingWindowAttention(config)
        x = torch.randn(2, 5, 63)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

        try:
            attn(x, position_ids)
            assert False, "Expected ValueError for invalid x last dimension."
        except ValueError as exc:
            assert "hidden_size" in str(exc)

    def test_invalid_position_shape_raises(self):
        """The module should reject position_ids whose shape does not match (B, N)."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, _ = make_input(config, batch=2, seq=5)
        bad_position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

        try:
            attn(x, bad_position_ids)
            assert False, "Expected ValueError for invalid position_ids shape."
        except ValueError as exc:
            assert "position_ids" in str(exc)


# ---------------------------------------------------------------------------
# Local-window and causal behavior
# ---------------------------------------------------------------------------

class TestLocalWindowBehavior:
    def test_tokens_outside_local_window_do_not_affect_output(self):
        """Changing tokens older than the local window must not affect a late output."""
        config = small_config(window_size=3)
        attn = SlidingWindowAttention(config)
        attn.eval()
        torch.manual_seed(0)

        x, position_ids = make_input(config, batch=1, seq=6)
        out_original = attn(x, position_ids)

        # For the last token at position 5 with window_size=3, only positions 3, 4, 5
        # are inside the active local window. Positions 0, 1, 2 must not contribute.
        x_modified = x.clone()
        x_modified[:, :3, :] = torch.randn_like(x_modified[:, :3, :])
        out_modified = attn(x_modified, position_ids)

        torch.testing.assert_close(out_original[:, 5:6, :], out_modified[:, 5:6, :])

    def test_future_tokens_do_not_affect_past_outputs(self):
        """Replacing future tokens must leave earlier outputs unchanged."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config)
        attn.eval()
        torch.manual_seed(1)

        x, position_ids = make_input(config, batch=1, seq=6)
        out_original = attn(x, position_ids)

        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn_like(x_modified[:, 4:, :])
        out_modified = attn(x_modified, position_ids)

        torch.testing.assert_close(out_original[:, :4, :], out_modified[:, :4, :])


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

class TestSlidingWindowCache:
    def test_forward_consumes_and_updates_cache(self):
        """A forward pass with cache must populate the provided DynamicSlidingWindowLayer."""
        config = small_config(window_size=4, num_sliding_window_heads=4, head_dim=8)
        attn = SlidingWindowAttention(config)
        x, position_ids = make_input(config, batch=2, seq=3)
        cache = make_cache(config)

        attn(x, position_ids, cache=cache)

        assert cache.get_seq_length() == 3
        expected_shape = (2, config.num_sliding_window_heads, 3, config.head_dim)
        assert cache.keys.shape == expected_shape
        assert cache.values.shape == expected_shape

    def test_repeated_cached_forward_retains_last_window_minus_one_tokens(self):
        """Once the cache is full, it should retain only the last window_size - 1 tokens."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config)
        attn.eval()
        torch.manual_seed(2)

        x, position_ids = make_input(config, batch=1, seq=5)
        cache = make_cache(config)

        # Prefill 2 tokens.
        attn(x[:, :2, :], position_ids[:, :2], cache=cache)
        assert cache.get_seq_length() == 2
        assert cache.keys.shape[-2] == 2
        assert cache.values.shape[-2] == 2

        # Add token 2.
        attn(x[:, 2:3, :], position_ids[:, 2:3], cache=cache)
        assert cache.get_seq_length() == 3
        assert cache.keys.shape[-2] == 3
        assert cache.values.shape[-2] == 3

        # Add token 3: cumulative length reaches window, retained cache stays at 3.
        attn(x[:, 3:4, :], position_ids[:, 3:4], cache=cache)
        assert cache.get_seq_length() == 4
        assert cache.keys.shape[-2] == config.window_size - 1
        assert cache.values.shape[-2] == config.window_size - 1

        # Add token 4: cumulative length grows again, retained cache still stays at 3.
        attn(x[:, 4:5, :], position_ids[:, 4:5], cache=cache)
        assert cache.get_seq_length() == 5
        assert cache.keys.shape[-2] == config.window_size - 1
        assert cache.values.shape[-2] == config.window_size - 1

    def test_cached_generation_matches_full_forward(self):
        """Cached generation must match the corresponding positions of a full forward pass."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config)
        attn.eval()
        torch.manual_seed(3)

        x, position_ids = make_input(config, batch=1, seq=6)

        # Full forward across all tokens.
        out_full = attn(x, position_ids)

        # Prefill 3 tokens, then decode one at a time.
        cache = make_cache(config)
        attn(x[:, :3, :], position_ids[:, :3], cache=cache)

        out_3 = attn(x[:, 3:4, :], position_ids[:, 3:4], cache=cache)
        out_4 = attn(x[:, 4:5, :], position_ids[:, 4:5], cache=cache)
        out_5 = attn(x[:, 5:6, :], position_ids[:, 5:6], cache=cache)

        torch.testing.assert_close(out_3, out_full[:, 3:4, :])
        torch.testing.assert_close(out_4, out_full[:, 4:5, :])
        torch.testing.assert_close(out_5, out_full[:, 5:6, :])


# ---------------------------------------------------------------------------
# RoPE behavior
# ---------------------------------------------------------------------------

class TestLocalRoPE:
    def test_local_rope_is_constructed_in_default_mode(self):
        """The local path must always construct RoPE in default mode."""
        config = small_config(local_rope_theta=12345.0)
        attn = SlidingWindowAttention(config)

        assert attn.rope.mode == "default"
        assert attn.rope.attention_scaling == 1.0

    def test_local_behavior_does_not_change_when_yarn_fields_change_elsewhere(self):
        """Changing non-local YaRN fields elsewhere in config must not affect local outputs."""
        base_config = small_config(
            local_rope_theta=10000.0,
            mosrah_rope_theta=10000.0,
            training_sequence_length=128,
            inference_sequence_length=128,
            alpha=1.0,
            beta=32.0,
        )
        altered_config = small_config(
            local_rope_theta=10000.0,
            mosrah_rope_theta=7777.0,
            training_sequence_length=256,
            inference_sequence_length=4096,
            alpha=2.0,
            beta=16.0,
        )

        torch.manual_seed(4)
        attn_a = SlidingWindowAttention(base_config)
        torch.manual_seed(4)
        attn_b = SlidingWindowAttention(altered_config)

        x, position_ids = make_input(base_config, batch=1, seq=6)

        out_a = attn_a(x, position_ids)
        out_b = attn_b(x, position_ids)

        torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# Backend-facing behavior
# ---------------------------------------------------------------------------

class TestBackendBehavior:
    def test_block_mask_cache_reuses_recent_shapes(self):
        """The cached block-mask helper should reuse identical recent requests."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config)

        SlidingWindowAttention._make_block_mask.cache_clear()
        info_before = SlidingWindowAttention._make_block_mask.cache_info()

        mask_1 = attn._make_block_mask(
            batch_size=1,
            num_heads=config.num_sliding_window_heads,
            query_len=1,
            kv_len=4,
            window_size=config.window_size,
            device_str="cpu",
        )
        info_mid = SlidingWindowAttention._make_block_mask.cache_info()

        mask_2 = attn._make_block_mask(
            batch_size=1,
            num_heads=config.num_sliding_window_heads,
            query_len=1,
            kv_len=4,
            window_size=config.window_size,
            device_str="cpu",
        )
        info_after = SlidingWindowAttention._make_block_mask.cache_info()

        assert info_mid.misses == info_before.misses + 1
        assert info_after.hits == info_mid.hits + 1
        assert mask_1 is mask_2