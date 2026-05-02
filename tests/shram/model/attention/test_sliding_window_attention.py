# tests/shram/model/attention/test_sliding_window_attention.py

"""Tests for SlidingWindowAttention.

Verifies the local-path invariants after masked continuation support:
- output shape is preserved
- tokens outside the local window do not affect local outputs
- causality is preserved within the local window
- current-chunk active_mask is validated and respected
- the real LocalSlidingWindowLayerCache path works across repeated calls
- cached generation consistency with full forward execution
- dead current-chunk tokens do not affect later live outputs
- dead cached tokens do not affect later live outputs
- local RoPE remains default-mode and insensitive to non-local YaRN fields
"""

import pytest
import torch

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.sliding_window_attention import SlidingWindowAttention
from src.shram.model.cache.sliding_window_cache import LocalSlidingWindowLayerCache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlexAttention does not support backward on CPU",
)

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
        alpha=1.0,
        beta=32.0,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def synthetic_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_sliding_window_heads=1,
        num_mosrah_heads=1,
        num_selected_heads=1,
        head_dim=8,
        window_size=4,
        attention_dropout=0.0,
        local_rope_theta=10000.0,
        mosrah_rope_theta=10000.0,
        training_sequence_length=64,
        inference_sequence_length=64,
        alpha=1.0,
        beta=32.0,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def make_input(
    config: ShramConfig,
    batch: int = 2,
    seq: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq, config.hidden_size)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    active_mask = torch.ones(batch, seq, dtype=torch.bool)
    return x, position_ids, active_mask


def make_cache(
    config: ShramConfig,
    *,
    batch: int = 1,
) -> LocalSlidingWindowLayerCache:
    return LocalSlidingWindowLayerCache(
        sliding_window=config.window_size,
        num_heads=config.num_sliding_window_heads,
        head_dim=config.head_dim,
        batch_size=batch,
        device=torch.device("cpu"),
    )


def synthetic_tokens(
    values: list[list[float]],
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Build small hand-written token sequences with repeated channel values."""
    base = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
    return base.repeat(1, 1, hidden_size)


# ---------------------------------------------------------------------------
# Shape / composition compatibility
# ---------------------------------------------------------------------------


class TestShape:
    def test_output_shape(self):
        """(B, N, hidden_size) -> (B, N, hidden_size)."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids, active_mask = make_input(config)
        out = attn(x, position_ids, active_mask)
        assert out.shape == x.shape

    def test_output_dtype_matches_input(self):
        """Output dtype should remain compatible with downstream SHRAM composition."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids, active_mask = make_input(config)
        out = attn(x, position_ids, active_mask)
        assert out.dtype == x.dtype

    def test_invalid_position_shape_raises(self):
        """The module should reject position_ids whose shape does not match (B, N)."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, _, active_mask = make_input(config, batch=2, seq=5)
        bad_position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

        with pytest.raises(ValueError) as exc_info:
            attn(x, bad_position_ids, active_mask)

        assert "position_ids" in str(exc_info.value)

    def test_invalid_active_mask_shape_raises(self):
        """The module should reject active_mask whose shape does not match (B, N)."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids, _ = make_input(config, batch=2, seq=5)
        bad_active_mask = torch.ones(2, 6, dtype=torch.bool)

        with pytest.raises(ValueError) as exc_info:
            attn(x, position_ids, bad_active_mask)

        assert "active_mask" in str(exc_info.value)

    def test_invalid_active_mask_dtype_raises(self):
        """The module should reject active_mask whose dtype is not torch.bool."""
        config = small_config()
        attn = SlidingWindowAttention(config)
        x, position_ids, _ = make_input(config, batch=2, seq=5)
        bad_active_mask = torch.ones(2, 5, dtype=torch.long)

        with pytest.raises(ValueError) as exc_info:
            attn(x, position_ids, bad_active_mask)

        assert "torch.bool" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Local-window and causal behavior
# ---------------------------------------------------------------------------


class TestLocalWindowBehavior:
    def test_tokens_outside_local_window_do_not_affect_output(self):
        """Changing tokens older than the local window must not affect a late output."""
        config = small_config(window_size=3)
        attn = SlidingWindowAttention(config).eval()

        torch.manual_seed(0)
        x, position_ids, active_mask = make_input(config, batch=1, seq=6)
        out_original = attn(x, position_ids, active_mask)

        # For the last token at position 5 with window_size=3, only positions 3, 4, 5
        # are inside the active local window. Positions 0, 1, 2 must not contribute.
        x_modified = x.clone()
        x_modified[:, :3, :] = torch.randn_like(x_modified[:, :3, :])
        out_modified = attn(x_modified, position_ids, active_mask)

        torch.testing.assert_close(out_original[:, 5:6, :], out_modified[:, 5:6, :])

    def test_future_tokens_do_not_affect_past_outputs(self):
        """Replacing future tokens must leave earlier outputs unchanged."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        torch.manual_seed(1)
        x, position_ids, active_mask = make_input(config, batch=1, seq=6)
        out_original = attn(x, position_ids, active_mask)

        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn_like(x_modified[:, 4:, :])
        out_modified = attn(x_modified, position_ids, active_mask)

        torch.testing.assert_close(out_original[:, :4, :], out_modified[:, :4, :])

    def test_causal_ordering_is_preserved_within_active_window(self):
        """Token i must not attend to token j > i even when both are inside the active window."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        torch.manual_seed(5)
        x, position_ids, active_mask = make_input(config, batch=1, seq=5)
        out_original = attn(x, position_ids, active_mask)

        x_modified = x.clone()
        x_modified[:, 3:4, :] = torch.randn_like(x_modified[:, 3:4, :])
        out_modified = attn(x_modified, position_ids, active_mask)

        torch.testing.assert_close(out_original[:, 2:3, :], out_modified[:, 2:3, :])


# ---------------------------------------------------------------------------
# Synthetic masked-continuation checks
# ---------------------------------------------------------------------------


class TestMaskedContinuationSemantics:
    def test_noncontiguous_live_token_inside_semantic_window_affects_output(self):
        """A live token outside the raw tail but inside the semantic window must matter."""
        torch.manual_seed(11)
        config = synthetic_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        x = synthetic_tokens(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
            hidden_size=config.hidden_size,
        )
        position_ids = torch.arange(7).unsqueeze(0)
        active_mask = torch.tensor(
            [[True, True, True, False, False, True, True]],
            dtype=torch.bool,
        )

        out_original = attn(x, position_ids, active_mask)

        # For the final live query, the semantic live sequence is
        # [0, 1, 2, 5, 6], so with window_size=4 the visible live keys are
        # positions [1, 2, 5, 6]. Position 1 is outside the raw tail but inside
        # the semantic active window.
        x_modified = x.clone()
        x_modified[:, 1:2, :] = x_modified[:, 1:2, :] + 500.0
        out_modified = attn(x_modified, position_ids, active_mask)

        assert not torch.allclose(
            out_original[:, 6:7, :],
            out_modified[:, 6:7, :],
        )

    def test_dead_current_chunk_token_does_not_affect_later_live_output(self):
        """A dead token in the current chunk must not affect a later live query."""
        torch.manual_seed(12)
        config = synthetic_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        x = synthetic_tokens(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
            hidden_size=config.hidden_size,
        )
        position_ids = torch.arange(7).unsqueeze(0)
        active_mask = torch.tensor(
            [[True, True, True, False, False, True, True]],
            dtype=torch.bool,
        )

        out_original = attn(x, position_ids, active_mask)

        x_modified = x.clone()
        x_modified[:, 3:4, :] = x_modified[:, 3:4, :] + 500.0
        out_modified = attn(x_modified, position_ids, active_mask)

        torch.testing.assert_close(
            out_original[:, 6:7, :],
            out_modified[:, 6:7, :],
        )

    def test_dead_query_row_does_not_respond_to_context_changes(self):
        """A dead query row should be insensitive to context changes."""
        torch.manual_seed(13)
        config = synthetic_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        x = synthetic_tokens(
            [[1.0, 2.0, 3.0, 4.0]],
            hidden_size=config.hidden_size,
        )
        position_ids = torch.arange(4).unsqueeze(0)
        active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

        out_original = attn(x, position_ids, active_mask)

        x_modified = x.clone()
        x_modified[:, 0:1, :] = x_modified[:, 0:1, :] + 500.0
        x_modified[:, 2:3, :] = x_modified[:, 2:3, :] - 500.0
        out_modified = attn(x_modified, position_ids, active_mask)

        torch.testing.assert_close(
            out_original[:, 1:2, :],
            out_modified[:, 1:2, :],
        )


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestSlidingWindowCache:
    def test_forward_runs_with_real_local_cache(self):
        """A forward pass with the real local cache should run and return sane output."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config)
        x, position_ids, active_mask = make_input(config, batch=2, seq=3)
        cache = make_cache(config, batch=2)

        out = attn(x, position_ids, active_mask, cache=cache)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_repeated_cached_forward_runs_with_real_local_cache(self):
        """Repeated decode calls with the real cache should continue to run."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        torch.manual_seed(2)
        x, position_ids, active_mask = make_input(config, batch=1, seq=5)
        cache = make_cache(config, batch=1)

        attn(x[:, :2, :], position_ids[:, :2], active_mask[:, :2], cache=cache)
        attn(x[:, 2:3, :], position_ids[:, 2:3], active_mask[:, 2:3], cache=cache)
        out_3 = attn(x[:, 3:4, :], position_ids[:, 3:4], active_mask[:, 3:4], cache=cache)
        out_4 = attn(x[:, 4:5, :], position_ids[:, 4:5], active_mask[:, 4:5], cache=cache)

        assert out_3.shape == (1, 1, config.hidden_size)
        assert out_4.shape == (1, 1, config.hidden_size)

    def test_cached_generation_matches_full_forward(self):
        """Cached all-live generation must match the corresponding full forward pass."""
        config = small_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        torch.manual_seed(3)
        x, position_ids, active_mask = make_input(config, batch=1, seq=6)

        out_full = attn(x, position_ids, active_mask)

        cache = make_cache(config, batch=1)
        attn(x[:, :3, :], position_ids[:, :3], active_mask[:, :3], cache=cache)
        out_3 = attn(x[:, 3:4, :], position_ids[:, 3:4], active_mask[:, 3:4], cache=cache)
        out_4 = attn(x[:, 4:5, :], position_ids[:, 4:5], active_mask[:, 4:5], cache=cache)
        out_5 = attn(x[:, 5:6, :], position_ids[:, 5:6], active_mask[:, 5:6], cache=cache)

        torch.testing.assert_close(out_3, out_full[:, 3:4, :])
        torch.testing.assert_close(out_4, out_full[:, 4:5, :])
        torch.testing.assert_close(out_5, out_full[:, 5:6, :])

    def test_dead_cached_token_does_not_affect_later_live_output(self):
        """A dead cached token must not affect a later live query."""
        torch.manual_seed(14)
        config = synthetic_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        prefill_x = synthetic_tokens(
            [[1.0, 2.0, 3.0, 4.0]],
            hidden_size=config.hidden_size,
        )
        prefill_pos = torch.arange(4).unsqueeze(0)
        prefill_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

        next_x = synthetic_tokens([[5.0]], hidden_size=config.hidden_size)
        next_pos = torch.tensor([[4]])
        next_mask = torch.tensor([[True]], dtype=torch.bool)

        cache_a = make_cache(config, batch=1)
        cache_b = make_cache(config, batch=1)

        attn(prefill_x, prefill_pos, prefill_mask, cache=cache_a)

        prefill_x_modified = prefill_x.clone()
        prefill_x_modified[:, 2:3, :] = prefill_x_modified[:, 2:3, :] + 500.0
        attn(prefill_x_modified, prefill_pos, prefill_mask, cache=cache_b)

        out_original = attn(next_x, next_pos, next_mask, cache=cache_a)
        out_modified = attn(next_x, next_pos, next_mask, cache=cache_b)

        torch.testing.assert_close(out_original, out_modified)

    def test_live_cached_token_inside_semantic_window_affects_later_live_output(self):
        """A live cached token outside the raw tail but inside the semantic window must matter."""
        torch.manual_seed(15)
        config = synthetic_config(window_size=4)
        attn = SlidingWindowAttention(config).eval()

        prefill_x = synthetic_tokens(
            [[1.0, 2.0, 3.0, 4.0]],
            hidden_size=config.hidden_size,
        )
        prefill_pos = torch.arange(4).unsqueeze(0)
        prefill_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

        next_x = synthetic_tokens([[5.0]], hidden_size=config.hidden_size)
        next_pos = torch.tensor([[4]])
        next_mask = torch.tensor([[True]], dtype=torch.bool)

        cache_a = make_cache(config, batch=1)
        cache_b = make_cache(config, batch=1)

        attn(prefill_x, prefill_pos, prefill_mask, cache=cache_a)

        # On the next step the returned frame is positions [0, 1, 2, 3, 4] with mask
        # [T, T, F, F, T]. Position 0 is outside the raw last-4 tail but inside the
        # semantic active window for the final live query.
        prefill_x_modified = prefill_x.clone()
        prefill_x_modified[:, 0:1, :] = prefill_x_modified[:, 0:1, :] + 500.0
        attn(prefill_x_modified, prefill_pos, prefill_mask, cache=cache_b)

        out_original = attn(next_x, next_pos, next_mask, cache=cache_a)
        out_modified = attn(next_x, next_pos, next_mask, cache=cache_b)

        assert not torch.allclose(out_original, out_modified)


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

        x, position_ids, active_mask = make_input(base_config, batch=1, seq=6)

        out_a = attn_a(x, position_ids, active_mask)
        out_b = attn_b(x, position_ids, active_mask)

        torch.testing.assert_close(out_a, out_b)