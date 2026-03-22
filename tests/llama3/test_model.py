"""Tests for Llama3Model (backbone).

Verifies the invariants documented in the plan for Unit 6. Llama3ForCausalLM
is tested separately in a later unit. These tests do not replicate decoder
layer or attention correctness — those are covered by their own unit tests.
"""

import torch
import pytest

from src.llama3.configuration import Llama3Config
from src.llama3.model import Llama3Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    """Minimal config for fast tests — small dimensions, shallow stack."""
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


@pytest.fixture
def model() -> Llama3Model:
    config = small_config()
    return Llama3Model(config).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_last_hidden_state_shape(self, model):
        """last_hidden_state must be (batch, seq_len, hidden_size)."""
        input_ids = torch.randint(0, 256, (2, 8))
        out = model(input_ids)
        assert out["last_hidden_state"].shape == (2, 8, model.config.hidden_size)

    def test_single_token_shape(self, model):
        """Shape is correct for a single-token input (typical generation step)."""
        input_ids = torch.randint(0, 256, (1, 1))
        out = model(input_ids)
        assert out["last_hidden_state"].shape == (1, 1, model.config.hidden_size)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_use_cache_true_returns_one_entry_per_layer(self, model):
        """past_key_values must contain one KVCache per decoder layer."""
        input_ids = torch.randint(0, 256, (1, 4))
        out = model(input_ids, use_cache=True)
        assert len(out["past_key_values"]) == model.config.num_hidden_layers

    def test_use_cache_false_returns_none(self, model):
        """past_key_values must be None when use_cache=False."""
        input_ids = torch.randint(0, 256, (1, 4))
        out = model(input_ids, use_cache=False)
        assert out["past_key_values"] is None

    def test_cached_generation_matches_full_forward(self, model):
        """Hidden state at each position via cached generation must match a full forward pass.

        Prefill positions 0..T-1 into the cache, then generate position T one token
        at a time. The last_hidden_state at position T must equal position T from a
        single full forward pass over the complete sequence.
        """
        torch.manual_seed(0)
        seq_len = 5
        input_ids = torch.randint(0, 256, (1, seq_len))

        # Full forward over the complete sequence.
        with torch.no_grad():
            full_out = model(input_ids, use_cache=False)
            full_hs = full_out["last_hidden_state"]  # (1, seq_len, hidden_size)

        # Prefill: forward over tokens 0..T-2, capturing the cache.
        with torch.no_grad():
            prefill_out = model(input_ids[:, :-1], use_cache=True)
            cache = prefill_out["past_key_values"]

        # Single-step: forward over the last token using the cache.
        with torch.no_grad():
            step_out = model(input_ids[:, -1:], past_key_values=cache, use_cache=True)
            step_hs = step_out["last_hidden_state"]  # (1, 1, hidden_size)

        torch.testing.assert_close(step_hs[:, 0, :], full_hs[:, -1, :])


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------

class TestHiddenStates:
    def test_output_hidden_states_false_returns_none(self, model):
        input_ids = torch.randint(0, 256, (1, 4))
        out = model(input_ids, output_hidden_states=False)
        assert out["hidden_states"] is None

    def test_output_hidden_states_true_correct_count(self, model):
        """Must return embedding output + one tensor per decoder layer."""
        input_ids = torch.randint(0, 256, (1, 4))
        out = model(input_ids, output_hidden_states=True)
        expected = model.config.num_hidden_layers + 1  # embedding + each layer
        assert len(out["hidden_states"]) == expected

    def test_output_hidden_states_correct_shape(self, model):
        """Every collected hidden state must be (batch, seq_len, hidden_size)."""
        input_ids = torch.randint(0, 256, (1, 4))
        out = model(input_ids, output_hidden_states=True)
        for hs in out["hidden_states"]:
            assert hs.shape == (1, 4, model.config.hidden_size)

    def test_config_output_hidden_states_respected(self):
        """output_hidden_states from config is used when not passed in forward."""
        config = small_config(output_hidden_states=True)
        m = Llama3Model(config).eval()
        input_ids = torch.randint(0, 256, (1, 3))
        out = m(input_ids)
        assert out["hidden_states"] is not None
