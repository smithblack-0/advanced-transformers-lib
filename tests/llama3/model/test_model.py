"""Tests for Llama3Model.

Verifies the invariants documented in plan.md for Unit 6. Tests are scoped to
the backbone only — decoder layer and attention correctness are covered by their
own unit tests and are not replicated here.

The backbone accepts pre-embedded inputs (inputs_embeds), not token IDs. Tests
construct random float tensors of shape (batch, seq_len, hidden_size) directly,
which is the correct interface.
"""

import torch
import pytest
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.llama3.model.configuration import Llama3Config
from src.llama3.model.model import Llama3Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    """Minimal-dimension config for fast tests."""
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


def random_embeds(batch: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    """Random float tensor standing in for pre-embedded inputs."""
    return torch.randn(batch, seq_len, hidden_size)


@pytest.fixture
def model() -> Llama3Model:
    return Llama3Model(small_config()).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_last_hidden_state_shape(self, model):
        """last_hidden_state must be (batch, seq_len, hidden_size)."""
        embeds = random_embeds(2, 8, model.config.hidden_size)
        out = model(embeds)
        assert out["last_hidden_state"].shape == (2, 8, model.config.hidden_size)

    def test_single_token_shape(self, model):
        """Shape is correct for a single-token input — the typical generation step."""
        embeds = random_embeds(1, 1, model.config.hidden_size)
        out = model(embeds)
        assert out["last_hidden_state"].shape == (1, 1, model.config.hidden_size)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_use_cache_true_returns_one_entry_per_layer(self, model):
        """past_key_values must contain one KVCache per decoder layer."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, use_cache=True)
        assert len(out["past_key_values"]) == model.config.num_hidden_layers

    def test_use_cache_false_returns_none(self, model):
        """past_key_values must be None when use_cache=False."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, use_cache=False)
        assert out.past_key_values is None

    def test_cached_generation_matches_full_forward(self, model):
        """Hidden state at the final position via cached generation must equal
        the same position from a full forward pass over the complete sequence.

        This is the core correctness guarantee of the KV cache: caching must not
        change what the model computes, only how efficiently it computes it.
        """
        torch.manual_seed(0)
        embeds = random_embeds(1, 5, model.config.hidden_size)

        with torch.no_grad():
            full_hs = model(embeds, use_cache=False)["last_hidden_state"]

        with torch.no_grad():
            prefill = model(embeds[:, :-1, :], use_cache=True)

        with torch.no_grad():
            step = model(
                embeds[:, -1:, :],
                past_key_values=prefill["past_key_values"],
                use_cache=True,
            )

        torch.testing.assert_close(step["last_hidden_state"][:, 0, :], full_hs[:, -1, :])


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------

class TestHiddenStates:
    def test_output_hidden_states_false_returns_none(self, model):
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, output_hidden_states=False)
        assert out.hidden_states is None

    def test_output_hidden_states_true_correct_count(self, model):
        """Must return inputs_embeds plus one tensor per decoder layer."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, output_hidden_states=True)
        assert len(out["hidden_states"]) == model.config.num_hidden_layers + 1

    def test_output_hidden_states_correct_shape(self, model):
        """Every collected hidden state must have shape (batch, seq_len, hidden_size)."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, output_hidden_states=True)
        for hs in out["hidden_states"]:
            assert hs.shape == (1, 4, model.config.hidden_size)

    def test_config_output_hidden_states_respected(self):
        """output_hidden_states from config is used when not passed explicitly."""
        m = Llama3Model(small_config(output_hidden_states=True)).eval()
        out = m(random_embeds(1, 3, 64))
        assert out["hidden_states"] is not None


# ---------------------------------------------------------------------------
# Return type and _init_weights
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_base_model_output_with_past(self, model):
        """forward() must return BaseModelOutputWithPast, not a plain dict."""
        out = model(random_embeds(1, 4, model.config.hidden_size))
        assert isinstance(out, BaseModelOutputWithPast)

    def test_attribute_access_works(self, model):
        """ModelOutput fields must be accessible as attributes."""
        out = model(random_embeds(1, 4, model.config.hidden_size))
        _ = out.last_hidden_state
        assert out.last_hidden_state is not None

    def test_init_weights_is_noop(self, model):
        """_init_weights must not modify weights — PyTorch constructor defaults stand.

        HF's default _init_weights reinitialises all weights with normal(0, 0.02).
        Our override suppresses this. Verified by calling _init_weights on a linear
        module and confirming weights are unchanged.
        """
        import torch.nn as nn
        layer = nn.Linear(64, 64, bias=False)
        original = layer.weight.data.clone()
        model._init_weights(layer)
        torch.testing.assert_close(layer.weight.data, original)
