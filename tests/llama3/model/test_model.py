"""Tests for Llama3Model.

Verifies the invariants documented in plan.md. Tests are scoped to the backbone
only — decoder layer and attention correctness are covered by their own unit tests
and are not replicated here.

The backbone accepts pre-embedded inputs (inputs_embeds), not token IDs. Tests
construct random float tensors of shape (batch, seq_len, hidden_size) directly,
which is the correct interface.

Llama3Model is a plain nn.Module. It returns a plain dict. No HF lifecycle
machinery (post_init, _init_weights, save/load) is present or tested here.
"""

import torch
import pytest
from transformers import DynamicCache

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
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dict(self, model):
        """forward() must return a plain dict, not a HF ModelOutput subclass."""
        out = model(random_embeds(1, 4, model.config.hidden_size))
        assert type(out) is dict

    def test_dict_has_expected_keys(self, model):
        """Output dict must contain exactly the documented keys."""
        out = model(random_embeds(1, 4, model.config.hidden_size))
        assert set(out.keys()) == {"last_hidden_state", "past_key_values", "hidden_states"}


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_with_cache_returns_populated_cache(self, model):
        """When a DynamicCache is provided, it must be returned populated with
        one entry per decoder layer.
        """
        embeds = random_embeds(1, 4, model.config.hidden_size)
        cache = DynamicCache()
        out = model(embeds, past_key_values=cache)
        assert out["past_key_values"] is cache
        assert len(out["past_key_values"]) == model.config.num_hidden_layers

    def test_without_cache_returns_none(self, model):
        """When no cache is provided, past_key_values must be None in the output."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds)
        assert out["past_key_values"] is None

    def test_cached_generation_matches_full_forward(self, model):
        """Hidden state at the final position via cached generation must equal
        the same position from a full forward pass over the complete sequence.

        This is the core correctness guarantee of the KV cache: caching must not
        change what the model computes, only how efficiently it computes it.
        """
        torch.manual_seed(0)
        embeds = random_embeds(1, 5, model.config.hidden_size)

        with torch.no_grad():
            full_hs = model(embeds)["last_hidden_state"]

        with torch.no_grad():
            prefill = model(embeds[:, :-1, :], past_key_values=DynamicCache())

        with torch.no_grad():
            step = model(
                embeds[:, -1:, :],
                past_key_values=prefill["past_key_values"],
            )

        torch.testing.assert_close(step["last_hidden_state"][:, 0, :], full_hs[:, -1, :])


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------

class TestHiddenStates:
    def test_output_hidden_states_false_returns_none(self, model):
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, output_hidden_states=False)
        assert out["hidden_states"] is None

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
