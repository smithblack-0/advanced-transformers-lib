"""Tests for MosaModel.

Verifies the invariants documented in plan.md. Tests are scoped to the backbone
only — decoder layer and attention correctness are covered by their own unit tests
and are not replicated here.

The backbone accepts pre-embedded inputs (inputs_embeds), not token IDs. Tests
construct random float tensors of shape (batch, seq_len, hidden_size) directly,
which is the correct interface.

MosaModel is a plain nn.Module. It returns a plain dict. No HF lifecycle
machinery (post_init, _init_weights, save/load) is present or tested here.
"""

import torch
import pytest
from transformers import DynamicCache

from src.mosa.model.configuration import MosaConfig
from src.mosa.model.model import MosaModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> MosaConfig:
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
    return MosaConfig(**defaults)


def random_embeds(batch: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    """Random float tensor standing in for pre-embedded inputs."""
    return torch.randn(batch, seq_len, hidden_size)


def pos_ids(batch: int, seq_len: int, offset: int = 0) -> torch.Tensor:
    """Absolute position ids of shape (batch, seq_len) starting from offset."""
    return torch.arange(offset, offset + seq_len).unsqueeze(0).expand(batch, -1)


def make_causal_mask(cache_position: torch.Tensor, k_len: int) -> torch.Tensor:
    """Build a boolean causal mask of shape (1, 1, q_len, k_len).

    True at (q, k) means query q may attend to key k. Required for any decode
    step where q_len < k_len — is_causal=True is only correct for square Q×K.
    """
    k_positions = torch.arange(k_len)
    return (k_positions[None, :] <= cache_position[:, None]).unsqueeze(0).unsqueeze(0)


@pytest.fixture
def model() -> MosaModel:
    return MosaModel(small_config()).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_last_hidden_state_shape(self, model):
        """last_hidden_state must be (batch, seq_len, hidden_size)."""
        embeds = random_embeds(2, 8, model.config.hidden_size)
        out = model(embeds, pos_ids(2, 8))
        assert out["last_hidden_state"].shape == (2, 8, model.config.hidden_size)

    def test_single_token_shape(self, model):
        """Shape is correct for a single-token input — the typical generation step."""
        embeds = random_embeds(1, 1, model.config.hidden_size)
        out = model(embeds, pos_ids(1, 1))
        assert out["last_hidden_state"].shape == (1, 1, model.config.hidden_size)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dict(self, model):
        """forward() must return a plain dict, not a HF ModelOutput subclass."""
        out = model(random_embeds(1, 4, model.config.hidden_size), pos_ids(1, 4))
        assert type(out) is dict

    def test_dict_has_expected_keys(self, model):
        """Output dict must contain exactly the documented keys."""
        out = model(random_embeds(1, 4, model.config.hidden_size), pos_ids(1, 4))
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
        out = model(embeds, pos_ids(1, 4), past_key_values=cache)
        assert out["past_key_values"] is cache
        assert len(out["past_key_values"]) == model.config.num_hidden_layers

    def test_without_cache_returns_none(self, model):
        """When no cache is provided, past_key_values must be None in the output."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, pos_ids(1, 4))
        assert out["past_key_values"] is None

    def test_cached_generation_matches_full_forward(self, model):
        """Hidden state at the final position via cached generation must equal
        the same position from a full forward pass over the complete sequence.

        This is the core correctness guarantee of the KV cache: caching must not
        change what the model computes, only how efficiently it computes it.

        The decode step requires an explicit causal mask: is_causal=True is only
        correct for square Q×K matrices (prefill). For a single-token decode step
        (q_len=1, k_len=5), PyTorch's built-in is_causal uses upper-left alignment
        and gives wrong results.
        """
        torch.manual_seed(0)
        seq = 5
        embeds = random_embeds(1, seq, model.config.hidden_size)

        with torch.no_grad():
            full_hs = model(embeds, pos_ids(1, seq))["last_hidden_state"]

        with torch.no_grad():
            prefill = model(embeds[:, :-1, :], pos_ids(1, seq - 1), past_key_values=DynamicCache())

        # The decode token sits at absolute position seq-1; k_len = seq after update.
        past_len = prefill["past_key_values"].get_seq_length(0)
        decode_mask = make_causal_mask(torch.tensor([past_len]), k_len=seq)

        with torch.no_grad():
            step = model(
                embeds[:, -1:, :],
                pos_ids(1, 1, offset=past_len),
                past_key_values=prefill["past_key_values"],
                causal_mask=decode_mask,
            )

        torch.testing.assert_close(step["last_hidden_state"][:, 0, :], full_hs[:, -1, :])


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------

class TestHiddenStates:
    def test_output_hidden_states_false_returns_none(self, model):
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, pos_ids(1, 4), output_hidden_states=False)
        assert out["hidden_states"] is None

    def test_output_hidden_states_true_correct_count(self, model):
        """Must return inputs_embeds plus one tensor per decoder layer."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, pos_ids(1, 4), output_hidden_states=True)
        assert len(out["hidden_states"]) == model.config.num_hidden_layers + 1

    def test_output_hidden_states_correct_shape(self, model):
        """Every collected hidden state must have shape (batch, seq_len, hidden_size)."""
        embeds = random_embeds(1, 4, model.config.hidden_size)
        out = model(embeds, pos_ids(1, 4), output_hidden_states=True)
        for hs in out["hidden_states"]:
            assert hs.shape == (1, 4, model.config.hidden_size)
