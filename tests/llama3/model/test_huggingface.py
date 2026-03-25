"""Tests for Llama3ForCausalLM.

Verifies the invariants documented in plan.md for Unit 7. Backbone correctness
is covered by test_model.py and is not replicated here. These tests focus on
the wrapper's own responsibilities: logit projection, loss, weight tying,
KV cache at the wrapper level, and the HF save/load round-trip.
"""

import torch
import pytest
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.llama3.model.configuration import Llama3Config
from src.llama3.model.huggingface import Llama3ForCausalLM


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
        vocab_size=256,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


@pytest.fixture
def model() -> Llama3ForCausalLM:
    return Llama3ForCausalLM(small_config()).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Return type and _init_weights
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_causal_lm_output_with_past(self, model):
        """forward() must return CausalLMOutputWithPast, not a plain dict."""
        out = model(torch.randint(0, 256, (1, 4)))
        assert isinstance(out, CausalLMOutputWithPast)

    def test_attribute_access_works(self, model):
        """ModelOutput fields must be accessible as attributes."""
        out = model(torch.randint(0, 256, (1, 4)))
        assert out.logits is not None

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


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_logits_shape(self, model):
        """logits must be (batch, seq_len, vocab_size)."""
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids)
        assert out["logits"].shape == (2, 8, model.config.vocab_size)

    def test_loss_none_without_labels(self, model):
        ids = torch.randint(0, 256, (1, 4))
        out = model(ids)
        assert out.loss is None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def test_loss_is_scalar_when_labels_provided(self, model):
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids, labels=ids)
        assert out["loss"].shape == ()

    def test_loss_is_positive(self, model):
        ids = torch.randint(0, 256, (1, 8))
        out = model(ids, labels=ids)
        assert out["loss"].item() > 0

    def test_loss_ignores_minus_100(self, model):
        """Positions labelled -100 must be excluded from the loss."""
        ids = torch.randint(0, 256, (1, 8))
        labels_all = ids.clone()
        labels_masked = ids.clone()
        labels_masked[:, 1:4] = -100  # mask some positions

        out_all = model(ids, labels=labels_all)
        out_masked = model(ids, labels=labels_masked)

        # Loss values differ because different positions contribute.
        assert not torch.isclose(out_all["loss"], out_masked["loss"])


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_tied_weights_share_storage(self):
        """When tie_word_embeddings=True, lm_head and embed_tokens share the same data."""
        m = Llama3ForCausalLM(small_config(tie_word_embeddings=True)).eval()
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()

    def test_untied_weights_are_independent(self):
        """When tie_word_embeddings=False, lm_head and embed_tokens are separate."""
        m = Llama3ForCausalLM(small_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight.data_ptr() != m.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_cached_generation_matches_full_forward(self, model):
        """Logits at the final position via cached generation must equal a full forward pass.

        Caching must not change what the model computes — only how efficiently.
        """
        torch.manual_seed(0)
        ids = torch.randint(0, 256, (1, 5))

        with torch.no_grad():
            full_logits = model(ids, use_cache=False)["logits"]

        with torch.no_grad():
            prefill = model(ids[:, :-1], use_cache=True)

        with torch.no_grad():
            step = model(
                ids[:, -1:],
                past_key_values=prefill["past_key_values"],
                use_cache=True,
            )

        torch.testing.assert_close(
            step["logits"][:, 0, :], full_logits[:, -1, :]
        )


# ---------------------------------------------------------------------------
# _reorder_cache
# ---------------------------------------------------------------------------

class TestReorderCache:
    """Unit tests for _reorder_cache.

    The correctness criterion: after reordering, cache entry i must contain
    the tensors that were at position beam_idx[i] in the original cache.

    DynamicCache.reorder_cache() modifies the cache in place, so snapshots
    must be taken before calling _reorder_cache.
    """

    def _make_cache(self, model: Llama3ForCausalLM, batch_size: int, seq_len: int):
        """Run a forward pass with a batch and return the resulting DynamicCache."""
        ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            out = model(ids, use_cache=True)
        return out.past_key_values

    def test_reorder_swaps_entries(self, model):
        """beam_idx=[1, 0] must swap the two batch entries in every K and V tensor."""
        cache = self._make_cache(model, batch_size=2, seq_len=4)

        # Snapshot before reorder — reorder_cache modifies in place.
        orig_keys = [layer.keys.clone() for layer in cache.layers]
        orig_vals = [layer.values.clone() for layer in cache.layers]

        beam_idx = torch.tensor([1, 0])
        model._reorder_cache(cache, beam_idx)

        for orig_k, layer in zip(orig_keys, cache.layers):
            torch.testing.assert_close(layer.keys[0], orig_k[1])
            torch.testing.assert_close(layer.keys[1], orig_k[0])
        for orig_v, layer in zip(orig_vals, cache.layers):
            torch.testing.assert_close(layer.values[0], orig_v[1])
            torch.testing.assert_close(layer.values[1], orig_v[0])

    def test_reorder_copies_winning_beam(self, model):
        """beam_idx=[0, 0] must copy entry 0 into both slots (beam collapse)."""
        cache = self._make_cache(model, batch_size=2, seq_len=4)

        orig_keys = [layer.keys.clone() for layer in cache.layers]

        beam_idx = torch.tensor([0, 0])
        model._reorder_cache(cache, beam_idx)

        for orig_k, layer in zip(orig_keys, cache.layers):
            torch.testing.assert_close(layer.keys[0], orig_k[0])
            torch.testing.assert_close(layer.keys[1], orig_k[0])

    def test_reorder_preserves_structure(self, model):
        """Output must be the same cache object with unchanged layer count and shapes."""
        cache = self._make_cache(model, batch_size=2, seq_len=4)
        num_layers = len(cache.layers)
        orig_shapes_k = [layer.keys.shape for layer in cache.layers]
        orig_shapes_v = [layer.values.shape for layer in cache.layers]

        beam_idx = torch.tensor([1, 0])
        returned = model._reorder_cache(cache, beam_idx)

        # Must return the same object (modified in place, not a copy).
        assert returned is cache
        assert len(cache.layers) == num_layers
        for shape, layer in zip(orig_shapes_k, cache.layers):
            assert layer.keys.shape == shape
        for shape, layer in zip(orig_shapes_v, cache.layers):
            assert layer.values.shape == shape


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_config_output_hidden_states_respected(self):
        """output_hidden_states from config must be used when not passed explicitly.

        Config resolution lives in Llama3ForCausalLM.forward(), which reads
        config.output_hidden_states as the default. Verified by constructing a model
        with output_hidden_states=True in the config and confirming hidden_states are
        returned without passing the flag at call time.
        """
        from src.llama3.model.configuration import Llama3Config
        config = Llama3Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=256,
            output_hidden_states=True,
        )
        m = Llama3ForCausalLM(config).eval()
        with torch.no_grad():
            out = m(torch.randint(0, 256, (1, 4)))
        assert out.hidden_states is not None

    def test_config_use_cache_respected(self):
        """use_cache from config must be used when not passed explicitly."""
        from src.llama3.model.configuration import Llama3Config
        config = Llama3Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=256,
            use_cache=False,
        )
        m = Llama3ForCausalLM(config).eval()
        with torch.no_grad():
            out = m(torch.randint(0, 256, (1, 4)))
        assert out.past_key_values is None


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_pretrained_round_trip(self, model, tmp_path):
        """All weights must be identical after save_pretrained / from_pretrained."""
        model.save_pretrained(tmp_path)
        loaded = Llama3ForCausalLM.from_pretrained(tmp_path)

        for (name, p1), (_, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")

    def test_auto_model_from_config(self):
        """AutoModelForCausalLM.from_config must instantiate without error."""
        from transformers import AutoConfig
        from src.llama3.model.configuration import Llama3Config
        from src.llama3.model.huggingface import Llama3ForCausalLM

        AutoConfig.register("llama3_baseline", Llama3Config)
        AutoModelForCausalLM.register(Llama3Config, Llama3ForCausalLM)

        config = small_config()
        m = AutoModelForCausalLM.from_config(config)
        assert isinstance(m, Llama3ForCausalLM)
