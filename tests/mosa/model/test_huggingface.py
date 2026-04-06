"""Tests for MosaForCausalLM.

Verifies the invariants documented in plan.md for Unit 7. Backbone correctness
is covered by test_model.py and is not replicated here. These tests focus on
the wrapper's own responsibilities: logit projection, loss, weight tying,
KV cache at the wrapper level, and the HF save/load round-trip.
"""

import torch
import pytest
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.mosa.model.configuration import MosaConfig
from src.mosa.model.huggingface import MosaForCausalLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> MosaConfig:
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


@pytest.fixture
def model() -> MosaForCausalLM:
    return MosaForCausalLM(small_config()).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Return type and _init_weights
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_causal_lm_output_with_past(self, model):
        """forward() must return CausalLMOutputWithPast, not a plain dict."""
        out = model(torch.randint(0, 256, (1, 4)), use_cache=False)
        assert isinstance(out, CausalLMOutputWithPast)

    def test_attribute_access_works(self, model):
        """ModelOutput fields must be accessible as attributes."""
        out = model(torch.randint(0, 256, (1, 4)), use_cache=False)
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
        out = model(ids, use_cache=False)
        assert out["logits"].shape == (2, 8, model.config.vocab_size)

    def test_loss_none_without_labels(self, model):
        ids = torch.randint(0, 256, (1, 4))
        out = model(ids, use_cache=False)
        assert out.loss is None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def test_loss_is_scalar_when_labels_provided(self, model):
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids, labels=ids, use_cache=False)
        assert out["loss"].shape == ()

    def test_loss_is_positive(self, model):
        ids = torch.randint(0, 256, (1, 8))
        out = model(ids, labels=ids, use_cache=False)
        assert out["loss"].item() > 0

    def test_loss_ignores_minus_100(self, model):
        """Positions labelled -100 must be excluded from the loss.

        Masks all but a single label position. Manually computes cross-entropy
        on that one position and asserts the model's loss matches exactly.
        This verifies exclusion, not merely that the loss changes — the weaker
        assertion would pass even if -100 positions were included but re-weighted.
        """
        torch.manual_seed(7)
        ids = torch.randint(0, model.config.vocab_size, (1, 8))

        # Mask all shifted label positions to -100 except the last.
        # The shift means label position i corresponds to logits[:, i-1, :].
        # Keeping only labels[:, 7] means only logits[:, 6, :] contributes.
        labels = ids.clone()
        labels[:, 1:7] = -100  # mask positions 1..6; keep 7

        with torch.no_grad():
            out = model(ids, labels=labels, use_cache=False)
            logits = out.logits

        # Manual cross-entropy on the single contributing position.
        # shift: logits[:, :-1, :] predicts labels[:, 1:]. The surviving
        # position is shifted index 6 (original label index 7).
        expected_loss = torch.nn.functional.cross_entropy(
            logits[:, 6, :],
            ids[:, 7],
        )
        torch.testing.assert_close(out.loss, expected_loss)


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_tied_weights_share_storage(self):
        """When tie_word_embeddings=True, lm_head and embed_tokens share the same data."""
        m = MosaForCausalLM(small_config(tie_word_embeddings=True)).eval()
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()

    def test_untied_weights_are_independent(self):
        """When tie_word_embeddings=False, lm_head and embed_tokens are separate."""
        m = MosaForCausalLM(small_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight.data_ptr() != m.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestContractEnforcement:
    def test_attention_mask_raises(self, model):
        """Passing attention_mask must raise ValueError.

        This model does not support padding masks. Silent acceptance would produce
        wrong results for any caller who passes one expecting it to be applied.
        """
        ids = torch.randint(0, 256, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="attention_mask"):
            model(ids, attention_mask=mask)

    def test_use_cache_without_cache_position_raises(self, model):
        """use_cache=True without cache_position must raise immediately.

        cache_position is GenerationMixin's contract. Silent derivation would
        produce wrong position encodings and corrupt checkpoints. We crash instead.
        """
        ids = torch.randint(0, 256, (1, 4))
        with pytest.raises(ValueError, match="cache_position"):
            model(ids, use_cache=True)


class TestKVCache:
    def test_cached_generation_matches_full_forward(self, model):
        """Logits at the final position via cached generation must equal a full forward pass.

        Caching must not change what the model computes — only how efficiently.
        cache_position is provided explicitly as GenerationMixin would supply it.
        """
        torch.manual_seed(0)
        ids = torch.randint(0, 256, (1, 5))

        with torch.no_grad():
            full_logits = model(ids, use_cache=False)["logits"]

        with torch.no_grad():
            prefill = model(
                ids[:, :-1],
                use_cache=True,
                cache_position=torch.arange(4),
            )

        with torch.no_grad():
            step = model(
                ids[:, -1:],
                past_key_values=prefill["past_key_values"],
                use_cache=True,
                cache_position=torch.tensor([4]),
            )

        torch.testing.assert_close(
            step["logits"][:, 0, :], full_logits[:, -1, :]
        )

    def test_multi_token_reprompt_with_cache_matches_full_forward(self, model):
        """Re-prompting with multiple tokens into an existing cache must produce
        the same logits as a full forward pass over the complete sequence.

        This exercises the case where q_len > 1 and k_len > q_len — the pattern
        that was previously handled incorrectly (no causal mask applied).
        cache_position is provided explicitly as GenerationMixin would supply it.
        """
        torch.manual_seed(1)
        ids = torch.randint(0, model.config.vocab_size, (1, 6))

        with torch.no_grad():
            full_logits = model(ids, use_cache=False)["logits"]

        # Prefill 3 tokens, then re-prompt with the remaining 3.
        with torch.no_grad():
            prefill = model(
                ids[:, :3],
                use_cache=True,
                cache_position=torch.arange(3),
            )

        with torch.no_grad():
            reprompt = model(
                ids[:, 3:],
                past_key_values=prefill["past_key_values"],
                use_cache=True,
                cache_position=torch.arange(3, 6),
            )

        torch.testing.assert_close(reprompt["logits"], full_logits[:, 3:, :])


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

    def _make_cache(self, model: MosaForCausalLM, batch_size: int, seq_len: int):
        """Run a forward pass with a batch and return the resulting DynamicCache."""
        ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            out = model(ids, use_cache=True, cache_position=torch.arange(seq_len))
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

        Config resolution lives in MosaForCausalLM.forward(), which reads
        config.output_hidden_states as the default. Verified by constructing a model
        with output_hidden_states=True in the config and confirming hidden_states are
        returned without passing the flag at call time.
        """
        from src.mosa.model.configuration import MosaConfig
        config = MosaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=256,
            output_hidden_states=True,
        )
        m = MosaForCausalLM(config).eval()
        with torch.no_grad():
            out = m(torch.randint(0, 256, (1, 4)), use_cache=False)
        assert out.hidden_states is not None

    def test_config_use_cache_respected(self):
        """use_cache from config must be used when not passed explicitly."""
        from src.mosa.model.configuration import MosaConfig
        config = MosaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=256,
            use_cache=False,
        )
        m = MosaForCausalLM(config).eval()
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
        loaded = MosaForCausalLM.from_pretrained(tmp_path)

        for (name, p1), (_, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")

    def test_auto_model_from_config(self):
        """AutoModelForCausalLM.from_config must instantiate without error."""
        from transformers import AutoConfig
        from src.mosa.model.configuration import MosaConfig
        from src.mosa.model.huggingface import MosaForCausalLM

        AutoConfig.register("mosa_baseline", MosaConfig)
        AutoModelForCausalLM.register(MosaConfig, MosaForCausalLM)

        config = small_config()
        m = AutoModelForCausalLM.from_config(config)
        assert isinstance(m, MosaForCausalLM)
