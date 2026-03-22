"""Tests for Llama3ForCausalLM.

Verifies the invariants documented in plan.md for Unit 7. Backbone correctness
is covered by test_model.py and is not replicated here. These tests focus on
the wrapper's own responsibilities: logit projection, loss, weight tying,
KV cache at the wrapper level, and the HF save/load round-trip.
"""

import torch
import pytest
from transformers import AutoModelForCausalLM

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

class TestOutputShape:
    def test_logits_shape(self, model):
        """logits must be (batch, seq_len, vocab_size)."""
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids)
        assert out["logits"].shape == (2, 8, model.config.vocab_size)

    def test_loss_none_without_labels(self, model):
        ids = torch.randint(0, 256, (1, 4))
        out = model(ids)
        assert out["loss"] is None


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
