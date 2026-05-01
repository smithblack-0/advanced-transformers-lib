"""Integration and end-to-end tests for the Llama 3 baseline.

Two test layers live here:

1. Integration tests (local, no network) — verify the assembled model works as a
   complete system before the Hub is involved. Three use cases: Generatable,
   Trainable, HF-loadable. Any bug discovered here is resolved as a new blocker
   before Unit 10 begins.

2. End-to-end tests (@pytest.mark.network) — the full user journey starting from
   the Hub. Unit 10. The starting point is always the Hub — never a locally
   constructed model. These replicate exactly what a researcher does when pulling
   and using this library.
"""

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM

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
# Integration — Generatable
# ---------------------------------------------------------------------------

class TestIntegrationGeneratable:
    def test_output_shape(self, model):
        """generate() must return (batch, input_len + max_new_tokens).

        eos_token_id is None in our config, so there is no early stopping —
        generation always runs to exactly max_new_tokens. The shape is deterministic.
        """
        ids = torch.randint(0, 256, (1, 4))
        out = model.generate(ids, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_valid_token_ids(self, model):
        """All generated token IDs must be in [0, vocab_size)."""
        ids = torch.randint(0, 256, (1, 4))
        out = model.generate(ids, max_new_tokens=5)
        assert (out >= 0).all() and (out < model.config.vocab_size).all()

    def test_determinism(self, model):
        """Greedy decoding must be deterministic: identical input produces identical output."""
        ids = torch.randint(0, 256, (1, 4))
        out1 = model.generate(ids, max_new_tokens=5)
        out2 = model.generate(ids, max_new_tokens=5)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Integration — Beam search (_reorder_cache)
# ---------------------------------------------------------------------------

class TestIntegrationBeamSearch:
    def test_cached_matches_uncached(self, model):
        """Beam search with use_cache=True must produce identical output to use_cache=False.

        With use_cache=False the model recomputes all key/values at every step —
        correct by construction, no cache involved. With use_cache=True,
        _reorder_cache is called at each step to reorder the KV cache to match
        the surviving beams. Any bug in _reorder_cache (wrong dimension, inverted
        index) causes the model to attend to incorrect history and produce different
        tokens, making the mismatch detectable here.
        """
        ids = torch.randint(0, 256, (1, 4))
        out_cached = model.generate(ids, max_new_tokens=5, num_beams=2, use_cache=True)
        out_uncached = model.generate(ids, max_new_tokens=5, num_beams=2, use_cache=False)
        assert torch.equal(out_cached, out_uncached)


# ---------------------------------------------------------------------------
# Integration — Trainable
# ---------------------------------------------------------------------------

class TestIntegrationTrainable:
    def test_loss_backward_runs(self):
        """loss.backward() must complete without error on the full assembled model."""
        m = Llama3ForCausalLM(small_config()).train()
        ids = torch.randint(0, 256, (1, 4))
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()

    def test_all_params_have_gradients(self):
        """Every trainable parameter must receive a gradient after backward().

        With tie_word_embeddings=False (our default), embed_tokens and lm_head are
        independent tensors and both must have gradients. The loss path runs through
        the full stack: embed_tokens → decoder layers → lm_head → cross-entropy.
        """
        m = Llama3ForCausalLM(small_config()).train()
        ids = torch.randint(0, 256, (1, 4))
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in m.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"



# ---------------------------------------------------------------------------
# Hub constant (Unit 10)
# ---------------------------------------------------------------------------

HUB_REPO = "smithblack-0/llama3_baseline"


# ---------------------------------------------------------------------------
# Shared Hub fixture (Unit 10)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hub_config():
    """Download config from Hub once per module to avoid repeated network calls.

    Module scope is correct here: the config object is read-only and safe to share
    across all Unit 10 test classes. Each class constructs its own model instance
    from this config, so there is no shared mutable state between classes.

    After loading, the fixture asserts that the config class was imported from
    HuggingFace's remote-code path (transformers_modules.*), not from a locally
    registered class. HF's trust_remote_code mechanism imports Hub code under
    transformers_modules; local AutoClass registrations import from src.*. Without
    this check, a prior test that calls AutoConfig.register() locally can cause this
    fixture to succeed via the local fallback even when the Hub is missing auto_map,
    producing false-positive network tests.
    """
    config = AutoConfig.from_pretrained(HUB_REPO, trust_remote_code=True)
    module = type(config).__module__
    assert "transformers_modules" in module, (
        f"hub_config loaded a locally-registered class ({module}) instead of Hub "
        f"remote code. The Hub repo may be missing auto_map in config.json."
    )
    return config


# ---------------------------------------------------------------------------
# End-to-End — HuggingFace-loadable (Unit 10)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2ELoadable:
    """The config and model must load from the Hub via AutoClass.

    Verifies the Hub distribution path itself: that the files on the Hub are
    correct, trust_remote_code finds the right classes, and the config has the
    expected model_type. Local AutoClass registration is covered by the integration
    tests above and is not replicated here.

    The Hub-downloaded class and the local src class have different Python identities
    (different module paths under transformers_modules), so type checks use the class
    name rather than isinstance.
    """

    def test_config_model_type(self, hub_config):
        """Config loaded from Hub must identify as the expected model type."""
        assert hub_config.model_type == "llama3_baseline"

    def test_model_instantiates_from_hub_config(self, hub_config):
        """AutoModelForCausalLM.from_config must produce a Llama3ForCausalLM."""
        model = AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True)
        assert type(model).__name__ == "Llama3ForCausalLM"


# ---------------------------------------------------------------------------
# End-to-End — Generatable (Unit 10)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2EGeneratable:
    """A Hub-loaded model must produce valid token sequences via generate()."""

    @pytest.fixture
    def model(self, hub_config):
        return AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True).eval()

    def test_output_shape(self, model):
        """generate() must return (batch, input_len + max_new_tokens)."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model.generate(ids, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_valid_token_ids(self, model):
        """All generated token IDs must be in [0, vocab_size)."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model.generate(ids, max_new_tokens=5)
        assert (out >= 0).all() and (out < model.config.vocab_size).all()


# ---------------------------------------------------------------------------
# End-to-End — Trainable (Unit 10)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2ETrainable:
    """A Hub-loaded model must support a complete training step."""

    @pytest.fixture
    def model(self, hub_config):
        return AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True).train()

    def test_loss_backward_runs(self, model):
        """loss.backward() must complete without error on a Hub-loaded model."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()

    def test_all_params_have_gradients(self, model):
        """Every trainable parameter must receive a gradient after backward().

        The loss path runs through the full stack: embed_tokens → decoder layers
        → lm_head → cross-entropy. A disconnected parameter would indicate a wiring
        bug introduced during Hub distribution (e.g. a missing relative import that
        caused silent fallback to a stub).
        """
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
