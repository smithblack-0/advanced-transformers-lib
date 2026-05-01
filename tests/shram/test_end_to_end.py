"""Integration and end-to-end tests for SHRAM.

Two test layers live here:

1. Integration tests (local, no network) — verify the assembled model works as a
   complete system. Three use cases: generatable (greedy), beam-search with cache,
   trainable, HF-loadable. These run without Hub access.

2. End-to-end tests (@pytest.mark.network) — the full researcher journey starting
   from the Hub. The starting point is always the Hub — never a locally constructed
   model. These replicate exactly what a researcher does when pulling and using
   this library.
"""

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.shram.model.configuration import ShramConfig
from src.shram.model.huggingface import ShramForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlexAttention does not support backward on CPU",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_sliding_window_heads=4,
        num_mosrah_heads=4,
        num_selected_heads=4,
        head_dim=16,
        window_size=8,
        rope_mode="main_sequence",
        training_sequence_length=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


@pytest.fixture
def model():
    return ShramForCausalLM(small_config()).eval()


# ---------------------------------------------------------------------------
# Integration — Generatable
# ---------------------------------------------------------------------------

class TestIntegrationGeneratable:
    def test_output_shape(self, model):
        """generate() must return (batch, input_len + max_new_tokens)."""
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
# Integration — Beam search (exercises reorder_cache)
# ---------------------------------------------------------------------------

class TestIntegrationBeamSearch:
    def test_cached_matches_uncached(self, model):
        """Beam search with use_cache=True must produce identical output to use_cache=False.

        With use_cache=False the model recomputes all key/values at every step —
        correct by construction. With use_cache=True, reorder_cache is called at
        each step to reorder the KV cache to match surviving beams. Any bug in
        reorder_cache causes the model to attend to incorrect history and produce
        different tokens, making the mismatch detectable here.
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
        m = ShramForCausalLM(small_config()).train()
        ids = torch.randint(0, 256, (1, 4))
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()

    def test_all_params_have_gradients(self):
        """Every trainable parameter must receive a gradient after backward()."""
        m = ShramForCausalLM(small_config()).train()
        ids = torch.randint(0, 256, (1, 4))
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in m.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"



# ---------------------------------------------------------------------------
# Hub constants
# ---------------------------------------------------------------------------

HUB_REPO = "smithblack-0/SHRAM"


# ---------------------------------------------------------------------------
# Shared Hub fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hub_config():
    """Download config from Hub once per module to avoid repeated network calls.

    After loading, asserts the config class was imported from HuggingFace's
    remote-code path (transformers_modules.*), not from a locally registered
    class. Without this check, a prior test calling AutoConfig.register() locally
    can cause this fixture to succeed via a local fallback even when the Hub is
    missing auto_map, producing false-positive network tests.
    """
    config = AutoConfig.from_pretrained(
        HUB_REPO,
        trust_remote_code=True,
        force_download=True,
    )
    module = type(config).__module__
    assert "transformers_modules" in module, (
        f"hub_config loaded a locally-registered class ({module}) instead of Hub "
        f"remote code. The Hub repo may be missing auto_map in config.json."
    )
    return config


# ---------------------------------------------------------------------------
# End-to-End — HuggingFace-loadable
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2ELoadable:
    """The config and model must load from the Hub via AutoClass."""

    def test_config_model_type(self, hub_config):
        """Config loaded from Hub must identify as the expected model type."""
        assert hub_config.model_type == "shram"

    def test_model_instantiates_from_hub_config(self, hub_config):
        """AutoModelForCausalLM.from_config must produce a ShramForCausalLM."""
        model = AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True)
        assert type(model).__name__ == "ShramForCausalLM"


# ---------------------------------------------------------------------------
# End-to-End — Tokenizer
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2ETokenizer:
    """The tokenizer must load from the Hub subfolder."""

    def test_tokenizer_loads(self):
        """AutoTokenizer.from_pretrained must succeed for the Hub repo."""
        tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
        assert tokenizer is not None

    def test_tokenizer_encodes_and_decodes(self):
        """Tokenizer must round-trip a simple string."""
        tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
        text = "Hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert text in decoded


# ---------------------------------------------------------------------------
# End-to-End — Generatable
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
# End-to-End — Trainable
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
        """Every trainable parameter must receive a gradient after backward()."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
