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

import shutil
import os
import torch
import torch._dynamo
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CompileConfig

from src.shram.model.configuration import ShramConfig
from src.shram.model.huggingface import ShramForCausalLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=256,
        embedding_width=64,
        mlp_width=128,
        num_decoder_layers=2,
        num_sliding_window_heads=4,
        num_mosrah_heads=4,
        num_selected_heads=4,
        head_dim=16,
        window_size=8,
        rope_mode="main_sequence",
        training_sequence_length=32,
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
# Integration — Compilable
# ---------------------------------------------------------------------------


class TestIntegrationCompilable:

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Training-path compilation requires CUDA; CPU compilation is not supported "
            "for the uncached (training) forward path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compile_uncached_forward(self):
        """torch.compile must succeed on the uncached forward path."""

        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()

        m = ShramForCausalLM(small_config()).cuda().eval()
        ids = torch.randint(0, 256, (1, 1)).cuda()
        m(ids, use_cache=False)
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        compiled(ids, use_cache=False)
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "CPU compilation only works for the cached inference path under specific "
            "conditions. Without _compile_all_devices=True, HuggingFace silently falls "
            "back to eager on CPU rather than raising, producing a false pass. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compile_cached_inference(self):
        """generate() with CompileConfig must complete on the cached inference path.

        Uses fullgraph=False so that remaining graph breaks do not hard-error —
        the test verifies the compiled path executes without crash. Upgrade to
        fullgraph=True once all graph breaks are resolved.

        _compile_all_devices=True is set so that if this test ever runs on CPU
        it fails loudly rather than silently falling back to eager.
        See https://github.com/pytorch/pytorch/issues/148752
        """
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        m = ShramForCausalLM(small_config()).cuda().eval()
        compile_config = CompileConfig(fullgraph=False, dynamic=True)
        compile_config._compile_all_devices = True
        ids = torch.randint(0, 256, (1, 4)).cuda()
        m.generate(ids, max_new_tokens=3, compile_config=compile_config)


# ---------------------------------------------------------------------------
# Hub constants
# ---------------------------------------------------------------------------

HUB_REPOS = {
    "main": "smithblack-0/SHRAM",
    "dev": "smithblack-0/SHRAM-dev",
}


# ---------------------------------------------------------------------------
# Shared Hub fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hub_repo(request):
    """Resolve the Hub repository name for the current --hub target."""
    return HUB_REPOS[request.config.getoption("--hub")]


@pytest.fixture(scope="module")
def hub_config(hub_repo):
    """Download config from Hub once per module to avoid repeated network calls.

    After loading, asserts the config class was imported from HuggingFace's
    remote-code path (transformers_modules.*), not from a locally registered
    class. Without this check, a prior test calling AutoConfig.register() locally
    can cause this fixture to succeed via a local fallback even when the Hub is
    missing auto_map, producing false-positive network tests.
    """
    config = AutoConfig.from_pretrained(
        hub_repo,
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

    def test_tokenizer_loads(self, hub_repo):
        """AutoTokenizer.from_pretrained must succeed for the Hub repo."""
        tokenizer = AutoTokenizer.from_pretrained(hub_repo)
        assert tokenizer is not None

    def test_tokenizer_encodes_and_decodes(self, hub_repo):
        """Tokenizer must round-trip a simple string."""
        tokenizer = AutoTokenizer.from_pretrained(hub_repo)
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
# End-to-End — Save/Load roundtrip via AutoClass
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestE2ESaveLoad:
    """Hub-loaded model and tokenizer must survive a save/load roundtrip via AutoClass.

    These tests specifically exercise AutoModelForCausalLM.from_pretrained and
    AutoTokenizer.from_pretrained with local_files_only=True — the untested path
    that exercises auto_map remote-code resolution against a locally saved checkpoint.
    The existing local save/load tests in test_huggingface.py use direct class
    instantiation and do not cover this path.
    """

    def test_model_save_load_roundtrip(self, hub_config, tmp_path) -> None:
        """All parameter values must survive Hub → save_pretrained → AutoModel.from_pretrained."""
        model = AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True)
        model.save_pretrained(tmp_path)

        loaded = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        for (name, p1), (_, p2) in zip(
            model.named_parameters(),
            loaded.named_parameters(),
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")

    def test_tokenizer_save_load_roundtrip(self, hub_repo, tmp_path) -> None:
        """Tokenizer must survive Hub → save_pretrained → AutoTokenizer.from_pretrained."""
        tokenizer = AutoTokenizer.from_pretrained(hub_repo)
        tokenizer.save_pretrained(tmp_path)

        loaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)

        text = "Hello world"
        ids_original = tokenizer.encode(text)
        ids_loaded = loaded.encode(text)
        assert ids_original == ids_loaded


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
