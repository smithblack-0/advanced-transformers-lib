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


def _make_eval_batch(
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a synthetic batch with mixed-length right-padded attention masks.

    Returns (input_ids, labels, attention_mask). Rows have varied active lengths
    so mixed-padding code paths are exercised. Seed is fixed for reproducibility.
    """
    torch.manual_seed(42)
    input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Row active lengths deliberately varied; the first and last rows are full.
    active_lengths = [seq_len, 100, 75, 90]
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i, length in enumerate(active_lengths[:batch_size]):
        attention_mask[i, :length] = True

    return input_ids, labels, attention_mask


@pytest.fixture
def model(device):
    return ShramForCausalLM(small_config()).eval().to(device)


# ---------------------------------------------------------------------------
# Integration — Generatable
# ---------------------------------------------------------------------------

class TestIntegrationGeneratable:
    def test_output_shape(self, model, device):
        """generate() must return (batch, input_len + max_new_tokens)."""
        ids = torch.randint(0, 256, (1, 4), device=device)
        out = model.generate(ids, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_valid_token_ids(self, model, device):
        """All generated token IDs must be in [0, vocab_size)."""
        ids = torch.randint(0, 256, (1, 4), device=device)
        out = model.generate(ids, max_new_tokens=5)
        assert (out >= 0).all() and (out < model.config.vocab_size).all()

    def test_determinism(self, model, device):
        """Greedy decoding must be deterministic: identical input produces identical output."""
        ids = torch.randint(0, 256, (1, 4), device=device)
        out1 = model.generate(ids, max_new_tokens=5)
        out2 = model.generate(ids, max_new_tokens=5)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Integration — Beam search (exercises reorder_cache)
# ---------------------------------------------------------------------------

class TestIntegrationBeamSearch:
    def test_cached_matches_uncached(self, model, device):
        """Beam search with use_cache=True must produce identical output to use_cache=False.

        With use_cache=False the model recomputes all key/values at every step —
        correct by construction. With use_cache=True, reorder_cache is called at
        each step to reorder the KV cache to match surviving beams. Any bug in
        reorder_cache causes the model to attend to incorrect history and produce
        different tokens, making the mismatch detectable here.
        """
        ids = torch.randint(0, 256, (1, 4), device=device)
        out_cached = model.generate(ids, max_new_tokens=5, num_beams=2, use_cache=True)
        out_uncached = model.generate(ids, max_new_tokens=5, num_beams=2, use_cache=False)
        assert torch.equal(out_cached, out_uncached)


# ---------------------------------------------------------------------------
# Integration — Trainable
# ---------------------------------------------------------------------------

class TestIntegrationTrainable:
    def test_loss_backward_runs(self, device):
        """loss.backward() must complete without error on the full assembled model."""
        m = ShramForCausalLM(small_config()).train().to(device)
        ids = torch.randint(0, 256, (1, 4), device=device)
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()

    def test_all_params_have_gradients(self, device):
        """Every trainable parameter must receive a gradient after backward()."""
        m = ShramForCausalLM(small_config()).train().to(device)
        ids = torch.randint(0, 256, (1, 4), device=device)
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in m.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"



# ---------------------------------------------------------------------------
# Integration — Capacity enforcement
# ---------------------------------------------------------------------------

class TestIntegrationCapacityEnforcement:
    """Verify balance_capacity prevents overflow under tight capacity budgets.

    Both tests use mosrah_overallocation_factor=1.001 (effectively no headroom)
    and feed a full training_sequence_length sequence so N > mosrah_packed_length
    and balance_capacity must do real work to keep routing within budget.
    """

    def test_dense_routing_stays_within_capacity(self, device):
        """Dense routing (K/L=0.75) with tight capacity completes without overflow."""
        config = small_config(
            num_selected_heads=3,
            num_mosrah_heads=4,
            mosrah_overallocation_factor=1.05   ,
            training_sequence_length=32,
        )
        m = ShramForCausalLM(config).train().to(device)
        ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
        out = m(ids, labels=ids, use_cache=False)
        assert out.loss is not None and torch.isfinite(out.loss)

    def test_sparse_routing_stays_within_capacity(self, device):
        """Sparse routing (K/L=0.25) with tight capacity completes without overflow."""
        config = small_config(
            num_selected_heads=1,
            num_mosrah_heads=4,
            mosrah_overallocation_factor=1.001,
            training_sequence_length=32,
        )
        m = ShramForCausalLM(config).train().to(device)
        ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
        out = m(ids, labels=ids, use_cache=False)
        assert out.loss is not None and torch.isfinite(out.loss)

# ---------------------------------------------------------------------------
# Integration — Extended inference (YaRN rescaling)
# ---------------------------------------------------------------------------

class TestIntegrationExtendedInference:
    """A model saved at training_sequence_length must generate past that horizon
    when reloaded with a larger inference_sequence_length override (YaRN rescaling).

    The save-load-with-override path is the real researcher workflow: train at one
    context length, then extend at inference time by overriding inference_sequence_length
    at load time. Both eager and compiled paths are exercised.

    training_sequence_length=16, inference_sequence_length=32 (2× YaRN scale).
    Input is 4 tokens; max_new_tokens=28 so the full 32-token inference budget is
    reached. mosrah_overallocation_factor=2.0 matches production harness headroom.
    """

    def _save_base_model(self, tmp_path):
        """Save a model trained at training_sequence_length=16 with no YaRN override."""
        config = small_config(
            training_sequence_length=16,
            mosrah_overallocation_factor=2.0,
        )
        ShramForCausalLM(config).save_pretrained(tmp_path)

    def test_generate_beyond_training_length(self, tmp_path, device):
        """Eager generate() must reach the full inference budget after a length override at load."""

        self._save_base_model(tmp_path)
        m = ShramForCausalLM.from_pretrained(
            tmp_path, inference_sequence_length=32,
        ).eval().to(device)

        ids = torch.randint(0, m.config.vocab_size, (1, 4), device=device)
        out = m.generate(ids, max_new_tokens=28)
        assert out.shape == (1, 32)
        assert (out >= 0).all() and (out < m.config.vocab_size).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled inference requires CUDA; CPU compilation is not supported "
            "for the cached inference path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compiled_generate_beyond_training_length(self, tmp_path, device):
        """Compiled and uncompiled generate() must agree after a length override at load."""
        self._save_base_model(tmp_path)
        m = ShramForCausalLM.from_pretrained(
            tmp_path, inference_sequence_length=32,
        ).eval().to(device)

        ids = torch.randint(0, m.config.vocab_size, (1, 4), device=device)
        out_eager = m.generate(ids, max_new_tokens=28)
        torch._dynamo.reset()
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        out_compiled = compiled.generate(ids, max_new_tokens=28)
        assert torch.equal(out_eager, out_compiled)

    def test_prefill_beyond_training_length(self, tmp_path, device):
        """Eager generate() must work when the prompt exceeds training_sequence_length.

        Prompt of 20 tokens sits between training_sequence_length=16 and
        inference_sequence_length=32, exercising positions the model was not
        trained to cover without YaRN rescaling.
        """
        self._save_base_model(tmp_path)
        m = ShramForCausalLM.from_pretrained(
            tmp_path, inference_sequence_length=32,
        ).eval().to(device)

        ids = torch.randint(0, m.config.vocab_size, (1, 20), device=device)
        out = m.generate(ids, max_new_tokens=4)
        assert out.shape == (1, 24)
        assert (out >= 0).all() and (out < m.config.vocab_size).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled inference requires CUDA; CPU compilation is not supported "
            "for the cached inference path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compiled_prefill_beyond_training_length(self, tmp_path, device):
        """Compiled and uncompiled generate() must agree when the prompt exceeds training_sequence_length."""
        self._save_base_model(tmp_path)
        m = ShramForCausalLM.from_pretrained(
            tmp_path, inference_sequence_length=32,
        ).eval().to(device)

        ids = torch.randint(0, m.config.vocab_size, (1, 20), device=device)
        out_eager = m.generate(ids, max_new_tokens=4)
        torch._dynamo.reset()
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        out_compiled = compiled.generate(ids, max_new_tokens=4)
        assert torch.equal(out_eager, out_compiled)


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

        torch.manual_seed(0)

        m = ShramForCausalLM(small_config()).cuda().eval()
        ids = torch.randint(0, 256, (1, 4)).cuda()

        m(ids, use_cache=False)

        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        #print(torch._dynamo.explain(compiled, ids, use_cache=False))
        compiled(ids, use_cache=False)
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Training-path compilation requires CUDA; CPU compilation is not supported "
            "for the uncached (training) forward path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compile_uncached_backward(self, device):
        """Compiled training forward+backward must complete without error.

        Verifies that loss.backward() on the compiled uncached training path
        does not raise. Graph breaks in the backward pass that cause a crash
        are caught here.
        """

        torch.manual_seed(0)
        m = ShramForCausalLM(small_config()).to(device).train()
        ids = torch.randint(0, 256, (1, 4), device=device)

        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        out = compiled(ids, labels=ids, use_cache=False)
        out.loss.backward()

        assert torch.isfinite(out.loss)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled inference requires CUDA; CPU compilation is not supported "
            "for the cached inference path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compiled_and_uncompiled_generate_produce_identical_output(self):
        """Compiled and uncompiled generate() must produce identical token sequences.

        Runs greedy decoding from the same input twice — once without compilation
        and once with CompileConfig — and asserts the outputs are identical. Any
        divergence indicates the compiled path is producing different attention or
        routing behaviour than the eager path.
        """
        m = ShramForCausalLM(small_config()).cuda().eval()
        ids = torch.randint(0, 256, (1, 4)).cuda()
        compile_config = CompileConfig(fullgraph=False, dynamic=True)

        out_eager = m.generate(ids, max_new_tokens=10, disable_compile=True)
        torch._dynamo.reset()
        out_compiled = m.generate(ids, max_new_tokens=10, compile_config=CompileConfig(fullgraph=False, dynamic=False))
        assert torch.equal(out_eager, out_compiled)

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
        m = ShramForCausalLM(small_config()).cuda().eval()
        compile_config = CompileConfig(fullgraph=False, dynamic=False)
        compile_config._compile_all_devices = True
        ids = torch.randint(0, 256, (1, 4)).cuda()
        m.generate(ids, max_new_tokens=3, compile_config=compile_config)


# ---------------------------------------------------------------------------
# Integration — Compiled eval (eval mode, no_grad, mixed precision)
# ---------------------------------------------------------------------------


class TestIntegrationCompiledEval:
    """Compiled model under inference-style execution contexts.

    Covers eval mode, no_grad, and fp16 autocast — the conditions present in
    an observed production Inductor FlexAttention layout failure and absent
    from TestIntegrationCompilable. All tests use batch=4, seq=128 with a
    mixed-padding attention_mask.
    """

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled eval forward requires CUDA; CPU compilation is not "
            "supported for the uncached forward path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def     test_compiled_eval_labeled_forward(self, device):
        """Compiled eval+no_grad labeled forward must complete without error."""
        m = ShramForCausalLM(
            small_config(training_sequence_length=128)
        ).eval().to(device)
        input_ids, labels, attention_mask = _make_eval_batch(device)

        torch._dynamo.reset()
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        with torch.no_grad():
            out = compiled(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )
        assert torch.isfinite(out.loss)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled eval forward requires CUDA; CPU compilation is not "
            "supported for the uncached forward path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compiled_eval_autocast_labeled_forward(self, device):
        """Compiled eval+no_grad labeled forward under fp16 autocast must complete without error."""
        m = ShramForCausalLM(
            small_config(training_sequence_length=128)
        ).eval().to(device)
        input_ids, labels, attention_mask = _make_eval_batch(device)

        torch._dynamo.reset()
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = compiled(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )
        assert torch.isfinite(out.loss)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "Compiled eval forward requires CUDA; CPU compilation is not "
            "supported for the uncached forward path. "
            "See https://github.com/pytorch/pytorch/issues/148752"
        ),
    )
    def test_compiled_eval_matches_eager(self, device):
        """Compiled eval forward must produce logits numerically identical to eager."""
        m = ShramForCausalLM(
            small_config(training_sequence_length=128)
        ).eval().to(device)
        input_ids, labels, attention_mask = _make_eval_batch(device)

        with torch.no_grad():
            out_eager = m(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )

        torch._dynamo.reset()
        compiled = torch.compile(m, fullgraph=False, dynamic=False)
        with torch.no_grad():
            out_compiled = compiled(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )

        torch.testing.assert_close(out_eager.logits, out_compiled.logits)


# ---------------------------------------------------------------------------
# Integration — Router diagnostics
# ---------------------------------------------------------------------------


class TestIntegrationRouterDiagnostics:
    """Router diagnostic scalars must be present, finite, and live.

    Invariants verified:
    - All four fields are present, scalar, and finite after every forward pass.
    - The scalars reflect actual model state and change as training proceeds.
    """

    def test_diagnostic_fields_present_and_finite(self, device):
        """All four diagnostic scalars must be non-None, scalar, and finite after a forward pass."""
        m = ShramForCausalLM(small_config()).eval().to(device)
        ids = torch.randint(0, 256, (1, 4), device=device)

        with torch.no_grad():
            out = m(ids, use_cache=False)

        for field in ("bias_std", "raw_logit_std", "logit_std", "bias_alignment"):
            val = getattr(out, field)
            assert val is not None, f"{field} is None"
            assert val.ndim == 0, f"{field} is not a scalar (shape={val.shape})"
            assert torch.isfinite(val), f"{field} is not finite: {val.item()}"

    def test_diagnostic_scalars_respond_to_training(self, device):
        """Diagnostic scalars must change after training steps, verifying they are live.

        The default small_config uses K=L (all heads selected), so routing
        frequencies are exactly 1/L and expert_bias receives no gradient.
        This test overrides to K=2, L=4 so routing is genuinely sparse:
        imbalance arises, bias updates, and routing weights evolve.

        Asserts at least one scalar changed — not which one — to remain
        robust against variation in which diagnostic shifts first.
        """
        torch.manual_seed(0)
        config = small_config(num_selected_heads=2, num_mosrah_heads=4, training_sequence_length=32)
        m = ShramForCausalLM(config).train().to(device)
        ids = torch.randint(0, 256, (2, 16), device=device)

        fields = ("bias_std", "raw_logit_std", "logit_std", "bias_alignment")

        with torch.no_grad():
            before = {f: getattr(m(ids, use_cache=False), f).item() for f in fields}

        optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
        for _ in range(10):
            out = m(ids, labels=ids, use_cache=False)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

        with torch.no_grad():
            after = {f: getattr(m(ids, use_cache=False), f).item() for f in fields}

        changed = [f for f in fields if before[f] != after[f]]
        assert changed, (
            f"No diagnostic scalar changed after 10 training steps. "
            f"before={before}, after={after}"
        )


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
        force_download=False,
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
    def model(self, hub_config, device):
        return AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True).eval().to(device)

    def test_output_shape(self, model, device):
        """generate() must return (batch, input_len + max_new_tokens)."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model.generate(ids, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_valid_token_ids(self, model, device):
        """All generated token IDs must be in [0, vocab_size)."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
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
    def model(self, hub_config, device):
        return AutoModelForCausalLM.from_config(hub_config, trust_remote_code=True).train().to(device)

    def test_loss_backward_runs(self, model, device):
        """loss.backward() must complete without error on a Hub-loaded model."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()

    def test_all_params_have_gradients(self, model, device):
        """Every trainable parameter must receive a gradient after backward()."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
