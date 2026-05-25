"""Tests for ShramForCausalLM.

These tests certify the HuggingFace-facing wrapper boundary only. They do not
retest backbone transformer semantics already covered elsewhere. The focus here
is on the wrapper's own responsibilities: output contract, wrapper-level loss,
attention-mask acceptance, cache/generation boundary behavior, tied-embedding
serialization, and basic HuggingFace integration.
"""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.configuration_utils import GenerationMode

from src.shram.model.cache.shram_cache import ShramCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.huggingface import ShramCausalLMOutput, ShramForCausalLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def small_config(**kwargs) -> ShramConfig:
    """Return a minimal ShramConfig suitable for fast unit tests."""
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
        inference_sequence_length=32,
        use_cache=True,
        output_hidden_states=False,
        tie_word_embeddings=False,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def build_cache(
    model: ShramForCausalLM,
    batch_size: int,
    device: torch.device | None = None,
) -> ShramCache:
    """Construct a ShramCache from the given model's config."""
    if device is None:
        device = model.embed_tokens.weight.device

    return ShramCache(
        config=model.config,
        batch_size=batch_size,
        device=device,
    )


@pytest.fixture
def model(device) -> ShramForCausalLM:
    """Provide an eval-mode ShramForCausalLM with a minimal config on the test device."""
    return ShramForCausalLM(small_config()).eval().to(device)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    """Verify the ShramCausalLMOutput wrapper boundary: required fields, shapes, and types."""

    def test_returns_shram_causal_lm_output(self, model: ShramForCausalLM, device) -> None:
        """Forward must return a ShramCausalLMOutput instance."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, use_cache=False)
        assert isinstance(out, ShramCausalLMOutput)

    def test_attribute_access_works(self, model: ShramForCausalLM, device) -> None:
        """Standard output fields must be present and non-None on a basic forward pass."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, use_cache=False)
        assert out.logits is not None
        assert out.load_balance_loss is not None
        assert out.max_vio is not None

    def test_logits_shape(self, model: ShramForCausalLM, device) -> None:
        """Logits must have shape (batch, seq_len, vocab_size)."""
        ids = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
        out = model(ids, use_cache=False)
        assert out.logits.shape == (2, 8, model.config.vocab_size)

    def test_load_balance_loss_is_scalar_and_finite(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """load_balance_loss must be a finite scalar on every forward pass."""
        ids = torch.randint(0, model.config.vocab_size, (1, 6), device=device)
        out = model(ids, use_cache=False)
        assert out.load_balance_loss is not None
        assert out.load_balance_loss.shape == ()
        assert torch.isfinite(out.load_balance_loss)

    def test_max_vio_is_scalar_finite_and_detached(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """max_vio must be a finite detached scalar — it is a monitoring value, not a loss."""
        ids = torch.randint(0, model.config.vocab_size, (1, 6), device=device)
        out = model(ids, use_cache=False)
        assert out.max_vio is not None
        assert out.max_vio.shape == ()
        assert torch.isfinite(out.max_vio)
        assert out.max_vio.requires_grad is False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestLoss:
    """Verify the combined loss contract: CE + load-balance weighting, gradient flow, label handling."""

    def test_loss_none_without_labels(self, model: ShramForCausalLM, device) -> None:
        """loss must be None when no labels are provided."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, use_cache=False)
        assert out.loss is None

    def test_loss_is_scalar_when_labels_provided(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """loss must be a scalar when labels are provided."""
        ids = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
        out = model(ids, labels=ids, use_cache=False)
        assert out.loss is not None
        assert out.loss.shape == ()

    def test_loss_is_positive(self, model: ShramForCausalLM, device) -> None:
        """Combined loss must be positive on a randomly initialized model."""
        ids = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
        out = model(ids, labels=ids, use_cache=False)
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_loss_ignores_minus_100(self, model: ShramForCausalLM, device) -> None:
        """Only unmasked shifted label positions may contribute to the CE loss."""
        torch.manual_seed(7)
        ids = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
        labels = ids.clone()
        labels[:, 1:7] = -100

        with torch.no_grad():
            out = model(ids, labels=labels, use_cache=False)

        expected_ce = torch.nn.functional.cross_entropy(
            out.logits[:, 6, :],
            ids[:, 7],
        )
        torch.testing.assert_close(out.ce_loss, expected_ce)

    def test_ce_loss_none_without_labels(self, model: ShramForCausalLM, device) -> None:
        """ce_loss must be None when no labels are provided."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        out = model(ids, use_cache=False)
        assert out.ce_loss is None

    def test_loss_combines_ce_and_load_balance(self, model: ShramForCausalLM, device) -> None:
        """loss must equal ce_weight * ce_loss + load_balance_weight * load_balance_loss."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        ce_weight, lb_weight = 1.0, 0.01
        with torch.no_grad():
            out = model(ids, labels=ids, use_cache=False,
                        ce_weight=ce_weight, load_balance_weight=lb_weight)
        expected = ce_weight * out.ce_loss + lb_weight * out.load_balance_loss
        torch.testing.assert_close(out.loss, expected)

    def test_custom_weights_scale_loss(self, model: ShramForCausalLM, device) -> None:
        """Custom weights must be applied correctly to both loss components."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        with torch.no_grad():
            out = model(ids, labels=ids, use_cache=False,
                        ce_weight=2.0, load_balance_weight=0.5)
        expected = 2.0 * out.ce_loss + 0.5 * out.load_balance_loss
        torch.testing.assert_close(out.loss, expected)

    def test_expert_bias_receives_gradient(self, device) -> None:
        """expert_bias must receive a gradient through out.loss so the router trains correctly."""
        m = ShramForCausalLM(small_config()).train().to(device)
        ids = torch.randint(0, m.config.vocab_size, (1, 4), device=device)
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()
        bias = m.model.layers[0].attention.sparse_attention.router.expert_bias
        assert bias.grad is not None


# ---------------------------------------------------------------------------
# Attention-mask behavior
# ---------------------------------------------------------------------------


class TestAttentionMaskBehavior:
    """Verify that the wrapper correctly accepts and threads the 2D attention mask."""

    def test_all_ones_attention_mask_matches_unmasked(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """Accepting a full-live mask must not change unmasked behavior."""
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 6), device=device)
        mask = torch.ones_like(ids)

        with torch.no_grad():
            unmasked = model(ids, use_cache=False)
            masked = model(ids, attention_mask=mask, use_cache=False)

        torch.testing.assert_close(masked.logits, unmasked.logits)
        torch.testing.assert_close(masked.load_balance_loss, unmasked.load_balance_loss)
        torch.testing.assert_close(masked.max_vio, unmasked.max_vio)

    def test_ragged_batched_generate_works_with_attention_mask(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """Wrapper must accept a full 2D mask during generation."""
        input_ids = torch.tensor(
            [
                [5, 6, 7, 8],
                [9, 10, 0, 0],
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                do_sample=False,
            )

        assert generated.shape == (2, 6)


# ---------------------------------------------------------------------------
# Tied embeddings and serialization
# ---------------------------------------------------------------------------


class TestWeightTying:
    """Verify config-controlled tied embedding behavior and its preservation through save/load."""

    def test_tied_weights_share_storage(self) -> None:
        """When tie_word_embeddings=True, lm_head and embed_tokens must share the same tensor."""
        m = ShramForCausalLM(small_config(tie_word_embeddings=True)).eval()
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()

    def test_untied_weights_are_independent(self) -> None:
        """When tie_word_embeddings=False, lm_head and embed_tokens must be independent tensors."""
        m = ShramForCausalLM(small_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight.data_ptr() != m.embed_tokens.weight.data_ptr()

    def test_tied_round_trip_preserves_tying(self, tmp_path) -> None:
        """save_pretrained / from_pretrained must preserve tied embedding state."""
        m = ShramForCausalLM(small_config(tie_word_embeddings=True)).eval()
        m.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path)

        assert loaded.config.tie_word_embeddings is True
        assert loaded.lm_head.weight.data_ptr() == loaded.embed_tokens.weight.data_ptr()

    def test_untied_round_trip_preserves_untied_state(self, tmp_path) -> None:
        """save_pretrained / from_pretrained must preserve untied embedding state."""
        m = ShramForCausalLM(small_config(tie_word_embeddings=False)).eval()
        m.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path)

        assert loaded.config.tie_word_embeddings is False
        assert loaded.lm_head.weight.data_ptr() != loaded.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# Direct cache policy at the wrapper boundary
# ---------------------------------------------------------------------------


class TestDirectCachePolicy:
    """Verify cache validation at the direct forward() wrapper boundary."""

    def test_use_cache_true_requires_explicit_shram_cache(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """use_cache=True without a supplied ShramCache must raise at the wrapper boundary."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        position_ids = torch.arange(4, device=device).unsqueeze(0)

        with pytest.raises(ValueError, match="requires an explicit ShramCache"):
            model(ids, position_ids=position_ids, use_cache=True)

    def test_explicit_shram_cache_is_used_unchanged(
        self,
        model: ShramForCausalLM,
        device,
    ) -> None:
        """A caller-supplied ShramCache must be passed through to the output unchanged."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        position_ids = torch.arange(4, device=device).unsqueeze(0)
        cache = build_cache(model, batch_size=1)

        out = model(
            ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        assert out.past_key_values is cache


# ---------------------------------------------------------------------------
# Uncached position constraint
# ---------------------------------------------------------------------------


class TestUncachedPositionConstraint:
    """Verify that uncached forwards reject nonzero starting positions in both eager and compiled modes."""
    def test_uncached_forward_rejects_nonzero_starting_position(
        self, model: ShramForCausalLM, device
    ) -> None:
        """Uncached forward must raise at the wrapper boundary when positions start nonzero."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        position_ids = torch.arange(10, 14, dtype=torch.long, device=device).unsqueeze(0)
        with pytest.raises(RuntimeError, match="nonzero starting positions"):
            model(ids, position_ids=position_ids, use_cache=False)

    def test_uncached_forward_accepts_zero_starting_position(
        self, model: ShramForCausalLM, device
    ) -> None:
        """Uncached forward must succeed when positions start at zero."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
        position_ids = torch.arange(4, dtype=torch.long, device=device).unsqueeze(0)
        out = model(ids, position_ids=position_ids, use_cache=False)
        assert out.logits is not None

    def test_compiled_enforce_raises_on_violation(self) -> None:
        """Static method compiled in isolation must raise when condition is False."""

        def _check_wrapper(condition: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            ShramForCausalLM._enforce_uncached_starting_position(condition)
            return x * condition.to(x.dtype)

        torch._dynamo.reset()
        original = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        try:
            compiled = torch.compile(_check_wrapper)
            with pytest.raises(RuntimeError):
                compiled(torch.tensor(False), torch.ones(1))
        finally:
            torch._dynamo.config.capture_scalar_outputs = original

    def test_compiled_enforce_accepts_valid_condition(self) -> None:
        """Static method compiled in isolation must not raise when condition is True."""

        def _check_wrapper(condition: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            ShramForCausalLM._enforce_uncached_starting_position(condition)
            return x * condition.to(x.dtype)

        torch._dynamo.reset()
        original = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        try:
            compiled = torch.compile(_check_wrapper)
            compiled(torch.tensor(True), torch.ones(1))
        finally:
            torch._dynamo.config.capture_scalar_outputs = original


# ---------------------------------------------------------------------------
# capture_scalar_outputs enforcement
# ---------------------------------------------------------------------------


class TestCaptureScalarOutputsEnforcement:
    """Verify that compiling without capture_scalar_outputs=True raises at compile time."""
    def test_compiled_without_flag_raises(self) -> None:
        """Compiling without capture_scalar_outputs must raise at compile time."""
        torch._dynamo.reset()
        original = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = False
        try:
            compiled = torch.compile(ShramForCausalLM._enforce_capture_scalar_outputs)
            with pytest.raises(RuntimeError, match="capture_scalar_outputs"):
                compiled()
        finally:
            torch._dynamo.config.capture_scalar_outputs = original

    def test_compiled_with_flag_does_not_raise(self) -> None:
        """Compiling with capture_scalar_outputs=True must not raise."""
        torch._dynamo.reset()
        original = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        try:
            compiled = torch.compile(ShramForCausalLM._enforce_capture_scalar_outputs)
            compiled()
        finally:
            torch._dynamo.config.capture_scalar_outputs = original


# ---------------------------------------------------------------------------
# Generation cache hook
# ---------------------------------------------------------------------------


class TestGenerationCacheHook:
    """Verify that _prepare_cache_for_generation constructs ShramCache and respects caller-supplied caches."""

    def test_prepare_cache_for_generation_constructs_shram_cache(
        self,
        model: ShramForCausalLM,
    ) -> None:
        """Hook must construct a ShramCache when no cache is pre-supplied."""
        generation_config = copy.deepcopy(model.generation_config)
        model_kwargs: dict[str, object] = {}

        model._prepare_cache_for_generation(
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            generation_mode=GenerationMode.GREEDY_SEARCH,
            batch_size=2,
            max_cache_length=16,
        )

        assert isinstance(model_kwargs["past_key_values"], ShramCache)

    def test_prepare_cache_for_generation_preserves_explicit_cache(
        self,
        model: ShramForCausalLM,
    ) -> None:
        """Hook must leave a caller-supplied ShramCache untouched."""
        generation_config = copy.deepcopy(model.generation_config)
        cache = build_cache(model, batch_size=2)
        model_kwargs: dict[str, object] = {"past_key_values": cache}

        model._prepare_cache_for_generation(
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            generation_mode=GenerationMode.GREEDY_SEARCH,
            batch_size=2,
            max_cache_length=16,
        )

        assert model_kwargs["past_key_values"] is cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGeneration:
    """Verify that generate() works end-to-end for supported generation modes."""

    def test_generate_works(self, model: ShramForCausalLM, device) -> None:
        """Greedy generation must complete and return the correct output shape."""
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=2,
                do_sample=False,
            )

        assert generated.shape == (1, 5)

    def test_beam_search_generate_works(self, model: ShramForCausalLM, device) -> None:
        """Beam search generation must complete and return the correct output shape."""
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=2,
                num_beams=2,
                do_sample=False,
            )

        assert generated.shape == (1, 5)


# ---------------------------------------------------------------------------
# Config defaults and HF integration
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verify that config-level defaults are correctly respected by the wrapper."""

    def test_config_output_hidden_states_respected(self, device) -> None:
        """output_hidden_states=True in config must produce non-None hidden_states in output."""
        config = small_config(output_hidden_states=True, use_cache=False)
        m = ShramForCausalLM(config).eval().to(device)

        with torch.no_grad():
            out = m(torch.randint(0, config.vocab_size, (1, 4), device=device))

        assert out.hidden_states is not None

    def test_config_use_cache_false_respected(self, device) -> None:
        """use_cache=False in config must produce None past_key_values in output."""
        config = small_config(use_cache=False)
        m = ShramForCausalLM(config).eval().to(device)

        with torch.no_grad():
            out = m(torch.randint(0, config.vocab_size, (1, 4), device=device))

        assert out.past_key_values is None


class TestSaveLoadAndAutoClass:
    """Verify HuggingFace save/load round-trip and AutoClass registration."""

    def test_save_pretrained_round_trip_preserves_parameter_values(
        self,
        model: ShramForCausalLM,
        tmp_path,
        device,
    ) -> None:
        """All parameter values must survive a save_pretrained / from_pretrained round-trip."""
        model.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path).to(device)
        for (name, p1), (_, p2) in zip(
            model.named_parameters(),
            loaded.named_parameters(),
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")
    def test_rope_tables_rebuilt_for_new_inference_length(self, tmp_path) -> None:
        """Rope cos/sin tables must reflect the loaded config's inference_sequence_length.

        HuggingFace's fast-init path (init_empty_weights) could silently copy stale
        rope tables back into place after __init__ rebuilds them. This test catches
        that regression by saving with one inference_sequence_length and loading with
        a different one, then asserting both rope instances were built to the new length.
        """
        original_length = 32
        new_length = 64

        model = ShramForCausalLM(small_config(inference_sequence_length=original_length))
        model.save_pretrained(tmp_path)

        loaded = ShramForCausalLM.from_pretrained(tmp_path, inference_sequence_length=new_length)

        local_rope = loaded.model.layers[0].attention.local_attention.rope
        bea_rope = loaded.model.layers[0].attention.sparse_attention.bea.rope

        assert local_rope._cos_cached.shape[0] == new_length, (
            f"Local rope table has length {local_rope._cos_cached.shape[0]}, expected {new_length}"
        )
        assert bea_rope._cos_cached.shape[0] == new_length, (
            f"BEA rope table has length {bea_rope._cos_cached.shape[0]}, expected {new_length}"
        )

    def test_auto_model_from_config(self) -> None:
        """AutoModelForCausalLM.from_config must return a ShramForCausalLM when registered."""
        AutoModelForCausalLM.register(ShramConfig, ShramForCausalLM)
        config = small_config(use_cache=False)
        model = AutoModelForCausalLM.from_config(config)
        assert isinstance(model, ShramForCausalLM)

class TestNumMosrahParameters:
    """Verify the MoSRAH parameter count method: scaling, partition, and stability."""

    def test_scaling_with_layers(self):
        """2× num_decoder_layers must produce exactly 2× the MoSRAH parameter count."""
        config_base = small_config(num_decoder_layers=2)
        config_double = small_config(num_decoder_layers=4)
        model_base = ShramForCausalLM(config_base)
        model_double = ShramForCausalLM(config_double)
        assert model_double.num_mosrah_parameters() == 2 * model_base.num_mosrah_parameters()

    def test_partition(self):
        """MoSRAH parameter count must be strictly less than total model parameter count."""
        model = ShramForCausalLM(small_config())
        total = sum(p.numel() for p in model.parameters())
        assert model.num_mosrah_parameters() < total

    def test_stability(self):
        """Two calls on the same model must return the same value."""
        model = ShramForCausalLM(small_config())
        assert model.num_mosrah_parameters() == model.num_mosrah_parameters()


class TestCreateMasksForGenerate:
    """create_masks_for_generate must return the 2D attention_mask unchanged (Unit 19.G.4)."""

    def test_returns_attention_mask_unchanged(self):
        """The override must pass the attention_mask through without modification."""
        attention_mask = torch.ones(2, 10, dtype=torch.bool)
        result = ShramForCausalLM.create_masks_for_generate(
            config=small_config(),
            inputs_embeds=torch.empty(2, 1, 0),
            attention_mask=attention_mask,
            past_key_values=None,
        )
        assert result is attention_mask

    def test_returns_none_when_mask_is_none(self):
        """None attention_mask must be returned as None."""
        result = ShramForCausalLM.create_masks_for_generate(
            config=small_config(),
            inputs_embeds=torch.empty(2, 1, 0),
            attention_mask=None,
            past_key_values=None,
        )
        assert result is None

    def test_result_is_still_2d(self):
        """Returned mask must remain 2D — not converted to 4D additive-bias format."""
        attention_mask = torch.ones(2, 10, dtype=torch.bool)
        result = ShramForCausalLM.create_masks_for_generate(
            config=small_config(),
            inputs_embeds=torch.empty(2, 1, 0),
            attention_mask=attention_mask,
            past_key_values=None,
        )
        assert result.ndim == 2


def test_shram_cache_initializes_correctly_for_batch_size_two() -> None:
    """ShramCache constructed with batch_size=2 must propagate that batch dimension correctly."""
    cache = ShramCache(
        config=small_config(),
        batch_size=2,
        device=torch.device("cpu"),
    )

    layer_cache = cache.layers[0]

    # Local cache path: first real tensors seen by the cache are batch size 2.
    local_keys = torch.randn(2, 4, 3, 16)
    local_values = torch.randn(2, 4, 3, 16)
    local_active_mask = torch.ones(2, 3, dtype=torch.bool)

    local_positions = torch.arange(3, dtype=torch.long).unsqueeze(0).expand(2, -1)
    returned_keys, returned_values, returned_mask, _ = layer_cache.sliding_window_cache.update(
        local_keys,
        local_values,
        local_active_mask,
        local_positions,
    )

    assert returned_keys.shape[0] == 2
    assert returned_values.shape[0] == 2
    assert returned_mask.shape[0] == 2

    # Top-level sequence length should also remain truthful at this wider batch size.
    assert cache.get_seq_length() == 3