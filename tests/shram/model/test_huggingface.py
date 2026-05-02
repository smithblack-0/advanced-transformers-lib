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
        inference_sequence_length=32,
        use_cache=True,
        output_hidden_states=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def build_cache(
    model: ShramForCausalLM,
    batch_size: int,
    device: torch.device | None = None,
) -> ShramCache:
    if device is None:
        device = model.embed_tokens.weight.device

    return ShramCache(
        num_hidden_layers=model.config.num_hidden_layers,
        sliding_window=model.config.window_size,
        num_local_heads=model.config.num_sliding_window_heads,
        local_head_dim=model.config.head_dim,
        num_mosrah_heads=model.config.num_mosrah_heads,
        mosrah_head_dim=model.config.hidden_size // model.config.num_selected_heads,
        batch_size=batch_size,
        device=device,
    )


@pytest.fixture
def model() -> ShramForCausalLM:
    return ShramForCausalLM(small_config()).eval()


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_returns_shram_causal_lm_output(self, model: ShramForCausalLM) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, use_cache=False)
        assert isinstance(out, ShramCausalLMOutput)

    def test_attribute_access_works(self, model: ShramForCausalLM) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, use_cache=False)
        assert out.logits is not None
        assert out.load_balance_loss is not None
        assert out.max_vio is not None

    def test_logits_shape(self, model: ShramForCausalLM) -> None:
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        out = model(ids, use_cache=False)
        assert out.logits.shape == (2, 8, model.config.vocab_size)

    def test_load_balance_loss_is_scalar_and_finite(
        self,
        model: ShramForCausalLM,
    ) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 6))
        out = model(ids, use_cache=False)
        assert out.load_balance_loss is not None
        assert out.load_balance_loss.shape == ()
        assert torch.isfinite(out.load_balance_loss)

    def test_max_vio_is_scalar_finite_and_detached(
        self,
        model: ShramForCausalLM,
    ) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 6))
        out = model(ids, use_cache=False)
        assert out.max_vio is not None
        assert out.max_vio.shape == ()
        assert torch.isfinite(out.max_vio)
        assert out.max_vio.requires_grad is False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestLoss:
    def test_loss_none_without_labels(self, model: ShramForCausalLM) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, use_cache=False)
        assert out.loss is None

    def test_loss_is_scalar_when_labels_provided(
        self,
        model: ShramForCausalLM,
    ) -> None:
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        out = model(ids, labels=ids, use_cache=False)
        assert out.loss is not None
        assert out.loss.shape == ()

    def test_loss_is_positive(self, model: ShramForCausalLM) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 8))
        out = model(ids, labels=ids, use_cache=False)
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_loss_ignores_minus_100(self, model: ShramForCausalLM) -> None:
        """Only unmasked shifted label positions may contribute to the CE loss."""
        torch.manual_seed(7)
        ids = torch.randint(0, model.config.vocab_size, (1, 8))
        labels = ids.clone()
        labels[:, 1:7] = -100

        with torch.no_grad():
            out = model(ids, labels=labels, use_cache=False)

        expected_ce = torch.nn.functional.cross_entropy(
            out.logits[:, 6, :],
            ids[:, 7],
        )
        torch.testing.assert_close(out.ce_loss, expected_ce)

    def test_ce_loss_none_without_labels(self, model: ShramForCausalLM) -> None:
        """ce_loss must be None when no labels are provided."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        out = model(ids, use_cache=False)
        assert out.ce_loss is None

    def test_loss_combines_ce_and_load_balance(self, model: ShramForCausalLM) -> None:
        """loss must equal ce_weight * ce_loss + load_balance_weight * load_balance_loss."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        ce_weight, lb_weight = 1.0, 0.01
        with torch.no_grad():
            out = model(ids, labels=ids, use_cache=False,
                        ce_weight=ce_weight, load_balance_weight=lb_weight)
        expected = ce_weight * out.ce_loss + lb_weight * out.load_balance_loss
        torch.testing.assert_close(out.loss, expected)

    def test_custom_weights_scale_loss(self, model: ShramForCausalLM) -> None:
        """Custom weights must be applied correctly to both loss components."""
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(ids, labels=ids, use_cache=False,
                        ce_weight=2.0, load_balance_weight=0.5)
        expected = 2.0 * out.ce_loss + 0.5 * out.load_balance_loss
        torch.testing.assert_close(out.loss, expected)

    def test_expert_bias_receives_gradient(self) -> None:
        """expert_bias must receive a gradient when out.loss.backward() is called."""
        m = ShramForCausalLM(small_config()).train()
        ids = torch.randint(0, m.config.vocab_size, (1, 4))
        out = m(ids, labels=ids, use_cache=False)
        out.loss.backward()
        bias = m.model.layers[0].attention.sparse_attention.router.expert_bias
        assert bias.grad is not None


# ---------------------------------------------------------------------------
# Attention-mask behavior
# ---------------------------------------------------------------------------


class TestAttentionMaskBehavior:
    def test_all_ones_attention_mask_matches_unmasked(
        self,
        model: ShramForCausalLM,
    ) -> None:
        """Accepting a full-live mask must not change unmasked behavior."""
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 6))
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
    ) -> None:
        """Wrapper must accept a full 2D mask during generation."""
        input_ids = torch.tensor(
            [
                [5, 6, 7, 8],
                [9, 10, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=torch.long,
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
    def test_tied_weights_share_storage(self) -> None:
        m = ShramForCausalLM(small_config(tie_word_embeddings=True)).eval()
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()

    def test_untied_weights_are_independent(self) -> None:
        m = ShramForCausalLM(small_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight.data_ptr() != m.embed_tokens.weight.data_ptr()

    def test_tied_round_trip_preserves_tying(self, tmp_path) -> None:
        m = ShramForCausalLM(small_config(tie_word_embeddings=True)).eval()
        m.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path)

        assert loaded.config.tie_word_embeddings is True
        assert loaded.lm_head.weight.data_ptr() == loaded.embed_tokens.weight.data_ptr()

    def test_untied_round_trip_preserves_untied_state(self, tmp_path) -> None:
        m = ShramForCausalLM(small_config(tie_word_embeddings=False)).eval()
        m.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path)

        assert loaded.config.tie_word_embeddings is False
        assert loaded.lm_head.weight.data_ptr() != loaded.embed_tokens.weight.data_ptr()


# ---------------------------------------------------------------------------
# Direct cache policy at the wrapper boundary
# ---------------------------------------------------------------------------


class TestDirectCachePolicy:
    def test_use_cache_true_requires_explicit_shram_cache(
        self,
        model: ShramForCausalLM,
    ) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        position_ids = torch.arange(4).unsqueeze(0)

        with pytest.raises(ValueError, match="requires an explicit ShramCache"):
            model(ids, position_ids=position_ids, use_cache=True)

    def test_explicit_shram_cache_is_used_unchanged(
        self,
        model: ShramForCausalLM,
    ) -> None:
        ids = torch.randint(0, model.config.vocab_size, (1, 4))
        position_ids = torch.arange(4).unsqueeze(0)
        cache = build_cache(model, batch_size=1)

        out = model(
            ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        assert out.past_key_values is cache


# ---------------------------------------------------------------------------
# Generation cache hook
# ---------------------------------------------------------------------------


class TestGenerationCacheHook:
    def test_prepare_cache_for_generation_constructs_shram_cache(
        self,
        model: ShramForCausalLM,
    ) -> None:
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
    def test_generate_works(self, model: ShramForCausalLM) -> None:
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=2,
                do_sample=False,
            )

        assert generated.shape == (1, 5)

    def test_beam_search_generate_works(self, model: ShramForCausalLM) -> None:
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)

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
    def test_config_output_hidden_states_respected(self) -> None:
        config = small_config(output_hidden_states=True, use_cache=False)
        m = ShramForCausalLM(config).eval()

        with torch.no_grad():
            out = m(torch.randint(0, config.vocab_size, (1, 4)))

        assert out.hidden_states is not None

    def test_config_use_cache_false_respected(self) -> None:
        config = small_config(use_cache=False)
        m = ShramForCausalLM(config).eval()

        with torch.no_grad():
            out = m(torch.randint(0, config.vocab_size, (1, 4)))

        assert out.past_key_values is None


class TestSaveLoadAndAutoClass:
    def test_save_pretrained_round_trip_preserves_parameter_values(
        self,
        model: ShramForCausalLM,
        tmp_path,
    ) -> None:
        model.save_pretrained(tmp_path)
        loaded = ShramForCausalLM.from_pretrained(tmp_path)

        for (name, p1), (_, p2) in zip(
            model.named_parameters(),
            loaded.named_parameters(),
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")

    def test_auto_model_from_config(self) -> None:
        AutoModelForCausalLM.register(ShramConfig, ShramForCausalLM)
        config = small_config(use_cache=False)
        model = AutoModelForCausalLM.from_config(config)
        assert isinstance(model, ShramForCausalLM)

def test_shram_cache_initializes_correctly_for_batch_size_two() -> None:
    cache = ShramCache(
        num_hidden_layers=2,
        sliding_window=8,
        num_local_heads=4,
        local_head_dim=16,
        num_mosrah_heads=4,
        mosrah_head_dim=16,
        device=torch.device("cpu"),
        batch_size=2,
    )

    layer_cache = cache.layers[0]

    # Local cache path: first real tensors seen by the cache are batch size 2.
    local_keys = torch.randn(2, 4, 3, 16)
    local_values = torch.randn(2, 4, 3, 16)
    local_active_mask = torch.ones(2, 3, dtype=torch.bool)

    returned_keys, returned_values, returned_mask = layer_cache.sliding_window_cache.update(
        local_keys,
        local_values,
        local_active_mask,
    )

    assert returned_keys.shape[0] == 2
    assert returned_values.shape[0] == 2
    assert returned_mask.shape[0] == 2

    # Top-level sequence length should also remain truthful at this wider batch size.
    assert cache.get_seq_length() == 3