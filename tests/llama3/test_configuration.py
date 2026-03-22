"""Tests for Llama3Config.

Each test verifies a specific invariant documented in the plan. The grouping mirrors
the structure of the invariants: defaults, parameter overrides, structural validation,
rope configuration, and serialisation.

RoPE scaling validation is owned by HF's RotaryEmbeddingConfigMixin and is not tested
here — we test that our config correctly passes parameters through to HF's system.
"""

import pytest

from src.llama3.configuration import Llama3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    """Return a config with small dimensions suitable for testing.

    Using full-scale defaults (hidden_size=4096 etc.) in tests that don't care about
    scale adds noise. This helper applies a consistent small baseline that satisfies
    all structural constraints.
    """
    defaults = dict(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1024,
        num_hidden_layers=4,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_default_instantiation_succeeds(self):
        """Config must instantiate without arguments — all parameters have defaults."""
        config = Llama3Config()
        assert config is not None

    def test_model_type(self):
        """model_type must be unique to avoid colliding with HF's built-in 'llama'."""
        assert Llama3Config.model_type == "llama3_baseline"

    def test_auto_map_present(self):
        """auto_map must be set so HuggingFace trust_remote_code can find the classes."""
        assert "AutoConfig" in Llama3Config.auto_map
        assert "AutoModelForCausalLM" in Llama3Config.auto_map

    def test_head_dim_computed_when_not_provided(self):
        """head_dim is derived from hidden_size // num_attention_heads when not set."""
        config = small_config(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------

class TestParameterStorage:
    def test_vocab_size_stored(self):
        config = small_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_num_hidden_layers_stored(self):
        config = small_config(num_hidden_layers=16)
        assert config.num_hidden_layers == 16

    def test_rope_theta_stored(self):
        config = small_config(rope_theta=10000.0)
        assert config.rope_theta == 10000.0

    def test_max_position_embeddings_stored(self):
        config = small_config(max_position_embeddings=4096)
        assert config.max_position_embeddings == 4096

    def test_head_dim_explicit_stored(self):
        """head_dim can be set directly to decouple it from hidden_size // num_heads."""
        config = small_config(head_dim=128)
        assert config.head_dim == 128

    def test_use_cache_stored(self):
        config = small_config(use_cache=False)
        assert config.use_cache is False

    def test_output_hidden_states_defaults_false(self):
        """output_hidden_states must default to False — opt-in, not opt-out."""
        config = small_config()
        assert config.output_hidden_states is False

    def test_output_hidden_states_stored(self):
        config = small_config(output_hidden_states=True)
        assert config.output_hidden_states is True

    def test_tie_word_embeddings_stored(self):
        config = small_config(tie_word_embeddings=True)
        assert config.tie_word_embeddings is True


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

class TestStructuralValidation:
    def test_hidden_size_not_divisible_by_num_heads_raises(self):
        """A hidden_size that doesn't divide evenly across heads is structurally invalid."""
        with pytest.raises(ValueError, match="hidden_size"):
            Llama3Config(hidden_size=100, num_attention_heads=32)

    def test_num_heads_not_divisible_by_kv_heads_raises(self):
        """GQA requires query heads to divide evenly across KV head groups."""
        with pytest.raises(ValueError, match="num_attention_heads"):
            small_config(num_attention_heads=32, num_key_value_heads=7)

    def test_mha_is_valid(self):
        """num_key_value_heads == num_attention_heads gives standard MHA — valid."""
        config = small_config(num_attention_heads=8, num_key_value_heads=8)
        assert config.num_key_value_heads == config.num_attention_heads

    def test_mqa_is_valid(self):
        """num_key_value_heads == 1 gives MQA — valid."""
        config = small_config(num_attention_heads=8, num_key_value_heads=1)
        assert config.num_key_value_heads == 1


# ---------------------------------------------------------------------------
# RoPE configuration
# ---------------------------------------------------------------------------

class TestRopeConfiguration:
    def test_no_rope_scaling_by_default(self):
        """Without rope_scaling, HF leaves rope_parameters as None — rope_theta is used directly."""
        config = small_config()
        assert config.rope_parameters is None

    def test_linear_rope_scaling_accepted(self):
        """Linear scaling is the simplest extension method — divides all frequencies by factor."""
        config = small_config(
            rope_scaling={"rope_type": "linear", "factor": 4.0}
        )
        assert config.rope_parameters["rope_type"] == "linear"

    def test_yarn_rope_scaling_accepted(self):
        """YaRN applies frequency-aware scaling and is fully supported via HF's system."""
        config = small_config(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
            }
        )
        assert config.rope_parameters["rope_type"] == "yarn"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_roundtrip_preserves_all_fields(self):
        """A config serialised to dict and restored must be identical to the original.

        This is the HF contract for config persistence. If it breaks, save/load of
        model checkpoints will silently use wrong architectural parameters.
        """
        original = small_config(
            rope_theta=10000.0,
            max_position_embeddings=4096,
            attention_dropout=0.1,
            use_cache=False,
        )
        restored = Llama3Config.from_dict(original.to_dict())

        assert restored.vocab_size == original.vocab_size
        assert restored.hidden_size == original.hidden_size
        assert restored.intermediate_size == original.intermediate_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_attention_heads == original.num_attention_heads
        assert restored.num_key_value_heads == original.num_key_value_heads
        assert restored.head_dim == original.head_dim
        assert restored.rms_norm_eps == original.rms_norm_eps
        assert restored.rope_theta == original.rope_theta
        assert restored.max_position_embeddings == original.max_position_embeddings
        assert restored.attention_dropout == original.attention_dropout
        assert restored.use_cache == original.use_cache
        assert restored.output_hidden_states == original.output_hidden_states
        assert restored.tie_word_embeddings == original.tie_word_embeddings
