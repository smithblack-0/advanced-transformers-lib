"""Tests for Llama3Config.

Each test verifies a specific invariant documented in the plan. The grouping mirrors
the structure of the invariants: defaults, parameter overrides, structural validation,
rope_scaling validation, and serialisation.
"""

import pytest

from llama3.configuration import Llama3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    """Return a config with small dimensions suitable for testing.

    Using full-scale defaults (hidden_size=4096 etc.) in tests that don't care about
    scale adds noise and slows things down. This helper applies a consistent small
    baseline that satisfies all structural constraints.
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

class TestDefaults:
    def test_default_instantiation_succeeds(self):
        config = Llama3Config()
        assert config is not None

    def test_model_type(self):
        """model_type must be unique to avoid colliding with HF's built-in 'llama'."""
        assert Llama3Config.model_type == "llama3_baseline"

    def test_default_vocab_size(self):
        """128,000 matches the Llama 3 tokenizer (100K base + 28K multilingual)."""
        assert Llama3Config().vocab_size == 128000

    def test_head_dim_computed_when_not_provided(self):
        """head_dim defaults to hidden_size // num_attention_heads."""
        config = small_config(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64

    def test_tie_word_embeddings_defaults_false(self):
        """Llama 3 does not tie the input embedding table and the LM head."""
        assert Llama3Config().tie_word_embeddings is False

    def test_rope_scaling_defaults_none(self):
        assert Llama3Config().rope_scaling is None

    def test_use_cache_defaults_true(self):
        assert Llama3Config().use_cache is True

    def test_auto_map_present(self):
        """auto_map must be set so HuggingFace trust_remote_code can find the classes."""
        assert "AutoConfig" in Llama3Config.auto_map
        assert "AutoModelForCausalLM" in Llama3Config.auto_map


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------

class TestParameterOverrides:
    def test_num_hidden_layers_override(self):
        config = small_config(num_hidden_layers=16)
        assert config.num_hidden_layers == 16

    def test_rope_theta_override(self):
        config = small_config(rope_theta=10000.0)
        assert config.rope_theta == 10000.0

    def test_head_dim_explicit_override(self):
        """head_dim can be set directly to decouple it from hidden_size // num_heads."""
        config = small_config(head_dim=128)
        assert config.head_dim == 128

    def test_use_cache_override(self):
        config = small_config(use_cache=False)
        assert config.use_cache is False


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
# RoPE scaling validation
# ---------------------------------------------------------------------------

class TestRopeScalingValidation:
    def test_none_is_valid(self):
        config = small_config(rope_scaling=None)
        assert config.rope_scaling is None

    def test_linear_scaling_valid(self):
        config = small_config(rope_scaling={"type": "linear", "factor": 4.0})
        assert config.rope_scaling["type"] == "linear"
        assert config.rope_scaling["factor"] == 4.0

    def test_yarn_type_accepted(self):
        """YaRN is a recognised type even though the rope.py implementation is a placeholder."""
        config = small_config(rope_scaling={"type": "yarn", "factor": 4.0})
        assert config.rope_scaling["type"] == "yarn"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            small_config(rope_scaling={"type": "unknown", "factor": 4.0})

    def test_missing_factor_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            small_config(rope_scaling={"type": "linear"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            small_config(rope_scaling={"factor": 4.0})

    def test_factor_below_one_raises(self):
        with pytest.raises(ValueError, match="factor"):
            small_config(rope_scaling={"type": "linear", "factor": 0.5})

    def test_factor_equal_to_one_raises(self):
        """factor=1.0 does not extend context and is not a valid use of rope_scaling."""
        with pytest.raises(ValueError, match="factor"):
            small_config(rope_scaling={"type": "linear", "factor": 1.0})


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
            rope_scaling={"type": "linear", "factor": 2.0},
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
        assert restored.rope_scaling == original.rope_scaling
        assert restored.attention_dropout == original.attention_dropout
        assert restored.use_cache == original.use_cache
        assert restored.tie_word_embeddings == original.tie_word_embeddings
