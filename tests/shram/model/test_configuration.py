"""Tests for ShramConfig.

Each test verifies a specific invariant documented in the plan. The grouping mirrors
the invariant categories: instantiation, parameter storage, structural validation,
rope parameters, scale property, and serialisation.
"""

import pytest

from src.shram.model.configuration import ShramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    """Return a config with small dimensions suitable for testing."""
    defaults = dict(
        hidden_size=512,
        num_sliding_window_heads=16,
        num_mosrah_heads=16,
        num_selected_heads=16,
        head_dim=16,
        window_size=128,
        intermediate_size=1024,
        num_hidden_layers=4,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_default_instantiation_succeeds(self):
        """Config must instantiate without arguments — all parameters have defaults."""
        config = ShramConfig()
        assert config is not None

    def test_model_type(self):
        """model_type must be unique to avoid colliding with HF's built-in 'llama'."""
        assert ShramConfig.model_type == "shram"

    def test_auto_map_present(self):
        """auto_map must be set so HuggingFace trust_remote_code can find the classes."""
        assert "AutoConfig" in ShramConfig.auto_map
        assert "AutoModelForCausalLM" in ShramConfig.auto_map

    def test_auto_map_points_to_correct_files(self):
        """auto_map paths must match the actual file layout used on the Hub."""
        assert ShramConfig.auto_map["AutoConfig"] == "configuration.ShramConfig"
        assert ShramConfig.auto_map["AutoModelForCausalLM"] == "huggingface.ShramForCausalLM"


# ---------------------------------------------------------------------------
# Parameter storage
# ---------------------------------------------------------------------------

class TestParameterStorage:
    def test_vocab_size_stored(self):
        config = small_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_num_hidden_layers_stored(self):
        config = small_config(num_hidden_layers=16)
        assert config.num_hidden_layers == 16

    def test_num_sliding_window_heads_stored(self):
        config = small_config(num_sliding_window_heads=8)
        assert config.num_sliding_window_heads == 8

    def test_num_mosrah_heads_stored(self):
        config = small_config(num_mosrah_heads=32)
        assert config.num_mosrah_heads == 32

    def test_num_selected_heads_stored(self):
        config = small_config(num_selected_heads=8)
        assert config.num_selected_heads == 8

    def test_head_dim_stored(self):
        config = small_config(head_dim=32)
        assert config.head_dim == 32

    def test_window_size_stored(self):
        config = small_config(window_size=256)
        assert config.window_size == 256

    def test_rope_mode_stored(self):
        config = small_config(rope_mode="semantic_sequence")
        assert config.rope_mode == "semantic_sequence"

    def test_local_rope_theta_stored(self):
        config = small_config(local_rope_theta=500000.0)
        assert config.local_rope_theta == 500000.0

    def test_mosrah_rope_theta_stored(self):
        config = small_config(mosrah_rope_theta=500000.0)
        assert config.mosrah_rope_theta == 500000.0

    def test_training_sequence_length_stored(self):
        config = small_config(training_sequence_length=4096)
        assert config.training_sequence_length == 4096

    def test_inference_sequence_length_stored(self):
        config = small_config(inference_sequence_length=16384)
        assert config.inference_sequence_length == 16384

    def test_alpha_stored(self):
        config = small_config(alpha=2.0)
        assert config.alpha == 2.0

    def test_beta_stored(self):
        config = small_config(beta=16.0)
        assert config.beta == 16.0

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
    def test_odd_head_dim_raises(self):
        """head_dim must be even — RoPE rotates dimensions in pairs for both paths."""
        with pytest.raises(ValueError, match="head_dim must be even"):
            small_config(head_dim=15)

    def test_invalid_rope_mode_raises(self):
        """rope_mode must be one of the two supported values."""
        with pytest.raises(ValueError, match="rope_mode"):
            small_config(rope_mode="invalid_mode")

    def test_main_sequence_rope_mode_valid(self):
        config = small_config(rope_mode="main_sequence")
        assert config.rope_mode == "main_sequence"

    def test_semantic_sequence_rope_mode_valid(self):
        config = small_config(rope_mode="semantic_sequence")
        assert config.rope_mode == "semantic_sequence"

    def test_zero_training_sequence_length_raises(self):
        """training_sequence_length must be positive — used as divisor in scale."""
        with pytest.raises(ValueError, match="training_sequence_length"):
            small_config(training_sequence_length=0)

    def test_zero_inference_sequence_length_raises(self):
        """inference_sequence_length must be positive."""
        with pytest.raises(ValueError, match="inference_sequence_length"):
            small_config(inference_sequence_length=0)


# ---------------------------------------------------------------------------
# Rope parameter defaults
# ---------------------------------------------------------------------------

class TestRopeParameterDefaults:
    """Verify default values match the paper's specifications."""

    def test_local_rope_theta_default(self):
        """Default local_rope_theta must be 10000.0 — paper §B.RoPE Mechanics (b=10000)."""
        config = ShramConfig()
        assert config.local_rope_theta == 10000.0

    def test_mosrah_rope_theta_default(self):
        """Default mosrah_rope_theta must be 10000.0 — paper §B.RoPE Mechanics (b=10000)."""
        config = ShramConfig()
        assert config.mosrah_rope_theta == 10000.0

    def test_alpha_default(self):
        """Default alpha must be 1.0 — paper §A.2 LLaMA-family recommendation."""
        config = ShramConfig()
        assert config.alpha == 1.0

    def test_beta_default(self):
        """Default beta must be 32.0 — paper §A.2 LLaMA-family recommendation."""
        config = ShramConfig()
        assert config.beta == 32.0

    def test_sequence_lengths_equal_by_default(self):
        """training and inference sequence lengths must default to equal values so scale=1."""
        config = ShramConfig()
        assert config.training_sequence_length == config.inference_sequence_length


# ---------------------------------------------------------------------------
# Scale property
# ---------------------------------------------------------------------------

class TestScaleProperty:
    """scale = inference_sequence_length / training_sequence_length.

    When scale == 1.0, YaRN reduces to standard RoPE. This is the default state.
    """

    def test_scale_one_when_lengths_equal(self):
        """scale must be 1.0 when inference equals training length."""
        config = small_config(training_sequence_length=8192, inference_sequence_length=8192)
        assert config.scale == 1.0

    def test_scale_computed_correctly(self):
        """scale must equal inference / training."""
        config = small_config(training_sequence_length=4096, inference_sequence_length=16384)
        assert config.scale == 4.0

    def test_scale_fractional(self):
        """scale may be non-integer."""
        config = small_config(training_sequence_length=8192, inference_sequence_length=12288)
        assert abs(config.scale - 1.5) < 1e-9

    def test_scale_is_not_stored(self):
        """scale must be a computed property, not a stored field."""
        config = small_config(training_sequence_length=4096, inference_sequence_length=16384)
        assert "scale" not in config.to_dict()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_roundtrip_preserves_all_fields(self):
        """A config serialised to dict and restored must be identical to the original."""
        original = small_config(
            local_rope_theta=500000.0,
            mosrah_rope_theta=200000.0,
            training_sequence_length=4096,
            inference_sequence_length=16384,
            alpha=2.0,
            beta=16.0,
            attention_dropout=0.1,
            use_cache=False,
            num_mosrah_heads=32,
            num_selected_heads=8,
            window_size=64,
            rope_mode="semantic_sequence",
            head_dim=16,
        )
        restored = ShramConfig.from_dict(original.to_dict())

        assert restored.vocab_size == original.vocab_size
        assert restored.hidden_size == original.hidden_size
        assert restored.intermediate_size == original.intermediate_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_sliding_window_heads == original.num_sliding_window_heads
        assert restored.num_mosrah_heads == original.num_mosrah_heads
        assert restored.num_selected_heads == original.num_selected_heads
        assert restored.head_dim == original.head_dim
        assert restored.window_size == original.window_size
        assert restored.rope_mode == original.rope_mode
        assert restored.rms_norm_eps == original.rms_norm_eps
        assert restored.local_rope_theta == original.local_rope_theta
        assert restored.mosrah_rope_theta == original.mosrah_rope_theta
        assert restored.training_sequence_length == original.training_sequence_length
        assert restored.inference_sequence_length == original.inference_sequence_length
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.attention_dropout == original.attention_dropout
        assert restored.use_cache == original.use_cache
        assert restored.output_hidden_states == original.output_hidden_states
        assert restored.tie_word_embeddings == original.tie_word_embeddings
